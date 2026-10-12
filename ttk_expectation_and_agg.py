"""
Expected time-to-kill (TTK) when each landed hit can be a head (H), torso (T)
or limb (L) hit with some probability.

Inputs come straight from the rest of the pipeline:
    damage  = damage_at_range(...)      # base damage at the range of interest
    profile = hit_profile(...)          # {"H": 1.6, "T": 1.3, "L": 1.0}
    rpm     = get_rpm(...)

A hit distribution is a dict like {"H": 0.10, "T": 0.56, "L": 0.34}. It is
"normalized to landed hits only": misses are NOT in it (see hit_chance).
Percentages (10, 56, 34) work too; everything is re-normalized to sum to 1.

Kill timing depends on the ORDER of hits: H T H kills on hit 3 even though
the set {H, H, T} kills on hit 2 in the order H H T. So a combination's chance
of killing on hit N counts only the orderings whose last hit is the one that
finishes the target (combo_kill_probability). Summed over every combination
that can kill on hit N, this equals the exact kill_probabilities() value.
"""

from collections import Counter
from math import factorial

DEFAULT_MAX_HEALTH = 100
_EPS = 1e-9  # guards float error when damage lands exactly on max_health

# From your archetype table (fractions of LANDED hits).
DISTRIBUTIONS = {
    "sniper":    {"H": 0.26, "T": 0.71, "L": 0.03},
    "dmr":       {"H": 0.25, "T": 0.60, "L": 0.15},
    "automatic": {"H": 0.10, "T": 0.56, "L": 0.34},
}


# --------------------------------------------------------------- distributions
def normalize_distribution(dist: dict) -> dict:
    """Validates a hit distribution and rescales it to sum to 1."""
    if not dist:
        raise ValueError("Hit distribution is empty.")
    if any(p < 0 for p in dist.values()):
        raise ValueError("Hit distribution has a negative probability.")
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("Hit distribution sums to zero.")
    return {label: p / total for label, p in dist.items()}


def align_distribution(dist: dict, labels) -> dict:
    """
    Makes the distribution match the hit types this gun actually has.
    hit_profile() drops "T" when torso damage equals limb damage, so any
    probability on a label the gun doesn't have is folded into "L" (the 1.0x
    baseline). Labels the gun has but the distribution lacks get 0.
    """
    labels = list(labels)
    out = {label: 0.0 for label in labels}
    for label, p in normalize_distribution(dist).items():
        if label in out:
            out[label] += p
        elif "L" in out:
            out["L"] += p
        else:
            raise ValueError(f"Can't place hit type '{label}': gun has {labels} and no 'L'.")
    return out


# ------------------------------------------------------------------ kill odds
def hit_damages(damage: float, profile: dict) -> dict:
    """Damage of one landed hit per type, e.g. {"H": 50.0, "T": 40.6, "L": 31.25}."""
    return {label: damage * mult for label, mult in profile.items()}


def kill_probabilities(hit_damage: dict, dist: dict,
                       max_health: float = DEFAULT_MAX_HEALTH, max_hits: int = 200) -> dict:
    """
    {n: probability the kill lands exactly on the n-th landed hit}. Sums to 1.
    Exact: tracks every possible (heads, torsos, limbs) count after each hit.
    `dist` must already be aligned to the labels in `hit_damage`.
    """
    labels = [label for label, p in dist.items() if p > 0]
    for label in labels:
        if hit_damage[label] <= 0:
            raise ValueError(f"Hit type '{label}' does zero damage, so it never kills.")

    alive = {tuple([0] * len(labels)): 1.0}   # counts per label -> probability
    kill = {}
    for n in range(1, max_hits + 1):
        survivors = {}
        for counts, mass in alive.items():
            for i, label in enumerate(labels):
                new_counts = counts[:i] + (counts[i] + 1,) + counts[i + 1:]
                new_mass = mass * dist[label]
                total = sum(c * hit_damage[l] for c, l in zip(new_counts, labels))
                if total >= max_health - _EPS:
                    kill[n] = kill.get(n, 0.0) + new_mass
                else:
                    survivors[new_counts] = survivors.get(new_counts, 0.0) + new_mass
        alive = survivors
        if not alive:
            return kill
    raise ValueError(f"No kill within {max_hits} hits; check the damage values.")


# ------------------------------------------------------------------------ TTK
def ttk_for_hits(n: int, rpm: float, hit_chance: float = 1.0) -> float:
    """
    Expected seconds to land the n-th hit. The first shot fires at t = 0, and
    reaching n hits takes n / hit_chance shots on average, so
    TTK = (n / hit_chance - 1) * (60 / rpm). With hit_chance = 1 a one-hit
    kill is 0 seconds.
    """
    if rpm <= 0:
        raise ValueError("rpm must be positive.")
    if not 0 < hit_chance <= 1:
        raise ValueError("hit_chance must be in (0, 1].")
    return max(0.0, n / hit_chance - 1.0) * 60.0 / rpm


def expected_ttk(damage: float, profile: dict, rpm: float,
                 distribution: dict = DISTRIBUTIONS["automatic"],
                 max_health: float = DEFAULT_MAX_HEALTH, hit_chance: float = 1.0) -> dict:
    """
    Expected TTK for one gun at one range.

    hit_chance is the chance that a fired shot lands at all (default 1.0 =
    every shot lands, matching a "landed hits only" distribution). Pass the
    miss-adjusted value later to fold misses into time.

    Returns {"expected_ttk": seconds, "expected_hits": float,
             "best_ttk": seconds, "worst_ttk": seconds,
             "by_hits": [(hits_to_kill, probability, ttk_seconds), ...]}.
    best_ttk / worst_ttk are the fastest and slowest kills that can happen at
    all under this distribution (however unlikely). For a tail that ignores
    freak outcomes, use ttk_percentile().
    """
    damages = hit_damages(damage, profile)
    dist = align_distribution(distribution, damages.keys())
    kill = kill_probabilities(damages, dist, max_health)
    by_hits = [(n, p, ttk_for_hits(n, rpm, hit_chance)) for n, p in sorted(kill.items())]
    return {
        "expected_ttk": sum(p * t for _, p, t in by_hits),
        "expected_hits": sum(n * p for n, p, _ in by_hits),
        "best_ttk": by_hits[0][2],
        "worst_ttk": by_hits[-1][2],
        "by_hits": by_hits,
    }


def ttk_percentile(by_hits: list, q: float) -> float:
    """
    TTK that a fraction q of kills are as fast as or faster than, from the
    "by_hits" list. q = 0.1 is a fast-but-realistic kill, q = 0.9 a slow-but-
    realistic one. Steadier than best_ttk / worst_ttk when the extremes are
    very unlikely.
    """
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1].")
    cumulative = 0.0
    for _, p, ttk in by_hits:
        cumulative += p
        if cumulative >= q - 1e-12:
            return ttk
    return by_hits[-1][2]

def expected_damage(
    damage: float,
    profile: dict[str, float],
    distribution: dict[str, float],
) -> float:
    """Calculate expected damage per hit using the hit distribution."""
    damages = hit_damages(damage, profile)
    dist = align_distribution(distribution, damages.keys())

    return sum(
        damages[hit] * probability
        for hit, probability in dist.items()
    )


# -------------------------------------------------- per-combination (display)
def combo_probability(combo, dist: dict) -> float:
    """
    Chance that the first len(combo) landed hits are exactly this set of hit
    types, in any order (multinomial). e.g. ["H", "H", "T"] -> 3 * pH^2 * pT.
    """
    counts = Counter(combo)
    ways = factorial(len(combo))
    for k in counts.values():
        ways //= factorial(k)
    prob = float(ways)
    for label, k in counts.items():
        prob *= dist.get(label, 0.0) ** k
    return prob


def combo_kill_probability(combo, hit_damage: dict, dist: dict,
                           max_health: float = DEFAULT_MAX_HEALTH) -> float:
    """
    Chance that the first len(combo) landed hits are exactly this set of hit
    types AND the kill lands on the last of them. The first N-1 hits are
    everything except the final hit, so an order only counts if the set minus
    its last hit is still below max_health. For each hit type that could be
    last, count the orderings of the other N-1 hits (multinomial).
    """
    counts = Counter(combo)
    n = len(combo)
    total = sum(hit_damage[label] * k for label, k in counts.items())
    if total < max_health - _EPS:
        return 0.0

    orderings = 0
    for last in counts:
        if total - hit_damage[last] < max_health - _EPS:      # others haven't killed yet
            rest = dict(counts)
            rest[last] -= 1
            ways = factorial(n - 1)
            for k in rest.values():
                ways //= factorial(k)
            orderings += ways

    prob = float(orderings)
    for label, k in counts.items():
        prob *= dist.get(label, 0.0) ** k
    return prob


def combos_with_probabilities(combos: dict, distribution: dict = DISTRIBUTIONS["automatic"],
                              *, hit_damage: dict, max_health: float = DEFAULT_MAX_HEALTH) -> dict:
    """
    {N: [(combo, probability), ...]} from shot_combinations() output, where
    probability is the chance that the kill lands on hit N with exactly that
    combination. Per N, the probabilities sum to kill_probabilities()[N], and
    across all N they sum to 1.

    hit_damage is required (damage of one hit per type, from hit_damages()),
    because counting valid orderings needs the actual damage values.
    """
    labels = {label for cbs in combos.values() for cb in cbs for label in cb}
    dist = align_distribution(distribution, labels)
    return {n: [(cb, combo_kill_probability(cb, hit_damage, dist, max_health)) for cb in cbs]
            for n, cbs in combos.items()}


# ---------------------------------------------------------------------- score
# Weights for (expected, slow tail, fast tail). Normalized to sum to 1.
DEFAULT_SCORE_WEIGHTS = {"expected": 0.675, "slow": 0.125, "fast": 0.2}
DEFAULT_SLOWEST_TTK = 0.5   # seconds; a TTK this slow (or slower) scores 0 points


def ttk_to_points(ttk: float, slowest_ttk: float = DEFAULT_SLOWEST_TTK) -> float:
    """
    Linear 0-100 points for one TTK: 0 s -> 100, slowest_ttk or slower -> 0.
    slowest_ttk is a judgment call, so tune it to taste.
    """
    if slowest_ttk <= 0:
        raise ValueError("slowest_ttk must be positive.")
    return 100.0 * max(0.0, 1.0 - ttk / slowest_ttk)


def ttk_score_details(damage: float, profile: dict, rpm: float,
                      distribution: dict = DISTRIBUTIONS["automatic"],
                      weights: dict = DEFAULT_SCORE_WEIGHTS,
                      slowest_ttk: float = DEFAULT_SLOWEST_TTK,
                      slow_q: float = 0.9, fast_q: float = 0.1,
                      max_health: float = DEFAULT_MAX_HEALTH, hit_chance: float = 1.0) -> dict:
    """
    Everything behind ttk_score(): the three TTKs, the points each earned, and
    the final score. The "slow" and "fast" tails are the slow_q / fast_q
    percentiles (default P90 / P10); pass slow_q=1.0 and fast_q=1e-9 to use
    the absolute worst and best case instead.
    """
    w = normalize_distribution(weights)           # same validate-and-rescale logic
    for key in ("expected", "slow", "fast"):
        w.setdefault(key, 0.0)

    r = expected_ttk(damage, profile, rpm, distribution, max_health, hit_chance)
    ttks = {
        "expected": r["expected_ttk"],
        "slow": ttk_percentile(r["by_hits"], slow_q),
        "fast": ttk_percentile(r["by_hits"], fast_q),
    }
    points = {k: ttk_to_points(v, slowest_ttk) for k, v in ttks.items()}
    score = sum(w[k] * points[k] for k in ttks)
    return {"score": score, "ttk": ttks, "points": points, "weights": w}


def ttk_score(damage: float, profile: dict, rpm: float, **kwargs) -> float:
    """
    Aggregate TTK performance, 0-100 (higher is better). A gun that kills in
    one hit every time has TTK 0 and scores exactly 100. Takes the same
    arguments as ttk_score_details().

    NOTE: one-hit kills always score 100 whatever the rpm or range, so this
    rates snipers very generously.
    """
    return ttk_score_details(damage, profile, rpm, **kwargs)["score"]


# --------------------------------------------------------- run on real guns
def _load_calculator():
    """Your calculator module (renamed copy first, then the original file name)."""
    try:
        import stkandttkcalculator as calculator
    except ImportError:
        import stk_calculator as calculator
    return calculator


def score_guns(data: dict, category: str, distances, distribution: dict, calculator) -> tuple:
    """
    Scores every gun in `category` (matched against the guns table's
    "category" column, ignoring case and spaces) at each distance.
    Returns (rows, skipped): rows sorted best average score first; skipped
    lists (gun_name, reason) for guns with missing data, e.g. no multipliers.
    """
    needed = ("gun_names", "damage_ranges", "damage_multis", "fire_modes")
    absent = [key for key in needed if key not in data]
    if absent:
        raise KeyError(f"data has no {absent}; it contains {list(data.keys())}. "
                       "Change the key names in score_guns() to match load_all().")

    guns = data["gun_names"]
    chosen = guns[guns["category"].astype(str).str.strip().str.lower() == category.strip().lower()]
    if chosen.empty:
        found = sorted(guns["category"].dropna().astype(str).unique())
        raise ValueError(f"No guns with category '{category}'. Categories found: {found}")

    rows, skipped = [], []
    for _, gun in chosen.iterrows():
        gun_id = gun["gun_id"]
        try:
            profile = calculator.hit_profile(data["damage_multis"], gun_id)
            rpm = calculator.get_rpm(data["fire_modes"], gun_id)
            results = {}
            for distance in distances:
                damage = calculator.damage_at_range(data["damage_ranges"], gun_id, distance)
                details = ttk_score_details(damage, profile, rpm, distribution)
                results[distance] = {"score": details["score"], "ttk": details["ttk"]["expected"]}
        except (ValueError, KeyError) as error:
            skipped.append((gun["gun_name"], str(error)))
            continue
        average = sum(r["score"] for r in results.values()) / len(results)
        rows.append({"gun_name": gun["gun_name"], "results": results, "average": average})

    rows.sort(key=lambda row: row["average"], reverse=True)
    return rows, skipped


def main():
    import data_loader

    CATEGORY = "Assault Rifle"       # must match a value in guns.csv's category column
    DISTANCES = (20, 50, 100, 150, 200)        # studs
    DISTRIBUTION = DISTRIBUTIONS["automatic"]

    data = data_loader.load_all("data")
    rows, skipped = score_guns(data, CATEGORY, DISTANCES, DISTRIBUTION, _load_calculator())

    rows.sort(key=lambda row: row["results"][20]["score"], reverse=True)

    print(f"{CATEGORY}s: TTK score 0-100 (expected TTK) at each distance\n")
    print(f"{'Gun':<22}" + "".join(f"{str(d) + ' studs':>20}" for d in DISTANCES) + f"{'average':>10}")
    for row in rows:
        cells = "".join(
            f"{row['results'][d]['score']:>9.2f} ({row['results'][d]['ttk']:.3f}s)" for d in DISTANCES
        )
        print(f"{row['gun_name']:<22}{cells}{row['average']:>10.2f}")

    if skipped:
        print("\nSkipped (missing data):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    


if __name__ == "__main__":
    main()