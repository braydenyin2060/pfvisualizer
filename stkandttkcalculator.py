"""
Phantom Forces STK (Shots-To-Kill) Calculator

Expects a CSV with columns:
    gun_name, category, caliber, fire_mode, rpm, muzzle velocity,
    wall penetration, damage0, range0, damage1, range1,
    headmulti, torsomulti, magazine, tactical reload, empty reload

damage0/range0 is the close-range (max damage) breakpoint, damage1/range1
is the far-range (falloff) breakpoint. Damage is constant at damage0 for
distance <= range0, linearly interpolates down to damage1 between range0
and range1, then stays flat at damage1 beyond range1.
"""

import math
import pandas as pd

DEFAULT_MAX_HEALTH = 100


def load_guns(csv_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Loads the gun stats CSV into a DataFrame. If your file is tab-delimited
    rather than comma-delimited, pass sep='\t' (or sep=None, engine='python'
    to auto-detect).
    """
    return pd.read_csv(csv_path, **read_csv_kwargs)

guns = load_guns("data/guns.csv")

def get_gun(df: pd.DataFrame, gun_name: str) -> pd.Series:
    """
    Looks up a single gun's row by name (case-insensitive), so you don't
    have to write the df[df["gun_name"] == ...].iloc[0] boilerplate every
    time. Raises a clear error if the name doesn't match anything.
    """
    matches = df[df["gun_name"].str.lower() == gun_name.lower()]
    if matches.empty:
        raise ValueError(f"No gun named '{gun_name}' found in the dataframe.")
    return matches.iloc[0]


def lowest_stk(df: pd.DataFrame, max_health: int = DEFAULT_MAX_HEALTH) -> pd.Series:
    """
    Lowest possible STK per gun, assuming every hit is a headshot. Since
    headshot damage is highest at close range (damage0), this is just the
    STK at range0. Returns a Series aligned with df's index — assign it to
    a column, e.g. df["lowest_stk"] = lowest_stk(df).
    """
    best_case_damage = df["damage0"] * df["headmulti"]
    return (max_health / best_case_damage).apply(math.ceil)


def damage_at_range(row: pd.Series, distance: float) -> float:
    """
    Base (unmultiplied) damage at a given distance, for one gun row.
    This is the building block for the later "categorize by STK across
    range" method — not needed for lowest_stk, but you'll want it next.
    """
    d0, r0, d1, r1 = row["damage0"], row["range0"], row["damage1"], row["range1"]
    if distance <= r0:
        return d0
    if distance >= r1:
        return d1
    t = (distance - r0) / (r1 - r0)
    return d0 + t * (d1 - d0)


def hit_profile(row: pd.Series) -> dict:
    """
    Maps hit-location label -> damage multiplier for one gun row.
    Always includes "H" (headshot) and "L" (limbshot, 1.0x baseline).
    Also includes "T" (torso) whenever the gun has a torso multiplier
    that's genuinely different from 1.0 — if torsomulti is missing or
    equal to 1.0, torso damage is identical to limb damage, so there's no
    separate "T" category and hits there are just counted as "L".
    """
    profile = {"H": row["headmulti"], "L": 1.0}
    torso_multi = row.get("torsomulti")
    if pd.notna(torso_multi) and not math.isclose(torso_multi, 1.0):
        profile["T"] = torso_multi
    return profile


def stk_at_range(row: pd.Series, distance: float, max_health: int = DEFAULT_MAX_HEALTH) -> dict:
    """
    All possible STK values at a specific distance, one per hit location,
    assuming every shot in the kill lands on that same location.
    e.g. {"H": 4, "L": 6} means an all-headshot kill takes 4 shots at this
    range, an all-limb kill takes 6 (or {"H": 4, "T": 6} for a gun that has
    a torso multiplier).
    """
    dmg = damage_at_range(row, distance)
    return {
        label: math.ceil(max_health / (dmg * multi))
        for label, multi in hit_profile(row).items()
    }

def stk_range_shotsonly(row: pd.Series, distance: float, max_health: int = DEFAULT_MAX_HEALTH) -> list:
    """
    All achievable STK counts at a specific distance, without the shot-type
    arrangements that produce them — e.g. for the AS Val this returns
    [2, 3, 4] instead of the full "H H", "H T T", "L L L L" breakdown.
    """
    return sorted(shot_combinations(row, distance, max_health).keys())

def _count_splits(total: int, num_labels: int):
    """
    Yields every tuple of non-negative ints of length num_labels that sums
    to total — e.g. _count_splits(3, 2) yields (3,0),(2,1),(1,2),(0,3).
    Used to enumerate every way to split N shots across the hit types.
    """
    if num_labels == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _count_splits(total - first, num_labels - 1):
            yield (first,) + rest

def ttk_seconds(num_shots: int, rpm: float) -> float:
    """
    Time (seconds) to land num_shots shots, given the gun's rate of fire
    in rounds per minute. The first shot lands instantly, so this is just
    the time between shots times (num_shots - 1) — a one-shot kill is
    always a TTK of 0, regardless of RPM.
    """
    if num_shots <= 1:
        return 0.0
    seconds_between_shots = 60.0 / rpm
    return (num_shots - 1) * seconds_between_shots

def shot_combinations(row: pd.Series, distance: float, max_health: int = DEFAULT_MAX_HEALTH) -> dict:
    """
    All minimal shot-type combinations that kill in exactly N shots, for
    every achievable N at this distance. Returns {N: [[label, ...], ...]}.

    A combo of N shots is "minimal" for N if the full set reaches
    max_health, but dropping the single weakest shot in the set would not
    — i.e. you genuinely need all N shots, not just some subset of them.
    Without this check you'd get redundant combos (e.g. "H H H L" showing
    up again with an extra wasted shot at N+1).

    Handles however many hit locations hit_profile() returns for this gun
    (typically "H" + "L", or "H" + "T" + "L" when the gun has a distinct
    torso multiplier) — not hardcoded to exactly two.
    """
    dmg = damage_at_range(row, distance)
    profile = hit_profile(row)
    labels = sorted(profile, key=lambda label: -profile[label])  # highest damage first
    dmgs = [dmg * profile[label] for label in labels]

    min_n = math.ceil(max_health / max(dmgs))
    max_n = math.ceil(max_health / min(dmgs))

    combos = {}
    for n in range(min_n, max_n + 1):
        valid = []
        for counts in _count_splits(n, len(labels)):
            total = sum(count * d for count, d in zip(counts, dmgs))
            if total < max_health:
                continue
            present_damages = [d for count, d in zip(counts, dmgs) if count > 0]
            if total - min(present_damages) < max_health:
                combo = []
                for count, label in zip(counts, labels):
                    combo.extend([label] * count)
                valid.append(combo)
        if valid:
            combos[n] = valid
    return combos

def ttk_seconds(num_shots: int, rpm: float) -> float:
    """
    Time (seconds) to land num_shots shots, given the gun's rate of fire
    in rounds per minute. The first shot lands instantly, so this is just
    the time between shots times (num_shots - 1) — a one-shot kill is
    always a TTK of 0, regardless of RPM.
    """
    if num_shots <= 1:
        return 0.0
    seconds_between_shots = 60.0 / rpm
    return (num_shots - 1) * seconds_between_shots
 
 
def format_combinations(combos: dict, rpm: float = None) -> str:
    """
    Pretty-prints shot_combinations() output as '4SK: H H H H, H H H L'.
    Pass the gun's rpm to include TTK per shot count instead:
    '4SK (0.300 sec): H H H H, H H H L'.
    """
    lines = []
    for n in sorted(combos):
        combo_strs = ", ".join(" ".join(combo) for combo in combos[n])
        if rpm is not None:
            ttk = ttk_seconds(n, rpm)
            lines.append(f"{n}SK ({ttk:.3f} sec): {combo_strs}")
        else:
            lines.append(f"{n}SK: {combo_strs}")
    return "\n".join(lines)

def stk_ttk_at_range(row: pd.Series, distance: float) -> list:
    stks = stk_range_shotsonly(row, distance)

    return [
        (shots, ttk_seconds(shots, row["rpm"]))
        for shots in stks
    ]
def compare_ttk(gun1, gun2, distance):
    ttk1 = min(
        ttk for shots, ttk in stk_ttk_at_range(gun1, distance)
    )

    ttk2 = min(
        ttk for shots, ttk in stk_ttk_at_range(gun2, distance)
    )

    if ttk1 < ttk2:
        return gun1
    elif ttk2 < ttk1:
        return gun2
    else:
        return None

if __name__ == "__main__":
    # Real usage, once you have your actual CSV:
    # guns = load_guns("guns.csv")  # or load_guns("guns.csv", sep="\t") if tab-delimited
    # gun = get_gun(guns, "C7A2")  # case-insensitive lookup by name
    # print(format_combinations(shot_combinations(gun, distance=90)))

    # --- everything below is just an inline demo using fake data, so this
    # --- file runs standalone without a real CSV ---

    for gun_name in guns["gun_name"]:
        gun = get_gun(guns, gun_name)
        print(f"\nShot combinations for {gun_name}")
        print(format_combinations(shot_combinations(gun, 300), rpm=gun["rpm"]))