"""
Phantom Forces STK (Shots-To-Kill) Calculator

Data lives in separate CSVs loaded by data_loader.py. This module is pure
math: it never reads files, it takes the tables (or plain numbers) as
arguments.

damage_ranges table: one row per falloff segment per gun, with columns
    gun_id, range_id, range_start, range_end, damage_start, damage_end
Within a segment, damage interpolates linearly from damage_start to
damage_end.

multipliers table: one row per gun, with columns
    gun_id, head_multi, torso_multi
"""

import math
import pandas as pd

DEFAULT_MAX_HEALTH = 100
_EPS = 1e-9   # guards float error when damage lands exactly on max_health


def _simplify(name) -> str:
    """Lowercase letters and digits only, so "VHS-D2", "vhs d2" and "VHSD2" all match."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def get_gun(df: pd.DataFrame, gun_name: str) -> pd.Series:
    """
    Looks up a single gun's row by name, ignoring case, spaces and dashes.
    Raises a clear error if the name doesn't match anything.
    """
    matches = df[df["gun_name"].map(_simplify) == _simplify(gun_name)]
    if matches.empty:
        raise ValueError(f"No gun named '{gun_name}' found in the dataframe.")
    return matches.iloc[0]


def damage_at_range(damage_ranges: pd.DataFrame, gun_id, distance: float) -> float:
    """
    Base (unmultiplied) damage for one gun at a given distance, using that
    gun's rows in the damage_ranges table.

    - Before the first segment starts: damage_start of the first segment.
    - Inside a segment (range_start <= distance < range_end): linear
      interpolation between damage_start and damage_end.
    - In a gap between segments: damage_end of the previous segment.
    - Past the last segment: damage_end of the last segment.
    - A segment with a missing/infinite range_end is treated as flat at
      damage_start.
    """
    segments = damage_ranges[damage_ranges["gun_id"] == gun_id].sort_values("range_start")
    if segments.empty:
        raise ValueError(f"No damage ranges found for gun_id {gun_id}.")

    first = segments.iloc[0]
    if distance <= first["range_start"]:
        return float(first["damage_start"])

    previous_end_damage = float(first["damage_start"])
    for seg in segments.itertuples():
        start, end = seg.range_start, seg.range_end
        if distance < start:  # fell in a gap between segments
            return previous_end_damage
        if pd.isna(end) or math.isinf(end):
            return float(seg.damage_start)
        if distance < end:
            if end == start:
                return float(seg.damage_start)
            t = (distance - start) / (end - start)
            return float(seg.damage_start + t * (seg.damage_end - seg.damage_start))
        previous_end_damage = float(seg.damage_end)
    return previous_end_damage


def lowest_stk(damage_ranges: pd.DataFrame, multipliers: pd.DataFrame, gun_id,
               max_health: int = DEFAULT_MAX_HEALTH) -> int:
    """
    Lowest possible STK for one gun, assuming every hit is a headshot at
    the gun's highest-damage point across all its segments.
    """
    segs = damage_ranges[damage_ranges["gun_id"] == gun_id]
    best_damage = max(segs["damage_start"].max(), segs["damage_end"].max())
    head_multi = hit_profile(multipliers, gun_id)["H"]
    return math.ceil(max_health / (best_damage * head_multi))


def hit_profile(multipliers: pd.DataFrame, gun_id) -> dict:
    """
    Maps hit-location label -> damage multiplier for one gun, using that
    gun's row in the multipliers table (columns: gun_id, head_multi,
    torso_multi).

    Always includes "H" (headshot) and "L" (limbshot, 1.0x baseline).
    Also includes "T" (torso) whenever the gun has a torso multiplier
    that's genuinely different from 1.0 — if torso_multi is missing or
    equal to 1.0, torso damage is identical to limb damage, so there's no
    separate "T" category and hits there are just counted as "L".
    """
    rows = multipliers[multipliers["gun_id"] == gun_id]
    if rows.empty:
        raise ValueError(f"No multipliers found for gun_id {gun_id}.")
    row = rows.iloc[0]

    profile = {"H": float(row["head_multi"]), "L": 1.0}
    torso_multi = row["torso_multi"]
    if pd.notna(torso_multi) and not math.isclose(torso_multi, 1.0):
        profile["T"] = float(torso_multi)
    return profile


def stk_at_range(damage: float, profile: dict, max_health: int = DEFAULT_MAX_HEALTH) -> dict:
    """
    All possible STK values at a specific distance, one per hit location,
    assuming every shot in the kill lands on that same location.
    e.g. {"H": 4, "L": 6} means an all-headshot kill takes 4 shots at this
    range, an all-limb kill takes 6 (or {"H": 4, "T": 6} for a gun that has
    a torso multiplier).

    damage is the base damage at the distance you care about, i.e. the
    output of damage_at_range(); profile is the output of hit_profile().
    """
    return {
        label: math.ceil(max_health / (damage * multi))
        for label, multi in profile.items()
    }


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


def shot_combinations(damage: float, profile: dict, max_health: int = DEFAULT_MAX_HEALTH) -> dict:
    """
    Every shot-type combination that can kill on exactly shot N, for every
    achievable N at this distance. Returns {N: [[label, ...], ...]}.

    A set of N hits counts for N if the full set reaches max_health, but
    firing the strongest hit LAST, the other N-1 hits would not have killed
    yet. Order matters: the same set can kill earlier in another order, e.g.
    "H H T" kills on shot 3 as H, T, H but on shot 2 as H, H, T. So a set
    appears under N whenever at least one order kills on shot N.

    This list is complete, but a set's chance of happening depends on how
    many of its orderings actually kill on shot N (see
    ttk_expectation.combos_with_probabilities).

    Handles however many hit locations hit_profile() returns for this gun
    (typically "H" + "L", or "H" + "T" + "L" when the gun has a distinct
    torso multiplier) — not hardcoded to exactly two.
    """
    dmg = damage
    labels = sorted(profile, key=lambda label: -profile[label])  # highest damage first
    dmgs = [dmg * profile[label] for label in labels]

    min_n = math.ceil(max_health / max(dmgs))
    max_n = math.ceil(max_health / min(dmgs))

    combos = {}
    for n in range(min_n, max_n + 1):
        valid = []
        for counts in _count_splits(n, len(labels)):
            total = sum(count * d for count, d in zip(counts, dmgs))
            if total < max_health - _EPS:
                continue
            present_damages = [d for count, d in zip(counts, dmgs) if count > 0]
            # strongest hit last: the first N-1 hits must still be alive
            if total - max(present_damages) < max_health - _EPS:
                combo = []
                for count, label in zip(counts, labels):
                    combo.extend([label] * count)
                valid.append(combo)
        if valid:
            combos[n] = valid
    return combos


def stk_range_shotsonly(damage: float, profile: dict, max_health: int = DEFAULT_MAX_HEALTH) -> list:
    """
    All achievable STK counts at a specific distance, without the shot-type
    arrangements that produce them — e.g. for the AS Val this returns
    [2, 3, 4] instead of the full "H H", "H T T", "L L L L" breakdown.
    """
    return sorted(shot_combinations(damage, profile, max_health).keys())


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


def get_rpm(fire_modes: pd.DataFrame, gun_id, fire_mode: str = None) -> float:
    """
    RPM for one gun from the fire_modes table. A gun can have several rows
    (one per fire mode); pass fire_mode (e.g. "auto") to pick one, otherwise
    the first row for that gun is used.
    """
    rows = fire_modes[fire_modes["gun_id"] == gun_id]
    if fire_mode is not None:
        rows = rows[rows["fire_mode"].str.lower() == fire_mode.lower()]
    if rows.empty:
        raise ValueError(f"No fire mode found for gun_id {gun_id}.")
    return float(rows.iloc[0]["start_rpm"])


def stk_ttk_at_range(data: dict, gun_id, distance: float, max_health: int = DEFAULT_MAX_HEALTH) -> list:
    """
    [(shots, ttk_seconds), ...] for every achievable STK at this distance,
    e.g. [(2, 0.085), (3, 0.170), (4, 0.255)].
    """
    dmg = damage_at_range(data["damage_ranges"], gun_id, distance)
    profile = hit_profile(data["multipliers"], gun_id)
    rpm = get_rpm(data["fire_modes"], gun_id)
    return [(n, ttk_seconds(n, rpm)) for n in stk_range_shotsonly(dmg, profile, max_health)]


def compare_ttk(data: dict, gun_id_a, gun_id_b, distance: float, max_health: int = DEFAULT_MAX_HEALTH):
    """
    Which gun has the faster best-case (all-headshot) TTK at this distance.
    Returns the winning gun_id, or None on a tie.
    """
    best_a = stk_ttk_at_range(data, gun_id_a, distance, max_health)[0][1]
    best_b = stk_ttk_at_range(data, gun_id_b, distance, max_health)[0][1]
    if math.isclose(best_a, best_b):
        return None
    return gun_id_a if best_a < best_b else gun_id_b


if __name__ == "__main__":
    import data_loader

    GUN_NAMES = ["M416", "VHS-D2"]        # matched ignoring case, spaces and dashes
    STK_DISTANCES = [0, 15, 50, 90, 150]   # studs
    COMBO_DISTANCE = 50                    # studs, for the combination breakdown

    data = data_loader.load_all("data")
    guns = data["gun_names"]

    for name in GUN_NAMES:
        gun = get_gun(guns, name)
        gun_id = gun["gun_id"]
        profile = hit_profile(data["damage_multis"], gun_id)
        rpm = get_rpm(data["fire_modes"], gun_id)   # first fire mode listed for this gun

        print("=" * 60)
        print(f"{gun['gun_name']} (gun_id {gun_id}, {rpm:.0f} rpm)")

        for dist in STK_DISTANCES:
            dmg = damage_at_range(data["damage_ranges"], gun_id, dist)
            print(f"  @ {dist:>3} studs: dmg={dmg:.2f}, STK={stk_at_range(dmg, profile)}")

        dmg = damage_at_range(data["damage_ranges"], gun_id, COMBO_DISTANCE)
        print(f"\n  Combinations with TTK @ {COMBO_DISTANCE} studs:")
        print("  " + format_combinations(shot_combinations(dmg, profile), rpm=rpm).replace("\n", "\n  "))
        print("  shots only:", stk_range_shotsonly(dmg, profile))
        print("  lowest_stk:", lowest_stk(data["damage_ranges"], data["damage_multis"], gun_id))