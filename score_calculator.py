import ttk_expectation_and_agg as ttkcalc


WEIGHTS_BY_DISTANCE = {
    50:  {"ttk": 65, "recoil": 1, "handling": 15, "uptime": 5, "wall_pen": 10, "muzzle_velocity": 0, "raw_damage": 4},
    100:  {"ttk": 60, "recoil": 6, "handling": 12.5, "uptime": 7.5, "wall_pen": 10, "muzzle_velocity": 1.5, "raw_damage": 3},
    150: {"ttk": 45, "recoil": 15, "handling": 10, "uptime": 10, "wall_pen": 10, "muzzle_velocity": 9, "raw_damage": 1},
    200: {"ttk": 35, "recoil": 22.5, "handling": 5, "uptime": 15, "wall_pen": 10, "muzzle_velocity": 12.5, "raw_damage": 1},
    250: {"ttk": 25, "recoil": 30, "handling": 2, "uptime": 20, "wall_pen": 10, "muzzle_velocity": 17, "raw_damage": 1},
}


def recoil_score(rating: float) -> float:
    """Convert a 1–5 recoil rating into a 0–100 score."""
    if not 1 <= rating <= 5:
        raise ValueError("Recoil rating must be between 1 and 5")

    if rating < 2.5:
        return (5 + (rating - 1.0) * 10) * 1.25
    elif rating < 3.0:
        return (20 + (rating - 2.5) * 20) * 1.25
    elif rating < 3.5:
        return (30 + (rating - 3.0) * 30) * 1.25
    elif rating < 4.0:
        return (45 + (rating - 3.5) * 30) * 1.25
    elif rating < 4.5:
        return (60 + (rating - 4.0) * 20) * 1.25
    else:
        return (70 + (rating - 4.5) * 20) * 1.25

def handling_score(rating: float) -> float:
    """Convert a 1–5 handling rating into a 0–100 score."""
    if not 1 <= rating <= 5:
        raise ValueError("Handling rating must be between 1 and 5")

    if rating < 2.0:
        return (rating - 1.0) * 20
    elif rating < 2.5:
        return 20 + (rating - 2.0) * 30
    elif rating < 3.0:
        return 35 + (rating - 2.5) * 30
    elif rating < 3.5:
        return 50 + (rating - 3.0) * 40
    else:
        return 70 + (rating - 3.5) * 20

def wall_pen_score(pen: float) -> float:
    """Convert wall penetration into a top-heavy 0–100 score."""
    if pen < 0:
        raise ValueError("Wall penetration cannot be negative")
    elif pen < 0.25:
        return 0
    elif pen < 1.0:
        return 20 + 40 * (pen - 0.25) / 0.75
    elif pen < 1.1:
        return 60 + 15 * (pen - 1.0) / 0.1
    elif pen < 1.2:
        return 75 + 10 * (pen - 1.1) / 0.1
    else:
        x = min(pen, 2.0)
        return 85 + 15 * ((x - 1.2) / 0.8) ** 0.5

def raw_damage_score(expected_damage: float) -> float:
    """Convert expected damage per hit into a 0–100 score."""
    if expected_damage < 0:
        raise ValueError("Expected damage cannot be negative")

    return min(100.0, expected_damage)

def muzzle_velocity_score(mv: float) -> float:
    """Convert muzzle velocity (studs/s) into a 0–100 score."""
    if mv < 0:
        raise ValueError("Muzzle velocity cannot be negative")
    elif mv < 700:
        return 10 * mv / 700
    elif mv < 1000:
        return 10 + 20 * (mv - 700) / 300
    elif mv < 1500:
        return 30 + 20 * (mv - 1000) / 500
    elif mv < 2000:
        return 50 + 15 * (mv - 1500) / 500
    elif mv < 2500:
        return 65 + 10 * (mv - 2000) / 500
    elif mv < 3000:
        return 75 + 15 * (mv - 2500) / 500
    elif mv < 3500:
        return 90 + 10 * (mv - 3000) / 500
    else:
        return 100


def calculate_uptime(
    mag_size: int,
    rpm: float,
    tactical_reload: float,
    empty_reload: float,
) -> float:
    """Calculate sustained-fire uptime as a fraction from 0 to 1."""
    if mag_size < 1:
        raise ValueError("Magazine size must be at least 1")
    if rpm <= 0:
        raise ValueError("RPM must be positive")
    if tactical_reload < 0 or empty_reload < 0:
        raise ValueError("Reload times cannot be negative")

    firing_time = (mag_size - 1) * 60 / rpm
    reload_time = 0.8 * tactical_reload + 0.2 * empty_reload
    cycle_time = firing_time + reload_time

    if cycle_time == 0:
        return 1.0

    return firing_time / cycle_time


def uptime_score(uptime: float) -> float:
    """Convert actual uptime (0–1) into a 0–100 score."""
    if not 0 <= uptime <= 1:
        raise ValueError("Uptime must be between 0 and 1")

    if uptime <= 0.20:
        return 0.0
    elif uptime < 0.35:
        return 30 * (uptime - 0.20) / 0.15
    elif uptime < 0.45:
        return 30 + 30 * (uptime - 0.35) / 0.10
    elif uptime < 0.55:
        return 60 + 10 * (uptime - 0.45) / 0.10
    elif uptime < 0.65:
        return 70 + 20 * (uptime - 0.55) / 0.10
    elif uptime < 0.70:
        return 90 + 5 * (uptime - 0.65) / 0.05
    elif uptime < 0.80:
        return 95 + 5 * (uptime - 0.70) / 0.10
    else:
        return 100.0

def aggregate_score(
    category_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Calculate a weighted average of category scores (0–100)."""
    if not weights:
        raise ValueError("At least one category weight is required")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Weights cannot be negative")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    missing = set(weights) - set(category_scores)
    if missing:
        raise ValueError(f"Missing category scores: {sorted(missing)}")

    for category in weights:
        if not 0 <= category_scores[category] <= 100:
            raise ValueError(
                f"{category} score must be between 0 and 100"
            )

    return sum(
        category_scores[category] * weight
        for category, weight in weights.items()
    ) / total_weight

 
 
def weights_at_distance(distance: float, anchors: dict) -> dict:
    """
    Weights for one distance, blended linearly between the anchor distances
    you defined. Closer than the first anchor or farther than the last, it
    uses that end's weights unchanged. Each anchor is normalized to sum to 1
    before blending, so anchors don't need to use the same scale.
    """
    if not anchors:
        raise ValueError("At least one distance anchor is required")
    points = sorted(anchors.items())
    categories = set(points[0][1])
 
    normalized = []
    for anchor_distance, weights in points:
        if set(weights) != categories:
            raise ValueError(f"Anchor {anchor_distance} has different categories than the first anchor")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"Anchor {anchor_distance} needs non-negative weights with a positive total")
        total = sum(weights.values())
        normalized.append((anchor_distance, {k: w / total for k, w in weights.items()}))
 
    if distance <= normalized[0][0]:
        return dict(normalized[0][1])
    if distance >= normalized[-1][0]:
        return dict(normalized[-1][1])
    for (d0, w0), (d1, w1) in zip(normalized, normalized[1:]):
        if d0 <= distance <= d1:
            t = (distance - d0) / (d1 - d0)
            return {k: w0[k] * (1 - t) + w1[k] * t for k in w0}


def score_weapon(gun, distance, distribution, weights):
    """Calculate all category scores and the overall weapon score."""

    # TTK score — using your existing TTK calculation
    ttk = ttkcalc.ttk_score(
    gun["damage"],
    gun["damage_profile"],
    gun["rpm"],
    distribution=distribution,
    )

    # Other category scores
    recoil = recoil_score(gun["recoil_rating"])
    handling = handling_score(gun["handling_rating"])

    uptime = uptime_score(
        calculate_uptime(
            gun["mag_size"],
            gun["rpm"],
            gun["tactical_reload"],
            gun["empty_reload"],
        )
    )

    wall_pen = wall_pen_score(gun["wall_pen"])
    muzzle_velocity = muzzle_velocity_score(gun["muzzle_velocity"])

    expected_dmg = ttkcalc.expected_damage(
        gun["damage"],
        gun["damage_profile"],
        distribution,
    )
    raw_damage = raw_damage_score(expected_dmg)

    categories = {
        "ttk": ttk,
        "recoil": recoil,
        "handling": handling,
        "uptime": uptime,
        "wall_pen": wall_pen,
        "muzzle_velocity": muzzle_velocity,
        "raw_damage": raw_damage,
    }

    return {
        "score": aggregate_score(categories, weights),
        "categories": categories,
    }




TABLE_KEYS = {
    "gun_names": "gun_names",
    "damage_ranges": "damage_ranges",
    "damage_multis": "damage_multis",
    "fire_modes": "fire_modes",
    "gun_ratings": "gun_ratings",
    "gun_stats": "gun_stats",
}
 
# Stats that score_weapon needs beyond damage / damage_profile / rpm (those
# three come from damage_ranges, the multipliers table and fire_modes).
# Format: gun-dict key -> (table, column). Each table is looked up by gun_id.
STAT_SOURCES = {
    "recoil_rating":   ("gun_ratings", "recoil_rating"),
    "handling_rating": ("gun_ratings", "handling_rating"),
    "mag_size":        ("gun_stats", "ammo_capacity"),
    "tactical_reload": ("gun_stats", "tactical_reload"),
    "empty_reload":    ("gun_stats", "empty_reload"),
    "wall_pen":        ("gun_stats", "wall_pen"),
    "muzzle_velocity": ("gun_stats", "muzzle_velocity"),
}
 
CATEGORY_ORDER = ["ttk", "recoil", "handling", "uptime", "wall_pen", "muzzle_velocity", "raw_damage"]
 
 
def _load_calculator():
    """Your STK calculator module (renamed copy first, then the original name)."""
    try:
        import stkandttkcalculator as calculator
    except ImportError:
        import stkandttkcalculator as calculator
    return calculator
 
 
def _check_columns(data):
    """Fails once, up front, if STAT_SOURCES points at a column that doesn't exist."""
    problems = []
    for stat, (table_key, column) in STAT_SOURCES.items():
        table = data[TABLE_KEYS[table_key]]
        if column not in table.columns:
            problems.append(f"  {stat}: no column '{column}' in {table_key} (columns: {list(table.columns)})")
    if problems:
        raise ValueError("Fix STAT_SOURCES at the top of the test-run section:\n" + "\n".join(problems))
 
 
def _lookup_stat(data, table_key, column, gun_id):
    import pandas as pd
 
    table = data[TABLE_KEYS[table_key]]
    rows = table[table["gun_id"] == gun_id]
    if rows.empty:
        raise ValueError(f"no row for gun_id {gun_id} in {table_key}")
    value = rows.iloc[0][column]
    if pd.isna(value):
        raise ValueError(f"'{column}' is blank")
    return value
 
 
def build_gun(data, calculator, gun_id, distance):
    """The dict score_weapon expects, with damage evaluated at `distance`."""
    gun = {
        "damage": calculator.damage_at_range(data[TABLE_KEYS["damage_ranges"]], gun_id, distance),
        "damage_profile": calculator.hit_profile(data[TABLE_KEYS["damage_multis"]], gun_id),
        "rpm": calculator.get_rpm(data[TABLE_KEYS["fire_modes"]], gun_id),
    }
    for stat, (table_key, column) in STAT_SOURCES.items():
        gun[stat] = _lookup_stat(data, table_key, column, gun_id)
    gun["mag_size"] = int(gun["mag_size"])
    return gun
 
 
def main():
    import data_loader
 
    CATEGORY = "Assault Rifle"      # must match a value in the guns table's category column
    DISTANCES = (20, 50, 100, 150, 175, 200, 250, 500)       # studs
    DISTRIBUTION = ttkcalc.DISTRIBUTIONS["automatic"]
    # Placeholder: equal weights. Replace with your real ones.
    for distance in DISTANCES:
        weights = weights_at_distance(distance, WEIGHTS_BY_DISTANCE)
 
    data = data_loader.load_all("data")
    calculator = _load_calculator()
    _check_columns(data)
 
    guns = data[TABLE_KEYS["gun_names"]]
    chosen = guns[guns["category"].astype(str).str.strip().str.lower() == CATEGORY.strip().lower()]
    if chosen.empty:
        found = sorted(guns["category"].dropna().astype(str).unique())
        raise ValueError(f"No guns with category '{CATEGORY}'. Categories found: {found}")
 
    rows, skipped = [], []
    for _, gun_row in chosen.iterrows():
        try:
            results = {}

            for distance in DISTANCES:
                gun = build_gun(
                    data, calculator, gun_row["gun_id"], distance
                )

                weights = weights_at_distance(
                    distance, WEIGHTS_BY_DISTANCE
                )

                results[distance] = score_weapon(
                    gun, distance, DISTRIBUTION, weights
                )

        except (ValueError, KeyError) as error:
            skipped.append((gun_row["gun_name"], str(error)))
            continue

        average = sum(
            r["score"] for r in results.values()
        ) / len(results)

        rows.append({
            "name": gun_row["gun_name"],
            "results": results,
            "average": average,
        })
 
    rows.sort(key=lambda row: row["average"], reverse=True)
 
    print(f"{CATEGORY}s: overall score (0-100)\n")
    print(f"{'Gun':<22}" + "".join(f"{str(d) + ' studs':>12}" for d in DISTANCES) + f"{'average':>10}")
    for row in rows:
        cells = "".join(f"{row['results'][d]['score']:>12.1f}" for d in DISTANCES)
        print(f"{row['name']:<22}{cells}{row['average']:>10.1f}")
 
    detail = DISTANCES[len(DISTANCES) // 2]
    print(f"\nCategory scores at {detail} studs\n")
    print(f"{'Gun':<22}" + "".join(f"{c:>16}" for c in CATEGORY_ORDER))
    for row in rows:
        cats = row["results"][detail]["categories"]
        print(f"{row['name']:<22}" + "".join(f"{cats[c]:>16.1f}" for c in CATEGORY_ORDER))
 
    if skipped:
        print("\nSkipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
 
 
if __name__ == "__main__":
    main()