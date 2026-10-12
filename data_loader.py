import pandas as pd

def load_guns(path="gun_names.csv"):
    return pd.read_csv(path)

def load_fire_modes(path="fire_modes.csv"):
    return pd.read_csv(path)

def load_damage_ranges(path="damage_ranges.csv"):
    return pd.read_csv(path).sort_values(["gun_id", "range_start"])

def load_projectile_profiles(path="projectile_profiles.csv"):
    return pd.read_csv(path)

def load_damage_multis(path="damage_multis.csv"):
    return pd.read_csv(path)

def load_gun_ratings(path="gun_ratings.csv"):
    return pd.read_csv(path)

def load_gun_stats(path="gun_stats.csv"):
    return pd.read_csv(path)

def load_all(folder="."):
    return {
        "gun_names": load_guns(f"{folder}/gun_names.csv"),
        "fire_modes": load_fire_modes(f"{folder}/fire_modes.csv"),
        "damage_ranges": load_damage_ranges(f"{folder}/damage_ranges.csv"),
        "projectile_profiles": load_projectile_profiles(f"{folder}/projectile_profiles.csv"),
        "damage_multis": load_damage_multis(f"{folder}/damage_multis.csv"),
        "gun_ratings": load_gun_ratings(f"{folder}/gun_ratings.csv"),
        "gun_stats": load_gun_stats(f"{folder}/gun_stats.csv"),
    }