"""
TLE Loader Utility

Loads real satellite orbital data from TLE files.
Supports Starlink, OneWeb, and other catalogs.
"""

from skyfield.api import EarthSatellite


def load_tle_file(file_path):
    """
    Load satellites from a TLE file.

    Args:
        file_path (str): Path to TLE file

    Returns:
        list: List of EarthSatellite objects
    """
    satellites = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        try:
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            sat = EarthSatellite(line1, line2, name)
            satellites.append(sat)

        except Exception as e:
            print(f"Skipping invalid TLE at index {i}: {e}")
            continue

    return satellites


def load_all_satellites():
    """
    Load all available satellite groups.

    Returns:
        dict: Satellite groups
    """
    data_paths = {
        "starlink": "data/raw/tle/starlink.txt",
        "oneweb": "data/raw/tle/oneweb.txt",
        "active": "data/raw/tle/active_satellites.txt"
    }

    all_sats = {}

    for key, path in data_paths.items():
        try:
            sats = load_tle_file(path)
            all_sats[key] = sats
            print(f"Loaded {len(sats)} satellites from {key}")
        except FileNotFoundError:
            print(f"Warning: {path} not found")

    return all_sats