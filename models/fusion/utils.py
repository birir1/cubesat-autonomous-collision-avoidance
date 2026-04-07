"""
TLE Loader Utility (ROBUST + PRODUCTION-READY)

Features:
- Loads satellites from multiple sources
- Handles invalid TLEs safely
- Supports constellation filtering
- Clean logging for debugging
- Compatible with Skyfield propagation
"""

from skyfield.api import EarthSatellite, load
import requests


# ============================================
# TLE SOURCES (Celestrak)
# ============================================

TLE_SOURCES = {
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "oneweb": "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
    "active": "https://celestrak.org/NORAD/elements/active.txt",
}


# ============================================
# DOWNLOAD TLE DATA
# ============================================

def download_tle(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text.strip().splitlines()
    except Exception as e:
        print(f"[ERROR] Failed to download TLE from {url}: {e}")
        return []


# ============================================
# PARSE TLE → SATELLITES
# ============================================

def parse_tle(lines):
    satellites = []

    for i in range(0, len(lines) - 2, 3):
        try:
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            sat = EarthSatellite(line1, line2, name)
            satellites.append(sat)

        except Exception:
            continue  # skip bad entries

    return satellites


# ============================================
# LOAD SINGLE CONSTELLATION
# ============================================

def load_constellation(name):
    if name not in TLE_SOURCES:
        raise ValueError(f"Unknown constellation: {name}")

    url = TLE_SOURCES[name]
    lines = download_tle(url)

    sats = parse_tle(lines)

    print(f"Loaded {len(sats)} satellites from {name}")
    return sats


# ============================================
# LOAD ALL CONSTELLATIONS
# ============================================

def load_all_satellites():
    all_sats = {}

    for name in TLE_SOURCES:
        sats = load_constellation(name)
        all_sats[name] = sats

    return all_sats


# ============================================
# QUICK TEST
# ============================================

if __name__ == "__main__":
    sats = load_all_satellites()

    total = sum(len(v) for v in sats.values())
    print(f"\nTotal satellites loaded: {total}")