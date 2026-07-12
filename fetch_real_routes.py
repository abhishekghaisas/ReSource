"""
One-time fetch of real road-network driving distances and route geometry
for the Palisades scenario's 5 real locations, from OSRM's public demo
server (no API key, no billing -- see FINDINGS.md "Real Road Routing" for
why this was chosen over Google Maps Directions).

Run this once, with your own internet access (this environment has none):
    python fetch_real_routes.py

Produces real_routes_cache.json, which run_palisades_scenario.py loads via
--real-routes-cache to replace straight-line distance with real driving
distance for TRAVEL TIME purposes only. Fire risk/blocking intentionally
keeps using straight-line proximity to the epicenter -- a real fire
doesn't spread along roads, so that part was never the accuracy gap being
closed here.

Note: OSRM's public server is meant for evaluation, not production load.
This script makes exactly 7 requests, once, with a short delay between
each -- reasonable, respectful use, not something to run in a loop or a
simulation's inner loop.
"""

import json
import time
import urllib.request
import urllib.error

# (lat, lon) -- same approximate real coordinates used in the dashboard
# and run_palisades_scenario.py's original research.
REAL_LATLNG = {
    "S1": (34.052, -118.444),  # Westwood Recreation Center
    "S2": (34.182, -118.610),  # El Camino Real Charter High School
    "S3": (34.144, -118.144),  # Pasadena Convention Center
    "D1": (34.039, -118.428),  # UCLA Research Park West
    "D2": (34.036, -118.678),  # Malibu Pier staging area
}

# Only the 7 pairs the simulation actually uses (6 depot-shelter + 1
# depot-depot) -- shelter-to-shelter isn't a route anything in the system
# queries, no need to fetch it.
PAIRS = [
    ("D1", "S1"), ("D1", "S2"), ("D1", "S3"),
    ("D2", "S1"), ("D2", "S2"), ("D2", "S3"),
    ("D1", "D2"),
]

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"


def fetch_route(a_id: str, b_id: str) -> dict:
    lat1, lon1 = REAL_LATLNG[a_id]
    lat2, lon2 = REAL_LATLNG[b_id]
    url = f"{OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM returned no route for {a_id}-{b_id}: {data.get('code')}")
    route = data["routes"][0]
    return {
        "distance_m": route["distance"],
        "duration_s": route["duration"],
        # GeoJSON coordinates are [lon, lat] -- flip to [lat, lon] to match
        # the [lat, lon] convention used everywhere else in this project
        # (REAL_LATLNG above, the dashboard's REAL_LATLNG constant).
        "geometry_latlon": [[lat, lon] for lon, lat in route["geometry"]["coordinates"]],
    }


def main():
    cache = {"locations": {k: list(v) for k, v in REAL_LATLNG.items()}, "routes": {}}
    for a_id, b_id in PAIRS:
        key = f"{a_id}-{b_id}"
        print(f"Fetching {key} ({REAL_LATLNG[a_id]} -> {REAL_LATLNG[b_id]})...")
        try:
            result = fetch_route(a_id, b_id)
            cache["routes"][key] = result
            print(f"  distance={result['distance_m']/1609.34:.2f} mi  "
                  f"duration={result['duration_s']/60:.1f} min  "
                  f"geometry_points={len(result['geometry_latlon'])}")
        except (urllib.error.URLError, RuntimeError) as e:
            print(f"  FAILED: {e}")
            cache["routes"][key] = None
        time.sleep(1.0)  # be polite to the public demo server

    with open("real_routes_cache.json", "w") as f:
        json.dump(cache, f, indent=2)

    n_ok = sum(1 for v in cache["routes"].values() if v is not None)
    print(f"\nWrote real_routes_cache.json: {n_ok}/{len(PAIRS)} routes fetched successfully.")
    if n_ok < len(PAIRS):
        print("Some routes failed -- run_palisades_scenario.py will fall back to "
              "straight-line distance for those specific pairs, not fail entirely.")


if __name__ == "__main__":
    main()