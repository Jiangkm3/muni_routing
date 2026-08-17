#!/usr/bin/env python3
"""
Analyze San Francisco Muni route alignments by street name.

Data sources
------------
1. 511 SF Bay active SFMTA GTFS feed (operator_id=SF) for routes, trips,
   calendars, and shapes.
2. OpenStreetMap, downloaded through OSMnx/Overpass, for street centerlines
   and names.

The calculation is directional: each GTFS route is split by direction_id, so
outbound/inbound or eastbound/westbound alignments are analyzed independently.
When the 511 GTFS+ directions.txt file is present, its official direction name
is included. The script also infers the dominant first and last stops for each
direction from stop_times.txt as a human-readable terminal pair.

The default calculation is *scheduled-service weighted*: each distinct GTFS
shape/service pattern is weighted by its share of scheduled trip occurrences for
ONE direction in the current feed. Street percentages are computed within each
pattern first, then averaged by those trip shares. Thus a longer variant does not
receive extra weight merely because its geometry is longer.

You can instead use --aggregation canonical to select the most-used shape in
each direction and treat that shape once. This is often more useful if you want
a "typical physical alignment" for each one-way direction.

Install
-------
    python -m pip install "osmnx>=2.1" pandas geopandas shapely pyproj requests

Usage
-----
    export MTC_511_API_KEY="your-511-token"
    python muni_route_street_directional_analysis_normalized_cablecar_named_fallback_tunnels.py

The default route types already include Muni cable cars.

Notes
-----
* GTFS route_type 0 = tram/streetcar/light rail, 3 = bus, 5 = cable tram.
* The main CSV includes street_length_m: the service-weighted mean meters per scheduled trip on each street/category.
* Unknowns are kept separate: [UNNAMED ROAD] means an OSM edge was matched
  within the distance cutoff but lacks a usable name/ref;
  [OFF-STREET / UNMATCHED] means no candidate edge was found within the
  cutoff (35 m by default).
* A conservative alias table rolls up known same-corridor labels such as
  Van Ness BRT/South Van Ness/Van Ness, Geary Street/Geary Boulevard, and
  several infrastructure-specific labels before percentages are aggregated.
* Normal street matching considers every road edge within 35 m and scores it by
  distance plus GTFS/road heading difference, then uses route continuity across
  adjacent samples to suppress intersection/cross-street glitches.
* The two configured rail-only sections (J 18th-22nd and M St Francis Circle-
  Eucalyptus) are labeled [RIGHT OF WAY] using their current GTFS stop boundaries.
* When the nearest matched edge is unnamed, a conservative second pass may use a nearby named edge
  only if it remains within 35 m, is at most 15 m farther than the unnamed edge, and aligns
  within 30 degrees of the GTFS segment. All thresholds are configurable.
* For GTFS route_type=0, only Twin Peaks Tunnel (K/L/M) and Sunset Tunnel (N)
  are eligible tunnel overrides. GTFS stop boundaries constrain each corridor,
  while OSM tunnel geometry confirms the underground alignment. Market Street
  Subway, Central Subway, Caltrain tunnels, and generic rail-tunnel names are
  intentionally ignored by the special-tunnel layer.
* This is geometric map-matching, not an authoritative SFMTA street-by-street
  turn list. Inspect low-confidence/off-street results before using them as
  ground truth.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString, Point


GTFS_URL = "https://api.511.org/transit/datafeeds"
MUNI_OPERATOR_ID = "SF"
UNNAMED_ROAD = "[UNNAMED ROAD]"
UNMATCHED_STREET = "[OFF-STREET / UNMATCHED]"
LEGACY_UNKNOWN_STREET = "[UNKNOWN]"
NORMALIZATION_VERSION = "2026-08-16-v10-pathwise-road-match-whitelisted-tunnels-and-row"

# Conservative aliases for cases where OSM uses multiple labels for what is
# effectively the same named transit corridor. Keep this list explicit so we
# do not accidentally merge genuinely different streets with similar names
# (for example, 15th Street and 15th Avenue).
STREET_ALIASES = {
    # Van Ness corridor / BRT infrastructure labels.
    "Van Ness Bus Rapid Transit": "Van Ness Avenue",
    "South Van Ness Avenue": "Van Ness Avenue",

    # Same continuous corridor with suffix/name variants.
    "Geary Street": "Geary Boulevard",
    "Mission Bay Boulevard North": "Mission Bay Boulevard",
    "Mission Bay Boulevard South": "Mission Bay Boulevard",
    "Buena Vista Avenue East": "Buena Vista Avenue",
    "Buena Vista Avenue West": "Buena Vista Avenue",

    # Infrastructure-specific OSM labels that should roll up to the street.
    "La Playa Terminal Loop": "La Playa Street",
    "Stockton Tunnel": "Stockton Street",
    "Stockton Tunnel / Stockton Street": "Stockton Street",
    "General Douglas MacArthur Tunnel": "Veterans Boulevard",
    "Veterans Boulevard / General Douglas MacArthur Tunnel": "Veterans Boulevard",
}

# Match aliases case-insensitively while preserving the canonical spelling above.
STREET_ALIASES_CASEFOLD = {key.casefold(): value for key, value in STREET_ALIASES.items()}


# Rail-tunnel matching is intentionally whitelisted. For this project's metric,
# only Twin Peaks Tunnel and Sunset Tunnel should override the ordinary street
# corridor matcher. Market Street Subway and Central Subway remain attributed
# to their surface corridors, and unrelated nearby railway tunnels are ignored.
#
# GTFS stop boundaries constrain each tunnel to the correct route interval. OSM
# ``railway`` + ``tunnel`` geometry is then used only as confirmation that a
# sampled GTFS segment is actually on underground rail. This avoids trusting
# inconsistent OSM line names such as "Muni Metro" or "M-Line" as tunnel names.
RAIL_TUNNEL_RAILWAY_VALUES = {"tram", "light_rail", "subway", "rail"}
TUNNEL_GENERIC_LABEL = "[TUNNEL]"  # retained for backwards-compatible parsing only
TUNNEL_BOUNDARY_MAX_DISTANCE_M = 120.0

WHITELISTED_TUNNEL_SPECS = {
    "Twin Peaks Tunnel": {
        "routes": ("K", "L", "M"),
        "boundary_a": (
            "west portal station",
            "west portal ave & ulloa st",
        ),
        "boundary_b": (
            "castro station",
            "market st & castro st",
        ),
        "osm_name_patterns": ("twin peaks tunnel",),
    },
    "Sunset Tunnel": {
        "routes": ("N",),
        "boundary_a": (
            "duboce ave & noe st",
            "duboce avenue & noe street",
        ),
        "boundary_b": (
            "carl st & cole st",
            "carl street & cole street",
        ),
        "osm_name_patterns": ("sunset tunnel",),
    },
}

RIGHT_OF_WAY_LABEL = "[RIGHT OF WAY]"

# Explicit rail-only ROW sections requested for this analysis. Boundaries are
# discovered from the current GTFS stops rather than hard-coded coordinates,
# then projected onto each route shape so the exact geometry between the two
# boundary stops is labeled [RIGHT OF WAY].
RIGHT_OF_WAY_STOP_PATTERNS = {
    "J": {
        "start": ("right of way/18th st", "church st & 18th st"),
        "end": ("right of way/22nd st", "church st & 22nd st"),
    },
    "M": {
        "start": ("west portal ave & sloat blvd", "west portal/sloat/st francis circle"),
        "end": ("right of way/eucalyptus dr",),
    },
}


ROUTE_TYPE_LABELS = {
    0: "tram/streetcar/light rail",
    1: "subway/metro",
    2: "rail",
    3: "bus",
    4: "ferry",
    5: "cable tram",
    6: "aerial lift",
    7: "funicular",
    11: "trolleybus",
    12: "monorail",
}

# Exclude things that are generally not streets. We retain pedestrian and
# service ways because transit can legitimately run on transit malls or
# service-only roadways.
NON_STREET_HIGHWAYS = {
    "bridleway",
    "corridor",
    "cycleway",
    "footway",
    "path",
    "steps",
    "track",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map-match SFMTA Muni GTFS route shapes to OSM streets by direction."
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("MTC_511_API_KEY"),
        help="511 SF Bay API key. Defaults to MTC_511_API_KEY env var.",
    )
    p.add_argument(
        "--output",
        default="muni_route_streets_directional_with_cablecar_pathmatch_whitelisted_tunnels_row.csv",
        help="Output CSV path (default: muni_route_streets_directional_with_cablecar_pathmatch_whitelisted_tunnels_row.csv).",
    )
    p.add_argument(
        "--route-types",
        default="0,3,5",
        help=(
            "Comma-separated GTFS route_type values. Default 0,3,5 includes "
            "Muni light rail/streetcar, bus, and cable cars."
        ),
    )
    p.add_argument(
        "--aggregation",
        choices=("scheduled", "canonical"),
        default="scheduled",
        help=(
            "scheduled = average each shape/service pattern by its share of scheduled "
            "trip occurrences; canonical = most-used shape per route/direction."
        ),
    )
    p.add_argument(
        "--sample-m",
        type=float,
        default=15.0,
        help="Map-match sampling interval in meters (default: 15).",
    )
    p.add_argument(
        "--max-match-distance-m",
        type=float,
        default=35.0,
        help="Maximum distance from route to OSM street (default: 35 m).",
    )
    p.add_argument(
        "--bbox-padding-deg",
        type=float,
        default=0.01,
        help="Padding around route bounding box before OSM download.",
    )
    p.add_argument(
        "--cache-dir",
        default=".muni_street_cache",
        help="Cache directory for GTFS and OSM data.",
    )
    p.add_argument(
        "--refresh-gtfs",
        action="store_true",
        help="Redownload the active Muni GTFS feed.",
    )
    p.add_argument(
        "--refresh-osm",
        action="store_true",
        help="Redownload the OSM street network.",
    )
    p.add_argument(
        "--min-percent",
        type=float,
        default=0.0,
        help="Only print/write streets at or above this percentage.",
    )
    p.add_argument(
        "--no-recover-unnamed",
        action="store_true",
        help="Disable conservative unnamed-edge -> nearby named-edge recovery.",
    )
    p.add_argument(
        "--named-fallback-max-distance-m",
        type=float,
        default=35.0,
        help="Maximum GTFS-to-named-edge distance for unnamed recovery (default: 35 m).",
    )
    p.add_argument(
        "--named-fallback-max-extra-distance-m",
        type=float,
        default=15.0,
        help=(
            "A named edge may be at most this much farther away than the nearest "
            "unnamed edge (default: 15 m)."
        ),
    )
    p.add_argument(
        "--named-fallback-max-angle-deg",
        type=float,
        default=30.0,
        help="Maximum undirected GTFS/road alignment difference for recovery (default: 30 degrees).",
    )
    p.add_argument(
        "--no-tunnel-overrides",
        action="store_true",
        help="Disable light-rail tunnel matching/overrides.",
    )
    p.add_argument(
        "--tunnel-match-distance-m",
        type=float,
        default=25.0,
        help="Maximum GTFS-to-OSM rail-tunnel distance for route_type=0 (default: 25 m).",
    )
    p.add_argument(
        "--tunnel-max-angle-deg",
        type=float,
        default=30.0,
        help="Maximum GTFS/tunnel-track alignment difference (default: 30 degrees).",
    )
    p.add_argument(
        "--road-match-angle-weight-m-per-deg",
        type=float,
        default=0.30,
        help=(
            "Heading penalty in the pathwise road matcher, expressed as meters "
            "of score per degree of GTFS/road misalignment (default: 0.30)."
        ),
    )
    p.add_argument(
        "--road-match-street-change-penalty-m",
        type=float,
        default=8.0,
        help=(
            "Continuity penalty when consecutive GTFS samples switch street labels "
            "(default: 8 m-equivalent; set 0 to disable continuity preference)."
        ),
    )
    p.add_argument(
        "--no-right-of-way-overrides",
        action="store_true",
        help="Disable explicit J/M light-rail [RIGHT OF WAY] overrides.",
    )
    p.add_argument(
        "--row-boundary-max-distance-m",
        type=float,
        default=80.0,
        help="Maximum distance from a configured GTFS ROW boundary stop to a route shape (default: 80 m).",
    )
    return p.parse_args()


def parse_route_types(value: str) -> set[int]:
    try:
        return {int(x.strip()) for x in value.split(",") if x.strip()}
    except ValueError as exc:
        raise SystemExit("--route-types must be comma-separated integers") from exc


def download_gtfs(api_key: str, cache_dir: Path, refresh: bool) -> Path:
    if not api_key:
        raise SystemExit(
            "A 511 SF Bay API key is required. Set MTC_511_API_KEY or pass --api-key."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "sf_muni_active_gtfs.zip"
    if out.exists() and not refresh:
        return out

    params = {
        "api_key": api_key,
        "operator_id": MUNI_OPERATOR_ID,
        "status": "active",
    }
    print("Downloading active SFMTA/Muni GTFS from 511 SF Bay...", file=sys.stderr)
    r = requests.get(GTFS_URL, params=params, timeout=120)
    r.raise_for_status()

    # Validate before replacing a good cache file. Some API errors can arrive as
    # text/html with HTTP 200, so checking ZIP structure is safer than MIME type.
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            required = {"routes.txt", "trips.txt", "shapes.txt"}
            missing = required - set(zf.namelist())
            if missing:
                raise RuntimeError(f"GTFS ZIP is missing required files: {sorted(missing)}")
    except zipfile.BadZipFile as exc:
        preview = r.text[:500] if r.content else "<empty response>"
        raise RuntimeError(f"511 response was not a GTFS ZIP. Response starts: {preview!r}") from exc

    out.write_bytes(r.content)
    return out


def read_csv_from_zip(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame | None:
    if name not in zf.namelist():
        return None
    with zf.open(name) as f:
        return pd.read_csv(f, low_memory=False, **kwargs)


def normalize_id_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def compute_service_day_counts(zf: zipfile.ZipFile) -> dict[str, int]:
    """Return number of active dates for each service_id in this GTFS feed."""
    counts: dict[str, int] = defaultdict(int)
    active_dates: dict[str, set[pd.Timestamp]] = defaultdict(set)

    calendar = read_csv_from_zip(zf, "calendar.txt", dtype={"service_id": "string"})
    if calendar is not None and not calendar.empty:
        weekday_cols = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for row in calendar.itertuples(index=False):
            service_id = str(row.service_id)
            start = pd.to_datetime(str(row.start_date), format="%Y%m%d", errors="coerce")
            end = pd.to_datetime(str(row.end_date), format="%Y%m%d", errors="coerce")
            if pd.isna(start) or pd.isna(end) or end < start:
                continue
            allowed = [bool(int(getattr(row, c))) for c in weekday_cols]
            for d in pd.date_range(start, end, freq="D"):
                if allowed[d.weekday()]:
                    active_dates[service_id].add(d.normalize())

    exceptions = read_csv_from_zip(
        zf,
        "calendar_dates.txt",
        dtype={"service_id": "string", "date": "string"},
    )
    if exceptions is not None and not exceptions.empty:
        for row in exceptions.itertuples(index=False):
            service_id = str(row.service_id)
            d = pd.to_datetime(str(row.date), format="%Y%m%d", errors="coerce")
            if pd.isna(d):
                continue
            d = d.normalize()
            if int(row.exception_type) == 1:
                active_dates[service_id].add(d)
            elif int(row.exception_type) == 2:
                active_dates[service_id].discard(d)

    for service_id, dates in active_dates.items():
        counts[service_id] = len(dates)
    return dict(counts)


def load_gtfs(gtfs_zip: Path, route_types: set[int]):
    with zipfile.ZipFile(gtfs_zip) as zf:
        routes = read_csv_from_zip(zf, "routes.txt", dtype={"route_id": "string"})
        trips = read_csv_from_zip(
            zf,
            "trips.txt",
            dtype={
                "route_id": "string",
                "service_id": "string",
                "trip_id": "string",
                "shape_id": "string",
                "direction_id": "string",
            },
        )
        shapes = read_csv_from_zip(zf, "shapes.txt", dtype={"shape_id": "string"})
        stops = read_csv_from_zip(zf, "stops.txt", dtype={"stop_id": "string"})
        stop_times = read_csv_from_zip(
            zf,
            "stop_times.txt",
            dtype={"trip_id": "string", "stop_id": "string"},
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
        directions = read_csv_from_zip(
            zf,
            "directions.txt",
            dtype={"route_id": "string", "direction_id": "string"},
        )
        service_day_counts = compute_service_day_counts(zf)

    assert routes is not None and trips is not None and shapes is not None
    routes = normalize_id_columns(routes, ["route_id"])
    trips = normalize_id_columns(
        trips, ["route_id", "service_id", "trip_id", "shape_id", "direction_id"]
    )
    shapes = normalize_id_columns(shapes, ["shape_id"])
    if stops is None:
        stops = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    else:
        stops = normalize_id_columns(stops, ["stop_id"])
        stops["stop_lat"] = pd.to_numeric(stops.get("stop_lat"), errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops.get("stop_lon"), errors="coerce")
    if stop_times is not None:
        stop_times = normalize_id_columns(stop_times, ["trip_id", "stop_id"])
    if directions is not None:
        directions = normalize_id_columns(directions, ["route_id", "direction_id"])

    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes = routes[routes["route_type"].isin(route_types)].copy()

    trips = trips[trips["route_id"].isin(set(routes["route_id"]))].copy()
    trips = trips[trips["shape_id"].notna()].copy()

    if "direction_id" not in trips.columns:
        trips["direction_id"] = "unknown"
    else:
        trips["direction_id"] = trips["direction_id"].astype("string").fillna("unknown")

    if "trip_headsign" not in trips.columns:
        trips["trip_headsign"] = pd.NA

    if service_day_counts:
        trips["service_days"] = trips["service_id"].map(service_day_counts).fillna(0).astype(float)
        # If a producer uses a non-calendar mechanism we don't understand, do not
        # erase the trip completely. A weight of 1 is a conservative fallback.
        trips.loc[trips["service_days"] <= 0, "service_days"] = 1.0
    else:
        trips["service_days"] = 1.0

    shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes = shapes.dropna(
        subset=["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]
    )

    if stop_times is not None:
        stop_times["stop_sequence"] = pd.to_numeric(
            stop_times["stop_sequence"], errors="coerce"
        )
        stop_times = stop_times.dropna(subset=["trip_id", "stop_id", "stop_sequence"])
        stop_times = stop_times[stop_times["trip_id"].isin(set(trips["trip_id"]))].copy()

    if directions is not None:
        directions = directions[directions["route_id"].isin(set(routes["route_id"]))].copy()

    return routes, trips, shapes, stops, stop_times, directions


def choose_shape_weights(trips: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """Return a normalized service-pattern weight within each direction.

    In scheduled mode, shape_weight is that shape's share of scheduled trip
    occurrences for its route/direction. We count service dates so the weight
    represents actual scheduled runs across the GTFS feed window.
    """
    usage = (
        trips.groupby(["route_id", "direction_id", "shape_id"], dropna=False)["service_days"]
        .sum()
        .reset_index(name="scheduled_trip_occurrences")
    )

    if aggregation == "scheduled":
        totals = usage.groupby(["route_id", "direction_id"])["scheduled_trip_occurrences"].transform("sum")
        usage["shape_weight"] = usage["scheduled_trip_occurrences"] / totals
        return usage[["route_id", "direction_id", "shape_id", "shape_weight", "scheduled_trip_occurrences"]]

    # Canonical: select the highest scheduled-use shape in each direction.
    usage = usage.sort_values(
        ["route_id", "direction_id", "scheduled_trip_occurrences", "shape_id"],
        ascending=[True, True, False, True],
    )
    canonical = usage.drop_duplicates(["route_id", "direction_id"], keep="first").copy()
    canonical["shape_weight"] = 1.0
    return canonical[["route_id", "direction_id", "shape_id", "shape_weight", "scheduled_trip_occurrences"]]


def make_shape_gdf(shapes: pd.DataFrame, selected: pd.DataFrame) -> gpd.GeoDataFrame:
    needed = set(selected["shape_id"])
    s = shapes[shapes["shape_id"].isin(needed)].copy()
    s = s.sort_values(["shape_id", "shape_pt_sequence"])

    rows = []
    for shape_id, group in s.groupby("shape_id", sort=False):
        coords = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
        # Remove consecutive duplicate coordinates, which can create zero-length
        # line segments and warnings in spatial operations.
        deduped = [coords[0]] if coords else []
        for c in coords[1:]:
            if c != deduped[-1]:
                deduped.append(c)
        if len(deduped) >= 2:
            rows.append({"shape_id": shape_id, "geometry": LineString(deduped)})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    merged = selected.merge(gdf, on="shape_id", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")



CARDINAL_DIRECTION_LABELS = {
    "north": "Northbound",
    "south": "Southbound",
    "east": "Eastbound",
    "west": "Westbound",
    "northeast": "Northeastbound",
    "northwest": "Northwestbound",
    "southeast": "Southeastbound",
    "southwest": "Southwestbound",
}


def display_direction_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return CARDINAL_DIRECTION_LABELS.get(text.lower(), text)


def build_direction_metadata(
    trips: pd.DataFrame,
    selected: pd.DataFrame,
    stops: pd.DataFrame | None,
    stop_times: pd.DataFrame | None,
    directions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build labels and dominant terminal pairs for each route/direction.

    Terminal pairs are weighted by scheduled service. In canonical mode the
    caller's `selected` table contains only one shape per direction, so the
    terminal inference is automatically restricted to trips using that shape.
    """
    allowed = selected[["route_id", "direction_id", "shape_id"]].drop_duplicates()
    trip_meta = trips.merge(allowed, on=["route_id", "direction_id", "shape_id"], how="inner")

    trip_meta = trip_meta[
        ["route_id", "direction_id", "trip_id", "trip_headsign", "service_days"]
    ].copy()
    trip_meta["service_days"] = pd.to_numeric(
        trip_meta["service_days"], errors="coerce"
    ).fillna(1.0)

    if stop_times is not None and not stop_times.empty:
        st = stop_times[stop_times["trip_id"].isin(set(trip_meta["trip_id"]))].copy()
        st = st.sort_values(["trip_id", "stop_sequence"])
        first = st.drop_duplicates("trip_id", keep="first")[["trip_id", "stop_id"]].rename(
            columns={"stop_id": "origin_stop_id"}
        )
        last = st.drop_duplicates("trip_id", keep="last")[["trip_id", "stop_id"]].rename(
            columns={"stop_id": "destination_stop_id"}
        )
        trip_meta = trip_meta.merge(first, on="trip_id", how="left").merge(
            last, on="trip_id", how="left"
        )
    else:
        trip_meta["origin_stop_id"] = pd.NA
        trip_meta["destination_stop_id"] = pd.NA

    stop_name_map: dict[str, str] = {}
    if stops is not None and not stops.empty and "stop_name" in stops.columns:
        stop_name_map = (
            stops.dropna(subset=["stop_id"])
            .drop_duplicates("stop_id")
            .set_index("stop_id")["stop_name"]
            .dropna()
            .astype(str)
            .to_dict()
        )

    def stop_label(stop_id: object) -> str | None:
        if stop_id is None or pd.isna(stop_id):
            return None
        sid = str(stop_id)
        name = stop_name_map.get(sid)
        return str(name).strip() if name and str(name).strip() else sid

    trip_meta["origin_terminal"] = trip_meta["origin_stop_id"].map(stop_label)
    trip_meta["destination_terminal"] = trip_meta["destination_stop_id"].map(stop_label)

    official_direction: dict[tuple[str, str], str] = {}
    if directions is not None and not directions.empty:
        direction_col = next(
            (c for c in directions.columns if str(c).lower() == "direction"), None
        )
        if direction_col is not None:
            for row in directions[["route_id", "direction_id", direction_col]].itertuples(
                index=False, name=None
            ):
                rid, did, name = row
                label = display_direction_name(name)
                if label:
                    official_direction[(str(rid), str(did))] = label

    records = []
    for (route_id, direction_id), group in trip_meta.groupby(
        ["route_id", "direction_id"], dropna=False
    ):
        total_weight = float(group["service_days"].sum())

        pair_group = group.dropna(subset=["origin_terminal", "destination_terminal"])
        origin = None
        destination = None
        pair_share = float("nan")
        if not pair_group.empty:
            pair_weights = (
                pair_group.groupby(
                    ["origin_terminal", "destination_terminal"], dropna=False
                )["service_days"]
                .sum()
                .sort_values(ascending=False)
            )
            origin, destination = pair_weights.index[0]
            if total_weight > 0:
                pair_share = 100.0 * float(pair_weights.iloc[0]) / total_weight

        headsign = None
        heads = group.dropna(subset=["trip_headsign"]).copy()
        if not heads.empty:
            heads["trip_headsign"] = heads["trip_headsign"].astype(str).str.strip()
            heads = heads[heads["trip_headsign"] != ""]
            if not heads.empty:
                headsign_weights = (
                    heads.groupby("trip_headsign")["service_days"]
                    .sum()
                    .sort_values(ascending=False)
                )
                headsign = str(headsign_weights.index[0])

        rid = str(route_id)
        did = str(direction_id)
        direction_name = official_direction.get((rid, did))
        if direction_name:
            direction_label = direction_name
        elif origin and destination:
            direction_label = f"{origin} -> {destination}"
        elif headsign:
            direction_label = f"toward {headsign}"
        else:
            direction_label = f"direction {did}"

        records.append(
            {
                "route_id": rid,
                "direction_id": did,
                "direction_name": direction_name,
                "direction_label": direction_label,
                "origin_terminal": origin,
                "destination_terminal": destination,
                "representative_headsign": headsign,
                "dominant_terminal_pair_share_pct": pair_share,
            }
        )

    return pd.DataFrame(records)

def bbox_with_padding(gdf: gpd.GeoDataFrame, padding_deg: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return (
        float(minx - padding_deg),
        float(miny - padding_deg),
        float(maxx + padding_deg),
        float(maxy + padding_deg),
    )


def bbox_cache_key(bbox: tuple[float, float, float, float]) -> str:
    return "_".join(f"{v:.3f}".replace("-", "m").replace(".", "p") for v in bbox)


def download_or_load_osm(
    bbox: tuple[float, float, float, float], cache_dir: Path, refresh: bool
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    graphml = cache_dir / f"osm_streets_raw_v2_{bbox_cache_key(bbox)}.graphml"
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir / "osmnx_http_cache")

    if graphml.exists() and not refresh:
        print(f"Loading cached OSM street graph: {graphml}", file=sys.stderr)
        return ox.io.load_graphml(graphml)

    print("Downloading OpenStreetMap street network through OSMnx/Overpass...", file=sys.stderr)
    G = ox.graph.graph_from_bbox(
        bbox,
        network_type="all_public",
        simplify=False,
        retain_all=True,
        truncate_by_edge=True,
    )
    ox.io.save_graphml(G, graphml)
    return G



def _tag_has_railway(value) -> bool:
    vals = {v.casefold() for v in flatten_tag(value)}
    return bool(vals & RAIL_TUNNEL_RAILWAY_VALUES)


def _tag_is_tunnel(value) -> bool:
    vals = {v.casefold() for v in flatten_tag(value)}
    if not vals:
        return False
    return any(v not in {"no", "false", "0", "none"} for v in vals)


def _tunnel_name(name, ref) -> str | None:
    names = [x.strip() for x in flatten_tag(name) if x.strip()]
    if names:
        return " / ".join(dict.fromkeys(names))
    refs = [x.strip() for x in flatten_tag(ref) if x.strip()]
    if refs:
        return " / ".join(dict.fromkeys(refs))
    return None


def _tunnel_label(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return TUNNEL_GENERIC_LABEL
    text = str(name).strip()
    return f"[TUNNEL: {text}]" if text else TUNNEL_GENERIC_LABEL


def download_or_load_rail_tunnels(
    bbox: tuple[float, float, float, float], cache_dir: Path, refresh: bool
) -> gpd.GeoDataFrame:
    """Download OSM rail tunnel features separately from the road graph."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"osm_rail_tunnels_v1_{bbox_cache_key(bbox)}.pkl"
    if cache_path.exists() and not refresh:
        print(f"Loading cached OSM rail tunnels: {cache_path}", file=sys.stderr)
        cached = pd.read_pickle(cache_path)
        return gpd.GeoDataFrame(cached, geometry="geometry", crs=getattr(cached, "crs", "EPSG:4326"))

    print("Downloading OpenStreetMap rail tunnel features through OSMnx/Overpass...", file=sys.stderr)
    # OSMnx tag dictionaries are union queries, so request both broad tag sets
    # then enforce railway AND tunnel locally below.
    raw = ox.features.features_from_bbox(
        bbox,
        tags={
            "railway": sorted(RAIL_TUNNEL_RAILWAY_VALUES),
            "tunnel": True,
        },
    )
    if raw.empty:
        out = gpd.GeoDataFrame(columns=["tunnel_name", "geometry"], geometry="geometry", crs="EPSG:4326")
        out.to_pickle(cache_path)
        return out

    raw = raw.copy()
    if "railway" not in raw.columns:
        raw["railway"] = None
    if "tunnel" not in raw.columns:
        raw["tunnel"] = None
    raw = raw[raw["railway"].map(_tag_has_railway) & raw["tunnel"].map(_tag_is_tunnel)].copy()
    raw = raw[raw.geometry.notna() & raw.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if raw.empty:
        out = gpd.GeoDataFrame(columns=["tunnel_name", "geometry"], geometry="geometry", crs="EPSG:4326")
        out.to_pickle(cache_path)
        return out

    raw = raw.reset_index()
    if "name" not in raw.columns:
        raw["name"] = None
    if "ref" not in raw.columns:
        raw["ref"] = None
    raw["tunnel_name"] = [_tunnel_name(n, r) for n, r in zip(raw["name"], raw["ref"])]
    keep = [c for c in ["element", "id", "railway", "tunnel", "layer", "name", "ref", "tunnel_name", "geometry"] if c in raw.columns]
    out = gpd.GeoDataFrame(raw[keep], geometry="geometry", crs=raw.crs)
    out.to_pickle(cache_path)
    return out


def prepare_rail_tunnels(raw: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if raw is None or raw.empty:
        return gpd.GeoDataFrame(columns=["tunnel_name", "geometry"], geometry="geometry", crs=target_crs)
    tunnels = raw.to_crs(target_crs).explode(index_parts=False, ignore_index=True)
    tunnels = tunnels[tunnels.geometry.notna() & tunnels.geometry.geom_type.eq("LineString")].copy()
    if "tunnel_name" not in tunnels.columns:
        tunnels["tunnel_name"] = None
    return gpd.GeoDataFrame(tunnels[["tunnel_name", "geometry"]], geometry="geometry", crs=target_crs)


def flatten_tag(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def highway_is_street(value) -> bool:
    vals = {v.lower() for v in flatten_tag(value)}
    if not vals:
        return False
    return any(v not in NON_STREET_HIGHWAYS for v in vals)


def normalize_street_label(label: object) -> str:
    """Normalize aliases while keeping the two unknown modes distinct.

    ``[UNNAMED ROAD]`` means a candidate OSM edge was found within the map-match
    distance, but the edge has no usable ``name`` or ``ref`` tag.

    ``[OFF-STREET / UNMATCHED]`` means no candidate OSM edge was found within
    ``--max-match-distance-m`` (35 m by default).

    Legacy ``[UNKNOWN]`` inputs are treated as unmatched because the old label
    did not preserve enough information to recover which case it represented.
    """
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return UNMATCHED_STREET

    text = str(label).strip()
    if not text:
        return UNMATCHED_STREET

    folded = text.casefold()
    if folded == UNNAMED_ROAD.casefold():
        return UNNAMED_ROAD
    if folded in {UNMATCHED_STREET.casefold(), LEGACY_UNKNOWN_STREET.casefold()}:
        return UNMATCHED_STREET

    # First allow an explicit alias for the entire label.
    whole_alias = STREET_ALIASES_CASEFOLD.get(folded)
    if whole_alias is not None:
        return whole_alias

    # Then normalize OSMnx composite labels component-by-component.
    parts = [part.strip() for part in text.split(" / ") if part.strip()]
    normalized_parts: list[str] = []
    for part in parts:
        part_folded = part.casefold()
        if part_folded == UNNAMED_ROAD.casefold():
            normalized = UNNAMED_ROAD
        elif part_folded in {UNMATCHED_STREET.casefold(), LEGACY_UNKNOWN_STREET.casefold()}:
            normalized = UNMATCHED_STREET
        else:
            normalized = STREET_ALIASES_CASEFOLD.get(part_folded, part)
        if normalized not in normalized_parts:
            normalized_parts.append(normalized)

    if not normalized_parts:
        return UNMATCHED_STREET
    if len(normalized_parts) == 1:
        return normalized_parts[0]
    return " / ".join(normalized_parts)


def coalesce_normalized_streets(
    result: pd.DataFrame, group_keys: list[str]
) -> pd.DataFrame:
    """Enforce street normalization at the final aggregation boundary.

    This is intentionally redundant with normalization during map matching. It
    guarantees that aliases such as Van Ness BRT/South Van Ness and legacy
    unknown labels cannot survive into printed/CSV output, even if an upstream
    code path supplies an unnormalized street label.
    """
    if result.empty:
        return result

    out = result.copy()
    out["street"] = out["street"].map(normalize_street_label)

    value_cols = ["service_weighted_share", "street_length_m"]
    missing = [c for c in value_cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"Expected columns before final street coalescing: {missing}")

    out = (
        out.groupby(group_keys + ["street"], as_index=False, dropna=False)[value_cols]
        .sum()
    )

    # Sanity check: no explicit alias key or legacy unknown label may remain.
    forbidden = {key.casefold() for key in STREET_ALIASES}
    forbidden.add(LEGACY_UNKNOWN_STREET.casefold())
    leaked = sorted(
        {
            str(value)
            for value in out["street"].dropna()
            if str(value).casefold() in forbidden
        }
    )
    if leaked:
        raise RuntimeError(
            "Street normalization invariant failed; unnormalized labels remain: "
            + ", ".join(leaked)
        )
    return out


def verify_normalization_contract() -> None:
    """Fail immediately if aliases or split-unknown semantics regress."""
    cases = {
        "Van Ness Bus Rapid Transit": "Van Ness Avenue",
        "Van Ness Avenue": "Van Ness Avenue",
        "South Van Ness Avenue": "Van Ness Avenue",
        "Van Ness Avenue / Van Ness Bus Rapid Transit": "Van Ness Avenue",
        UNNAMED_ROAD: UNNAMED_ROAD,
        UNMATCHED_STREET: UNMATCHED_STREET,
        LEGACY_UNKNOWN_STREET: UNMATCHED_STREET,
    }
    failures = []
    for raw, expected in cases.items():
        actual = normalize_street_label(raw)
        if actual != expected:
            failures.append(f"{raw!r} -> {actual!r}, expected {expected!r}")
    if failures:
        raise RuntimeError(
            "Street normalization self-test failed: " + "; ".join(failures)
        )


def coalesce_output_percentages(result: pd.DataFrame) -> pd.DataFrame:
    """Last-chance output normalization using already-computed percentages.

    This runs after analyze() and immediately before filtering, CSV writing, and
    console printing. It intentionally duplicates the earlier coalescing stage:
    if an alias somehow survives upstream, it is renamed and its percentage is
    summed into the canonical street here.
    """
    if result.empty:
        return result

    out = result.copy()
    out["street"] = out["street"].map(normalize_street_label)
    value_cols = [c for c in ["street_share_pct", "street_length_m"] if c in out.columns]
    group_cols = [c for c in out.columns if c not in {"street", *value_cols}]
    out = (
        out.groupby(group_cols + ["street"], as_index=False, dropna=False)[value_cols]
        .sum()
    )

    renormalized = out["street"].map(normalize_street_label)
    unstable = out.loc[renormalized != out["street"], "street"].astype(str).tolist()
    if unstable:
        raise RuntimeError(
            "Final output contains non-normalized street labels: " + ", ".join(sorted(set(unstable)))
        )

    forbidden = {key.casefold() for key in STREET_ALIASES}
    forbidden.add(LEGACY_UNKNOWN_STREET.casefold())
    leaked = sorted(
        {
            str(value)
            for value in out["street"].dropna()
            if str(value).casefold() in forbidden
        }
    )
    if leaked:
        raise RuntimeError(
            "Final output normalization invariant failed; aliases remain: "
            + ", ".join(leaked)
        )
    return out


def street_label(name, ref) -> str:
    names = [x.strip() for x in flatten_tag(name) if x.strip()]
    if names:
        # OSMnx sometimes returns multiple names after graph simplification.
        # Preserve the composite label unless it is a known same-corridor alias.
        return normalize_street_label(" / ".join(dict.fromkeys(names)))
    refs = [x.strip() for x in flatten_tag(ref) if x.strip()]
    if refs:
        return normalize_street_label(" / ".join(dict.fromkeys(refs)))
    return UNNAMED_ROAD


def prepare_street_edges(G) -> gpd.GeoDataFrame:
    Gp = ox.projection.project_graph(G)
    edges = ox.convert.graph_to_gdfs(Gp, nodes=False, fill_edge_geometry=True).reset_index()
    if "highway" in edges.columns:
        edges = edges[edges["highway"].map(highway_is_street)].copy()
    if edges.empty:
        raise RuntimeError("OSM query returned no usable street edges.")

    if "name" not in edges.columns:
        edges["name"] = None
    if "ref" not in edges.columns:
        edges["ref"] = None
    edges["street"] = [street_label(n, r) for n, r in zip(edges["name"], edges["ref"])]
    return gpd.GeoDataFrame(edges[["street", "geometry"]], geometry="geometry", crs=edges.crs)


def sampled_segments(line: LineString, sample_m: float) -> list[dict]:
    length = float(line.length)
    if length <= 0:
        return []
    distances = list(np.arange(0.0, length, sample_m))
    if not distances or distances[-1] != length:
        distances.append(length)
    if len(distances) < 2:
        distances = [0.0, length]

    rows = []
    for i, (a, b) in enumerate(zip(distances[:-1], distances[1:])):
        if b <= a:
            continue
        start = line.interpolate(a)
        end = line.interpolate(b)
        mid = line.interpolate((a + b) / 2.0)
        rows.append(
            {
                "segment_index": i,
                "segment_length_m": b - a,
                "shape_start_m": float(a),
                "shape_end_m": float(b),
                "shape_mid_m": float((a + b) / 2.0),
                "start_x": float(start.x),
                "start_y": float(start.y),
                "end_x": float(end.x),
                "end_y": float(end.y),
                "geometry": Point(mid.x, mid.y),
            }
        )
    return rows


def smooth_one_segment_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Fix isolated intersection-induced labels: A, B, A -> A, A, A."""
    if len(df) < 3:
        return df
    labels = df["street"].tolist()
    fixed = labels[:]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            fixed[i] = labels[i - 1]
    out = df.copy()
    out["street"] = fixed
    return out



def _edge_alignment_deg(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    point: Point,
    edge_geometry,
) -> float:
    """Return undirected angle (0-90 deg) between a GTFS sample and an OSM edge.

    The OSM tangent is measured locally around the point nearest the GTFS sample,
    so curved roads are compared using their local direction rather than endpoints.
    """
    if edge_geometry is None or getattr(edge_geometry, "is_empty", True):
        return math.nan

    geom = edge_geometry
    if getattr(geom, "geom_type", "") == "MultiLineString":
        parts = [g for g in geom.geoms if not g.is_empty and g.length > 0]
        if not parts:
            return math.nan
        geom = min(parts, key=lambda g: g.distance(point))
    if getattr(geom, "geom_type", "") != "LineString" or geom.length <= 0:
        return math.nan

    gx = float(end_x) - float(start_x)
    gy = float(end_y) - float(start_y)
    gnorm = math.hypot(gx, gy)
    if gnorm <= 0:
        return math.nan

    pos = float(geom.project(point))
    half_window = min(5.0, max(1.0, float(geom.length) / 4.0))
    a = max(0.0, pos - half_window)
    b = min(float(geom.length), pos + half_window)
    if b - a < 0.25:
        a, b = 0.0, float(geom.length)
    pa = geom.interpolate(a)
    pb = geom.interpolate(b)
    ex = float(pb.x) - float(pa.x)
    ey = float(pb.y) - float(pa.y)
    enorm = math.hypot(ex, ey)
    if enorm <= 0:
        return math.nan

    # abs(dot) makes the comparison direction-independent: a road digitized in
    # the opposite direction is still perfectly aligned.
    cosine = abs((gx * ex + gy * ey) / (gnorm * enorm))
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))



def match_points_to_streets_pathwise(
    points: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_distance_m: float,
    angle_weight_m_per_deg: float,
    street_change_penalty_m: float,
) -> gpd.GeoDataFrame:
    """Match GTFS samples to OSM streets using distance, heading, and continuity.

    For every sample we inspect all OSM street edges inside ``max_distance_m``.
    Candidate emission cost is:

        point-to-edge distance + angle_weight_m_per_deg * heading difference

    A dynamic-programming pass then adds ``street_change_penalty_m`` whenever
    adjacent samples switch street labels. This makes an isolated perpendicular
    cross street much less likely to steal a sample at an intersection while
    still allowing genuine turns when the subsequent geometry supports them.

    The state is the normalized street label, not the individual OSM edge. When
    parallel/reverse OSM edges share a name, only the best geometry for that
    street at that sample is kept.
    """
    if points.empty:
        return points.copy()

    sindex = edges.sindex
    candidate_rows: list[list[dict]] = []
    unknown_like = {UNMATCHED_STREET, LEGACY_UNKNOWN_STREET}

    for _, row in points.sort_values("segment_index").iterrows():
        point = row.geometry
        by_street: dict[str, dict] = {}
        if point is not None and not point.is_empty:
            positions = sindex.query(point.buffer(max_distance_m), predicate="intersects")
            for pos in positions:
                edge = edges.iloc[int(pos)]
                geom = edge.geometry
                distance = float(point.distance(geom))
                if distance > max_distance_m:
                    continue
                street = normalize_street_label(edge["street"])
                angle = _edge_alignment_deg(
                    row["start_x"], row["start_y"], row["end_x"], row["end_y"],
                    point, geom,
                )
                angle_for_score = 90.0 if pd.isna(angle) else float(angle)
                score = distance + angle_weight_m_per_deg * angle_for_score
                # On an exact geometric tie, prefer a named edge. The explicit
                # unnamed fallback still handles genuinely closer unnamed ways.
                if street == UNNAMED_ROAD:
                    score += 0.25
                cand = {
                    "street": street,
                    "match_distance_m": distance,
                    "match_alignment_deg": angle,
                    "match_path_emission_score": score,
                }
                old = by_street.get(street)
                if old is None or (
                    cand["match_path_emission_score"], cand["match_distance_m"]
                ) < (
                    old["match_path_emission_score"], old["match_distance_m"]
                ):
                    by_street[street] = cand

        if by_street:
            candidates = sorted(
                by_street.values(),
                key=lambda c: (
                    c["match_path_emission_score"],
                    c["match_distance_m"],
                    c["street"],
                ),
            )
        else:
            candidates = [{
                "street": UNMATCHED_STREET,
                "match_distance_m": math.nan,
                "match_alignment_deg": math.nan,
                "match_path_emission_score": 0.0,
            }]
        candidate_rows.append(candidates)

    # Viterbi-like dynamic program over street labels.
    costs: list[dict[str, float]] = []
    backs: list[dict[str, str | None]] = []
    selected_candidate: list[dict[str, dict]] = []
    for i, candidates in enumerate(candidate_rows):
        cmap = {c["street"]: c for c in candidates}
        selected_candidate.append(cmap)
        current_costs: dict[str, float] = {}
        current_backs: dict[str, str | None] = {}
        for street, cand in cmap.items():
            emission = float(cand["match_path_emission_score"])
            if i == 0:
                current_costs[street] = emission
                current_backs[street] = None
                continue
            best_prev = None
            best_cost = math.inf
            for prev_street, prev_cost in costs[i - 1].items():
                transition = 0.0
                if (
                    prev_street != street
                    and prev_street not in unknown_like
                    and street not in unknown_like
                ):
                    transition = street_change_penalty_m
                total = prev_cost + transition + emission
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_street
            current_costs[street] = best_cost
            current_backs[street] = best_prev
        costs.append(current_costs)
        backs.append(current_backs)

    last_street = min(costs[-1], key=costs[-1].get)
    chosen = [last_street]
    for i in range(len(costs) - 1, 0, -1):
        prev = backs[i][chosen[-1]]
        if prev is None:
            prev = min(costs[i - 1], key=costs[i - 1].get)
        chosen.append(prev)
    chosen.reverse()

    out = points.sort_values("segment_index").copy()
    streets = []
    distances = []
    angles = []
    emissions = []
    for i, street in enumerate(chosen):
        cand = selected_candidate[i][street]
        streets.append(street)
        distances.append(cand["match_distance_m"])
        angles.append(cand["match_alignment_deg"])
        emissions.append(cand["match_path_emission_score"])
    out["street"] = streets
    out["match_distance_m"] = distances
    out["match_alignment_deg"] = angles
    out["match_path_emission_score"] = emissions
    return out


def prepare_right_of_way_boundaries(
    stops: pd.DataFrame,
    target_crs,
) -> dict[str, dict[str, list[tuple[str, Point]]]]:
    """Find configured J/M ROW boundary stops in the active GTFS feed."""
    if stops is None or stops.empty:
        return {}
    required = {"stop_name", "stop_lat", "stop_lon"}
    if not required.issubset(stops.columns):
        return {}

    work = stops.dropna(subset=["stop_name", "stop_lat", "stop_lon"]).copy()
    if work.empty:
        return {}
    geometry = gpd.points_from_xy(work["stop_lon"], work["stop_lat"])
    gdf = gpd.GeoDataFrame(work, geometry=geometry, crs="EPSG:4326").to_crs(target_crs)
    folded = gdf["stop_name"].astype(str).str.casefold()

    result: dict[str, dict[str, list[tuple[str, Point]]]] = {}
    for route_short, sides in RIGHT_OF_WAY_STOP_PATTERNS.items():
        found: dict[str, list[tuple[str, Point]]] = {}
        for side, patterns in sides.items():
            mask = pd.Series(False, index=gdf.index)
            for pattern in patterns:
                mask = mask | folded.str.contains(pattern.casefold(), regex=False, na=False)
            matches = gdf.loc[mask]
            found[side] = [
                (str(r.stop_name), r.geometry)
                for r in matches.itertuples(index=False)
            ]
        if found.get("start") and found.get("end"):
            result[route_short] = found
    return result



def prepare_whitelisted_tunnel_boundaries(
    stops: pd.DataFrame,
    target_crs,
) -> dict[str, dict[str, object]]:
    """Locate GTFS stop boundaries for the two whitelisted tunnel corridors.

    These boundaries are deliberately independent of OSM tunnel names. They
    prevent a generic ``Muni Metro`` tunnel way from spilling the Twin Peaks
    override into the Market Street Subway, and prevent unrelated rail tunnels
    near a route from being considered at all.
    """
    if stops is None or stops.empty:
        return {}
    required = {"stop_name", "stop_lat", "stop_lon"}
    if not required.issubset(stops.columns):
        return {}

    work = stops.dropna(subset=["stop_name", "stop_lat", "stop_lon"]).copy()
    if work.empty:
        return {}
    geometry = gpd.points_from_xy(work["stop_lon"], work["stop_lat"])
    gdf = gpd.GeoDataFrame(work, geometry=geometry, crs="EPSG:4326").to_crs(target_crs)
    folded = gdf["stop_name"].astype(str).str.casefold()

    result: dict[str, dict[str, object]] = {}
    for canonical_name, spec in WHITELISTED_TUNNEL_SPECS.items():
        found: dict[str, object] = {
            "routes": tuple(str(x).upper() for x in spec["routes"]),
            "osm_name_patterns": tuple(str(x).casefold() for x in spec["osm_name_patterns"]),
        }
        for side in ("boundary_a", "boundary_b"):
            mask = pd.Series(False, index=gdf.index)
            for pattern in spec[side]:
                mask = mask | folded.str.contains(str(pattern).casefold(), regex=False, na=False)
            matches = gdf.loc[mask]
            found[side] = [
                (str(r.stop_name), r.geometry)
                for r in matches.itertuples(index=False)
            ]
        result[canonical_name] = found
    return result

def apply_manual_right_of_way_overrides(
    matched: gpd.GeoDataFrame,
    line: LineString,
    *,
    route_short_name: str,
    row_boundaries: dict[str, dict[str, list[tuple[str, Point]]]] | None,
    enabled: bool,
    boundary_max_distance_m: float,
) -> gpd.GeoDataFrame:
    """Label the configured J and M rail-only sections as [RIGHT OF WAY]."""
    out = matched.copy()
    key = str(route_short_name or "").strip().upper()
    if not enabled or out.empty or not row_boundaries or key not in row_boundaries:
        return out

    spec = row_boundaries[key]
    start_candidates = spec.get("start", [])
    end_candidates = spec.get("end", [])
    if not start_candidates or not end_candidates:
        return out

    _, start_pt = min(start_candidates, key=lambda item: float(line.distance(item[1])))
    _, end_pt = min(end_candidates, key=lambda item: float(line.distance(item[1])))
    if (
        float(line.distance(start_pt)) > boundary_max_distance_m
        or float(line.distance(end_pt)) > boundary_max_distance_m
    ):
        return out

    a = float(line.project(start_pt))
    b = float(line.project(end_pt))
    lo, hi = sorted((a, b))
    if hi - lo < 25.0:
        return out

    mask = (out["shape_end_m"] > lo) & (out["shape_start_m"] < hi)
    if bool(mask.any()):
        out.loc[mask, "street"] = RIGHT_OF_WAY_LABEL
    return out

def _osm_name_matches_whitelisted_tunnel(
    osm_name: object,
    canonical_name: str,
) -> bool:
    """Strict fallback when GTFS boundary stops cannot be resolved."""
    if osm_name is None or (isinstance(osm_name, float) and math.isnan(osm_name)):
        return False
    folded = str(osm_name).casefold()
    spec = WHITELISTED_TUNNEL_SPECS[canonical_name]
    return any(str(pattern).casefold() in folded for pattern in spec["osm_name_patterns"])


def _tunnel_interval_on_shape(
    line: LineString,
    canonical_name: str,
    tunnel_boundaries: dict[str, dict[str, object]] | None,
) -> tuple[float, float, str, str] | None:
    """Return the along-shape interval between this tunnel's GTFS boundary stops."""
    if not tunnel_boundaries or canonical_name not in tunnel_boundaries:
        return None
    spec = tunnel_boundaries[canonical_name]
    a_candidates = spec.get("boundary_a", [])
    b_candidates = spec.get("boundary_b", [])
    if not a_candidates or not b_candidates:
        return None

    a_name, a_pt = min(a_candidates, key=lambda item: float(line.distance(item[1])))
    b_name, b_pt = min(b_candidates, key=lambda item: float(line.distance(item[1])))
    if (
        float(line.distance(a_pt)) > TUNNEL_BOUNDARY_MAX_DISTANCE_M
        or float(line.distance(b_pt)) > TUNNEL_BOUNDARY_MAX_DISTANCE_M
    ):
        return None

    a = float(line.project(a_pt))
    b = float(line.project(b_pt))
    lo, hi = sorted((a, b))
    if hi - lo < 100.0:
        return None
    return lo, hi, a_name, b_name


def apply_rail_tunnel_overrides(
    matched: gpd.GeoDataFrame,
    line: LineString,
    tunnel_edges: gpd.GeoDataFrame | None,
    *,
    route_short_name: str,
    tunnel_boundaries: dict[str, dict[str, object]] | None,
    enabled: bool,
    max_distance_m: float,
    max_alignment_deg: float,
) -> gpd.GeoDataFrame:
    """Override only the whitelisted Twin Peaks and Sunset tunnel geometry."""
    out = matched.copy()
    if not enabled or tunnel_edges is None or tunnel_edges.empty or out.empty:
        return out

    route_key = str(route_short_name or "").strip().upper()
    eligible_names = [
        name
        for name, spec in WHITELISTED_TUNNEL_SPECS.items()
        if route_key in {str(x).upper() for x in spec["routes"]}
    ]
    if not eligible_names:
        return out

    spatial_index = tunnel_edges.sindex
    for canonical_name in eligible_names:
        interval = _tunnel_interval_on_shape(line, canonical_name, tunnel_boundaries)
        if interval is None:
            lo = hi = None
        else:
            lo, hi, _, _ = interval

        for left_idx, row in out.iterrows():
            if lo is not None and hi is not None:
                if not (float(row["shape_end_m"]) > lo and float(row["shape_start_m"]) < hi):
                    continue

            point = row.geometry
            if point is None or point.is_empty:
                continue
            positions = spatial_index.query(point.buffer(max_distance_m), predicate="intersects")
            if len(positions) == 0:
                continue

            candidates: list[tuple] = []
            for pos in positions:
                tunnel_row = tunnel_edges.iloc[int(pos)]
                geom = tunnel_row.geometry
                distance = float(point.distance(geom))
                if distance > max_distance_m:
                    continue
                angle = _edge_alignment_deg(
                    row["start_x"], row["start_y"], row["end_x"], row["end_y"],
                    point, geom,
                )
                if pd.isna(angle) or angle > max_alignment_deg:
                    continue
                osm_name = tunnel_row.get("tunnel_name")
                if lo is None and not _osm_name_matches_whitelisted_tunnel(osm_name, canonical_name):
                    continue
                osm_sort = "" if osm_name is None or pd.isna(osm_name) else str(osm_name)
                candidates.append((distance, angle, osm_sort))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                out.at[left_idx, "street"] = f"[TUNNEL: {canonical_name}]"

    return out

def recover_unnamed_with_named_edges(
    joined: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    enabled: bool,
    max_named_distance_m: float,
    max_extra_distance_m: float,
    max_alignment_deg: float,
) -> gpd.GeoDataFrame:
    """Conservatively replace an unnamed matched edge with a nearby named edge."""
    out = joined.copy()
    if not enabled or out.empty:
        return out

    unnamed_mask = out["street"].map(normalize_street_label).eq(UNNAMED_ROAD)
    if not bool(unnamed_mask.any()):
        return out

    named_edges = edges[
        ~edges["street"].map(normalize_street_label).isin(
            {UNNAMED_ROAD, UNMATCHED_STREET, LEGACY_UNKNOWN_STREET}
        )
    ].copy()
    if named_edges.empty:
        return out

    spatial_index = named_edges.sindex
    for left_idx, row in out.loc[unnamed_mask].iterrows():
        point = row.geometry
        if point is None or point.is_empty:
            continue

        positions = spatial_index.query(point.buffer(max_named_distance_m), predicate="intersects")
        if len(positions) == 0:
            continue

        original_distance = pd.to_numeric(
            pd.Series([row.get("match_distance_m")]), errors="coerce"
        ).iloc[0]
        if pd.isna(original_distance):
            original_distance = 0.0

        candidates: list[tuple] = []
        for pos in positions:
            edge_row = named_edges.iloc[int(pos)]
            edge_geom = edge_row.geometry
            candidate_distance = float(point.distance(edge_geom))
            if candidate_distance > max_named_distance_m:
                continue
            extra_distance = candidate_distance - float(original_distance)
            angle = _edge_alignment_deg(
                row["start_x"], row["start_y"], row["end_x"], row["end_y"],
                point, edge_geom,
            )
            if (
                extra_distance <= max_extra_distance_m
                and not pd.isna(angle)
                and angle <= max_alignment_deg
            ):
                candidates.append((
                    candidate_distance,
                    angle,
                    normalize_street_label(edge_row["street"]),
                ))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
            out.at[left_idx, "street"] = candidates[0][2]

    return out

def match_shape_to_streets(
    shape_id: str,
    line: LineString,
    edges: gpd.GeoDataFrame,
    sample_m: float,
    max_match_distance_m: float,
    *,
    recover_unnamed: bool,
    named_fallback_max_distance_m: float,
    named_fallback_max_extra_distance_m: float,
    named_fallback_max_angle_deg: float,
    tunnel_edges: gpd.GeoDataFrame | None = None,
    tunnel_overrides_enabled: bool = False,
    tunnel_match_distance_m: float = 25.0,
    tunnel_max_alignment_deg: float = 30.0,
    road_match_angle_weight_m_per_deg: float = 0.30,
    road_match_street_change_penalty_m: float = 8.0,
    route_short_name: str = "",
    tunnel_boundaries: dict[str, dict[str, object]] | None = None,
    row_boundaries: dict[str, dict[str, list[tuple[str, Point]]]] | None = None,
    right_of_way_overrides_enabled: bool = True,
    row_boundary_max_distance_m: float = 80.0,
) -> pd.DataFrame:
    rows = sampled_segments(line, sample_m)
    if not rows:
        return pd.DataFrame(columns=["street", "segment_length_m"])

    points = gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)
    points["segment_uid"] = [f"{shape_id}:{i}" for i in points["segment_index"]]

    joined = match_points_to_streets_pathwise(
        points,
        edges,
        max_distance_m=max_match_distance_m,
        angle_weight_m_per_deg=road_match_angle_weight_m_per_deg,
        street_change_penalty_m=road_match_street_change_penalty_m,
    )
    joined["street"] = joined["street"].map(normalize_street_label)

    # Only successful-but-unnamed matches enter this recovery pass.  True
    # unmatched points retain [OFF-STREET / UNMATCHED].
    joined = recover_unnamed_with_named_edges(
        joined,
        edges,
        enabled=recover_unnamed,
        max_named_distance_m=min(max_match_distance_m, named_fallback_max_distance_m),
        max_extra_distance_m=named_fallback_max_extra_distance_m,
        max_alignment_deg=named_fallback_max_angle_deg,
    )

    joined["street"] = joined["street"].map(normalize_street_label)
    joined = joined.sort_values("segment_index")
    joined = smooth_one_segment_glitches(joined)
    joined = apply_rail_tunnel_overrides(
        joined,
        line,
        tunnel_edges,
        route_short_name=route_short_name,
        tunnel_boundaries=tunnel_boundaries,
        enabled=tunnel_overrides_enabled,
        max_distance_m=tunnel_match_distance_m,
        max_alignment_deg=tunnel_max_alignment_deg,
    )
    joined = apply_manual_right_of_way_overrides(
        joined,
        line,
        route_short_name=route_short_name,
        row_boundaries=row_boundaries,
        enabled=right_of_way_overrides_enabled,
        boundary_max_distance_m=row_boundary_max_distance_m,
    )
    return joined[["street", "segment_length_m"]]


def analyze(
    routes: pd.DataFrame,
    shape_gdf: gpd.GeoDataFrame,
    street_edges: gpd.GeoDataFrame,
    direction_metadata: pd.DataFrame,
    sample_m: float,
    max_match_distance_m: float,
    aggregation: str,
    recover_unnamed: bool,
    named_fallback_max_distance_m: float,
    named_fallback_max_extra_distance_m: float,
    named_fallback_max_angle_deg: float,
    rail_tunnels: gpd.GeoDataFrame,
    tunnel_boundaries: dict[str, dict[str, object]],
    tunnel_overrides_enabled: bool,
    tunnel_match_distance_m: float,
    tunnel_max_alignment_deg: float,
    road_match_angle_weight_m_per_deg: float,
    road_match_street_change_penalty_m: float,
    row_boundaries: dict[str, dict[str, list[tuple[str, Point]]]],
    right_of_way_overrides_enabled: bool,
    row_boundary_max_distance_m: float,
) -> pd.DataFrame:
    projected_shapes = shape_gdf.to_crs(street_edges.crs)

    # Percentages preserve the established methodology: normalize each service
    # pattern first, then weight its percentage distribution by scheduled trips.
    accum_share: dict[tuple[str, str, str], float] = defaultdict(float)

    # Length is the service-weighted mean meters traveled on each street/category
    # per scheduled trip in this direction. It is intentionally tracked
    # separately because percentage weighting normalizes each pattern first.
    accum_length_m: dict[tuple[str, str, str], float] = defaultdict(float)

    route_type_num = pd.to_numeric(routes.get("route_type"), errors="coerce")
    rail_route_ids = set(routes.loc[route_type_num.eq(0), "route_id"].astype(str))
    route_short_by_id = routes.set_index("route_id")["route_short_name"].fillna("").astype(str).to_dict() if "route_short_name" in routes.columns else {}

    total = len(projected_shapes)
    for idx, row in enumerate(projected_shapes.itertuples(index=False), start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"Map-matching route shape {idx}/{total}...", file=sys.stderr)
        matched = match_shape_to_streets(
            str(row.shape_id),
            row.geometry,
            street_edges,
            sample_m,
            max_match_distance_m,
            recover_unnamed=recover_unnamed,
            named_fallback_max_distance_m=named_fallback_max_distance_m,
            named_fallback_max_extra_distance_m=named_fallback_max_extra_distance_m,
            named_fallback_max_angle_deg=named_fallback_max_angle_deg,
            tunnel_edges=rail_tunnels if str(row.route_id) in rail_route_ids else None,
            tunnel_overrides_enabled=(tunnel_overrides_enabled and str(row.route_id) in rail_route_ids),
            tunnel_match_distance_m=tunnel_match_distance_m,
            tunnel_max_alignment_deg=tunnel_max_alignment_deg,
            road_match_angle_weight_m_per_deg=road_match_angle_weight_m_per_deg,
            road_match_street_change_penalty_m=road_match_street_change_penalty_m,
            route_short_name=route_short_by_id.get(str(row.route_id), ""),
            tunnel_boundaries=tunnel_boundaries,
            row_boundaries=row_boundaries,
            right_of_way_overrides_enabled=(right_of_way_overrides_enabled and str(row.route_id) in rail_route_ids),
            row_boundary_max_distance_m=row_boundary_max_distance_m,
        )
        by_street = matched.groupby("street")["segment_length_m"].sum()
        shape_total_m = float(by_street.sum())
        if shape_total_m <= 0:
            continue
        weight = float(row.shape_weight)
        route_id = str(row.route_id)
        direction_id = str(row.direction_id)
        for street, meters in by_street.items():
            key = (route_id, direction_id, str(street))
            pattern_street_share = float(meters) / shape_total_m
            accum_share[key] += pattern_street_share * weight
            accum_length_m[key] += float(meters) * weight

    records = [
        {
            "route_id": route_id,
            "direction_id": direction_id,
            "street": street,
            "service_weighted_share": share,
            "street_length_m": accum_length_m[(route_id, direction_id, street)],
        }
        for (route_id, direction_id, street), share in accum_share.items()
    ]
    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError("No route/street matches were produced.")

    result = coalesce_normalized_streets(result, ["route_id", "direction_id"])

    totals = (
        result.groupby(["route_id", "direction_id"])["service_weighted_share"]
        .sum()
        .rename("direction_share_total")
    )
    result = result.merge(totals, on=["route_id", "direction_id"], how="left")
    result["street_share_pct"] = (
        100.0 * result["service_weighted_share"] / result["direction_share_total"]
    )

    route_cols = [
        c
        for c in ["route_id", "route_short_name", "route_long_name", "route_type"]
        if c in routes.columns
    ]
    result = result.merge(routes[route_cols].drop_duplicates("route_id"), on="route_id", how="left")
    if not direction_metadata.empty:
        result = result.merge(direction_metadata, on=["route_id", "direction_id"], how="left")
    result["route_type_label"] = result["route_type"].map(ROUTE_TYPE_LABELS).fillna("other")
    result["aggregation"] = aggregation

    ordered = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_type_label",
        "direction_id",
        "direction_name",
        "direction_label",
        "origin_terminal",
        "destination_terminal",
        "representative_headsign",
        "dominant_terminal_pair_share_pct",
        "street",
        "street_length_m",
        "street_share_pct",
        "aggregation",
    ]
    return result[[c for c in ordered if c in result.columns]]

def main() -> int:
    args = parse_args()
    print(
        f"Street normalization: {NORMALIZATION_VERSION}",
        file=sys.stderr,
    )
    verify_normalization_contract()
    if args.sample_m <= 0:
        raise SystemExit("--sample-m must be > 0")
    if args.max_match_distance_m <= 0:
        raise SystemExit("--max-match-distance-m must be > 0")
    if args.named_fallback_max_distance_m <= 0:
        raise SystemExit("--named-fallback-max-distance-m must be > 0")
    if args.named_fallback_max_extra_distance_m < 0:
        raise SystemExit("--named-fallback-max-extra-distance-m must be >= 0")
    if not (0 <= args.named_fallback_max_angle_deg <= 90):
        raise SystemExit("--named-fallback-max-angle-deg must be between 0 and 90")
    if args.road_match_angle_weight_m_per_deg < 0:
        raise SystemExit("--road-match-angle-weight-m-per-deg must be >= 0")
    if args.road_match_street_change_penalty_m < 0:
        raise SystemExit("--road-match-street-change-penalty-m must be >= 0")
    if args.row_boundary_max_distance_m <= 0:
        raise SystemExit("--row-boundary-max-distance-m must be > 0")
    if args.tunnel_match_distance_m <= 0:
        raise SystemExit("--tunnel-match-distance-m must be > 0")
    if not (0 <= args.tunnel_max_angle_deg <= 90):
        raise SystemExit("--tunnel-max-angle-deg must be between 0 and 90")

    route_types = parse_route_types(args.route_types)
    cache_dir = Path(args.cache_dir)

    gtfs_zip = download_gtfs(args.api_key, cache_dir, args.refresh_gtfs)
    routes, trips, shapes, stops, stop_times, directions = load_gtfs(gtfs_zip, route_types)
    if routes.empty:
        raise RuntimeError(f"No Muni routes matched route types {sorted(route_types)}")

    selected = choose_shape_weights(trips, args.aggregation)
    direction_metadata = build_direction_metadata(
        trips, selected, stops, stop_times, directions
    )
    shape_gdf = make_shape_gdf(shapes, selected)
    if shape_gdf.empty:
        raise RuntimeError("No GTFS shapes could be built for the selected Muni routes.")

    bbox = bbox_with_padding(shape_gdf, args.bbox_padding_deg)
    G = download_or_load_osm(bbox, cache_dir, args.refresh_osm)
    street_edges = prepare_street_edges(G)
    row_boundaries = prepare_right_of_way_boundaries(stops, street_edges.crs)
    tunnel_boundaries = prepare_whitelisted_tunnel_boundaries(stops, street_edges.crs)

    if 0 in route_types and not args.no_tunnel_overrides:
        raw_tunnels = download_or_load_rail_tunnels(bbox, cache_dir, args.refresh_osm)
        rail_tunnels = prepare_rail_tunnels(raw_tunnels, street_edges.crs)
    else:
        rail_tunnels = gpd.GeoDataFrame(columns=["tunnel_name", "geometry"], geometry="geometry", crs=street_edges.crs)

    result = analyze(
        routes,
        shape_gdf,
        street_edges,
        direction_metadata,
        args.sample_m,
        args.max_match_distance_m,
        args.aggregation,
        not args.no_recover_unnamed,
        args.named_fallback_max_distance_m,
        args.named_fallback_max_extra_distance_m,
        args.named_fallback_max_angle_deg,
        rail_tunnels,
        tunnel_boundaries,
        not args.no_tunnel_overrides,
        args.tunnel_match_distance_m,
        args.tunnel_max_angle_deg,
        args.road_match_angle_weight_m_per_deg,
        args.road_match_street_change_penalty_m,
        row_boundaries,
        not args.no_right_of_way_overrides,
        args.row_boundary_max_distance_m,
    )
    result = coalesce_output_percentages(result)
    result = result[result["street_share_pct"] >= args.min_percent].copy()
    result = result.sort_values(
        ["route_short_name", "direction_id", "street_share_pct"],
        ascending=[True, True, False],
        na_position="last",
    )

    # Round only at output time so percentages are computed from full precision.
    out = result.copy()
    out["street_length_m"] = pd.to_numeric(out["street_length_m"], errors="coerce").round(1)
    out["street_share_pct"] = out["street_share_pct"].round(3)
    if "dominant_terminal_pair_share_pct" in out.columns:
        out["dominant_terminal_pair_share_pct"] = out[
            "dominant_terminal_pair_share_pct"
        ].round(3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out):,} route/street rows to {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
