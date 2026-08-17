#!/usr/bin/env python3
"""
Analyze the streets served by stops on each San Francisco Muni bus route.

The script downloads the active SFMTA/Muni GTFS feed from 511 SF Bay, reads
routes/trips/stop_times/stops, groups trips into distinct ordered stop patterns,
and computes the percentage of stop calls on each street.

Intersection semantics
----------------------
A stop named, for example, ``Mission St & 16th St`` counts once toward Mission
Street and once toward 16th Street. The denominator remains the number of stop
calls in the service pattern, so street percentages for a route are NOT
expected to sum to 100%; for intersection-heavy routes they can approach 200%.

Service weighting
-----------------
The default ``--aggregation scheduled`` mode mirrors the route/street analyzer:
each distinct ordered stop pattern is weighted by its share of scheduled trip
occurrences in the active GTFS feed. A trip active on 20 service dates
contributes 20 scheduled occurrences. Percentages are computed within each stop
pattern first, then averaged by those pattern weights. Thus a longer pattern
does not get extra influence merely because it contains more stops.

By default results combine both directions of a route. Pass ``--directional``
to calculate percentages separately for each GTFS direction_id.

Install
-------
    python -m pip install pandas requests

Usage
-----
    export MTC_511_API_KEY="your-511-token"
    python muni_route_stop_street_analysis_normalized.py

Directional output:
    python muni_route_stop_street_analysis_normalized.py --directional \
        --output muni_bus_stop_streets_directional.csv

The default route type is 3 (bus), which includes Muni motorbus/trolleybus
services represented as GTFS bus routes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


GTFS_URL = "https://api.511.org/transit/datafeeds"
MUNI_OPERATOR_ID = "SF"
UNKNOWN_STREET = "[UNKNOWN]"
NORMALIZATION_VERSION = "2026-08-16-stop-streets-v1"

# Keep this aligned with the normalized route/street analyzers.
STREET_ALIASES = {
    "Van Ness Bus Rapid Transit": "Van Ness Avenue",
    "South Van Ness Avenue": "Van Ness Avenue",
    "Geary Street": "Geary Boulevard",
    "Mission Bay Boulevard North": "Mission Bay Boulevard",
    "Mission Bay Boulevard South": "Mission Bay Boulevard",
    "Buena Vista Avenue East": "Buena Vista Avenue",
    "Buena Vista Avenue West": "Buena Vista Avenue",
    "La Playa Terminal Loop": "La Playa Street",
    "Stockton Tunnel": "Stockton Street",
    "Stockton Tunnel / Stockton Street": "Stockton Street",
    "General Douglas MacArthur Tunnel": "Veterans Boulevard",
    "Veterans Boulevard / General Douglas MacArthur Tunnel": "Veterans Boulevard",
}
STREET_ALIASES_CASEFOLD = {k.casefold(): v for k, v in STREET_ALIASES.items()}

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

# GTFS stop names commonly abbreviate suffixes even when OSM uses the full form.
# Only expand suffix-like tokens at the end of the street name or immediately
# before a trailing cardinal word, so a name like "St Francis" is not changed.
STREET_SUFFIXES = {
    "St": "Street",
    "Ave": "Avenue",
    "Blvd": "Boulevard",
    "Rd": "Road",
    "Dr": "Drive",
    "Ln": "Lane",
    "Ct": "Court",
    "Pl": "Place",
    "Ter": "Terrace",
    "Pkwy": "Parkway",
    "Hwy": "Highway",
    "Expy": "Expressway",
    "Cir": "Circle",
    "Sq": "Square",
}

TRAILING_CARDINALS = ("North", "South", "East", "West")
PREFIX_CARDINALS = {"N": "North", "S": "South", "E": "East", "W": "West"}

# Common intersection separators in stop names. Ampersand is by far the most
# common in Muni GTFS; the word variants are included defensively.
INTERSECTION_RE = re.compile(r"\s+(?:&|at|and)\s+", flags=re.IGNORECASE)

# Used to decide whether a one-part stop name is plausibly a street rather than
# a landmark/terminal name such as "Salesforce Transit Center Bay 7".
STREET_WORD_RE = re.compile(
    r"\b(?:Street|Avenue|Boulevard|Road|Drive|Lane|Court|Place|Terrace|Parkway|"
    r"Highway|Expressway|Circle|Square|Way|St|Ave|Blvd|Rd|Dr|Ln|Ct|Pl|Ter|Pkwy|Hwy|Expy|Cir|Sq)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute service-weighted Muni stop percentages by street."
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("MTC_511_API_KEY"),
        help="511 SF Bay API key. Defaults to MTC_511_API_KEY env var.",
    )
    p.add_argument(
        "--output",
        default="muni_bus_stop_street_percentages.csv",
        help="Summary CSV path.",
    )
    p.add_argument(
        "--stops-output",
        default="muni_bus_route_stops.csv",
        help=(
            "Detail CSV listing route/stop/street membership and weighted service "
            "presence. Pass an empty string to disable."
        ),
    )
    p.add_argument(
        "--route-types",
        default="3",
        help="Comma-separated GTFS route_type values. Default: 3 (bus).",
    )
    p.add_argument(
        "--aggregation",
        choices=("scheduled", "canonical"),
        default="scheduled",
        help=(
            "scheduled = weight every stop pattern by scheduled-trip share; "
            "canonical = choose the most-used stop pattern in each direction."
        ),
    )
    p.add_argument(
        "--directional",
        action="store_true",
        help="Calculate percentages separately for each GTFS direction_id.",
    )
    p.add_argument(
        "--cache-dir",
        default=".muni_street_cache",
        help="Cache directory for the active Muni GTFS ZIP.",
    )
    p.add_argument(
        "--refresh-gtfs",
        action="store_true",
        help="Redownload the active Muni GTFS feed.",
    )
    p.add_argument(
        "--min-percent",
        type=float,
        default=0.0,
        help="Only print/write streets at or above this percentage.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-route percentages to stdout.",
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

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            required = {"routes.txt", "trips.txt", "stop_times.txt", "stops.txt"}
            missing = required - set(zf.namelist())
            if missing:
                raise RuntimeError(f"GTFS ZIP is missing required files: {sorted(missing)}")
    except zipfile.BadZipFile as exc:
        preview = r.text[:500] if r.content else "<empty response>"
        raise RuntimeError(
            f"511 response was not a GTFS ZIP. Response starts: {preview!r}"
        ) from exc

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

    return {service_id: len(dates) for service_id, dates in active_dates.items()}


def expand_gtfs_street_name(text: str) -> str:
    """Expand common GTFS street abbreviations without mangling names."""
    value = re.sub(r"\s+", " ", str(text).strip())
    value = re.sub(r"\s+\([^)]*\)\s*$", "", value).strip()

    # Expand a one-letter cardinal prefix, e.g. "S Van Ness Ave".
    words = value.split()
    if len(words) >= 2 and words[0].upper() in PREFIX_CARDINALS:
        words[0] = PREFIX_CARDINALS[words[0].upper()]
        value = " ".join(words)

    cardinal_alt = "|".join(TRAILING_CARDINALS)
    for short, full in STREET_SUFFIXES.items():
        # Expand when the abbreviation is the suffix, or when it is followed by
        # a terminal cardinal qualifier such as "Mission Bay Blvd North".
        value = re.sub(
            rf"\b{re.escape(short)}\.?\b(?=\s+(?:{cardinal_alt})\b|$)",
            full,
            value,
            flags=re.IGNORECASE,
        )
    return value.strip()


def normalize_street_label(label: object) -> str:
    """Normalize GTFS/OSM street labels to the shared canonical names."""
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return UNKNOWN_STREET

    text = expand_gtfs_street_name(str(label).strip())
    unknown_labels = {
        "[UNNAMED ROAD]",
        "[OFF-STREET / UNMATCHED]",
        UNKNOWN_STREET,
    }
    if not text or text.upper() in {value.upper() for value in unknown_labels}:
        return UNKNOWN_STREET

    whole_alias = STREET_ALIASES_CASEFOLD.get(text.casefold())
    if whole_alias is not None:
        return whole_alias

    # Preserve genuinely different slash-composite labels, but collapse aliases
    # component-by-component just like the route/street analyzer.
    parts = [part.strip() for part in text.split(" / ") if part.strip()]
    normalized_parts: list[str] = []
    for part in parts:
        part = expand_gtfs_street_name(part)
        if part.upper() in {value.upper() for value in unknown_labels}:
            normalized = UNKNOWN_STREET
        else:
            normalized = STREET_ALIASES_CASEFOLD.get(part.casefold(), part)
        if normalized not in normalized_parts:
            normalized_parts.append(normalized)

    named = [part for part in normalized_parts if part != UNKNOWN_STREET]
    if not named:
        return UNKNOWN_STREET
    if len(named) == 1:
        return named[0]
    return " / ".join(named)


def parse_stop_streets(stop_name: object) -> list[str]:
    """Return unique normalized streets credited by a GTFS stop name.

    ``A & B`` returns both A and B. If aliases cause both sides to resolve to
    the same canonical street, that street is counted only once for the stop.
    """
    if stop_name is None or (isinstance(stop_name, float) and math.isnan(stop_name)):
        return [UNKNOWN_STREET]

    text = re.sub(r"\s+", " ", str(stop_name).strip())
    if not text:
        return [UNKNOWN_STREET]

    pieces = [p.strip(" ,") for p in INTERSECTION_RE.split(text) if p.strip(" ,")]

    # A single non-street landmark/terminal label has no trustworthy street
    # encoded in its name. Keep it explicit rather than guessing.
    if len(pieces) == 1 and not STREET_WORD_RE.search(pieces[0]):
        return [UNKNOWN_STREET]

    normalized: list[str] = []
    for piece in pieces:
        street = normalize_street_label(piece)
        if street not in normalized:
            normalized.append(street)

    named = [s for s in normalized if s != UNKNOWN_STREET]
    return named if named else [UNKNOWN_STREET]


def verify_normalization_contract() -> None:
    cases = {
        "Van Ness Ave": "Van Ness Avenue",
        "Van Ness Bus Rapid Transit": "Van Ness Avenue",
        "South Van Ness Ave": "Van Ness Avenue",
        "Geary St": "Geary Boulevard",
        "Mission Bay Blvd North": "Mission Bay Boulevard",
        "[UNNAMED ROAD]": UNKNOWN_STREET,
        "[OFF-STREET / UNMATCHED]": UNKNOWN_STREET,
    }
    failures = []
    for raw, expected in cases.items():
        actual = normalize_street_label(raw)
        if actual != expected:
            failures.append(f"{raw!r} -> {actual!r}, expected {expected!r}")

    stop_cases = {
        "Van Ness Ave & Mission St": ["Van Ness Avenue", "Mission Street"],
        "South Van Ness Ave & 16th St": ["Van Ness Avenue", "16th Street"],
        "Geary St & Masonic Ave": ["Geary Boulevard", "Masonic Avenue"],
        # Both sides normalize to Van Ness, so the stop credits it only once.
        "Van Ness Ave & South Van Ness Ave": ["Van Ness Avenue"],
    }
    for raw, expected in stop_cases.items():
        actual = parse_stop_streets(raw)
        if actual != expected:
            failures.append(f"stop {raw!r} -> {actual!r}, expected {expected!r}")

    if failures:
        raise RuntimeError("Stop/street normalization self-test failed: " + "; ".join(failures))


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
            },
        )
        stop_times = read_csv_from_zip(
            zf,
            "stop_times.txt",
            dtype={"trip_id": "string", "stop_id": "string"},
        )
        stops = read_csv_from_zip(
            zf,
            "stops.txt",
            dtype={"stop_id": "string", "parent_station": "string"},
        )
        service_day_counts = compute_service_day_counts(zf)

    assert routes is not None and trips is not None and stop_times is not None and stops is not None

    routes = normalize_id_columns(routes, ["route_id"])
    trips = normalize_id_columns(trips, ["route_id", "service_id", "trip_id"])
    stop_times = normalize_id_columns(stop_times, ["trip_id", "stop_id"])
    stops = normalize_id_columns(stops, ["stop_id", "parent_station"])

    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes = routes[routes["route_type"].isin(route_types)].copy()
    trips = trips[trips["route_id"].isin(set(routes["route_id"]))].copy()

    if "direction_id" not in trips.columns:
        trips["direction_id"] = "unknown"
    else:
        trips["direction_id"] = trips["direction_id"].astype("string").fillna("unknown")

    if service_day_counts:
        trips["service_days"] = trips["service_id"].map(service_day_counts).fillna(0).astype(float)
        trips.loc[trips["service_days"] <= 0, "service_days"] = 1.0
    else:
        trips["service_days"] = 1.0

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["trip_id", "stop_id", "stop_sequence"])
    stop_times = stop_times[stop_times["trip_id"].isin(set(trips["trip_id"]))].copy()
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    # Only keep the stop fields needed downstream, but preserve coordinates in
    # the detail output when present.
    wanted_stop_cols = [
        c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"]
        if c in stops.columns
    ]
    stops = stops[wanted_stop_cols].drop_duplicates("stop_id")

    return routes, trips, stop_times, stops


def ordered_pattern_id(stop_ids: list[str]) -> str:
    payload = "\x1f".join(stop_ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def build_trip_patterns(
    trips: pd.DataFrame, stop_times: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Attach an ordered-stop pattern_id to each GTFS trip."""
    sequences = (
        stop_times.groupby("trip_id", sort=False)["stop_id"]
        .apply(lambda s: [str(x) for x in s.tolist()])
    )

    trip_rows = trips[trips["trip_id"].isin(sequences.index)].copy()
    pattern_sequences: dict[str, list[str]] = {}
    pattern_ids = {}
    for trip_id, stop_ids in sequences.items():
        pid = ordered_pattern_id(stop_ids)
        pattern_ids[str(trip_id)] = pid
        existing = pattern_sequences.get(pid)
        if existing is not None and existing != stop_ids:
            raise RuntimeError("Unexpected SHA-1 stop-pattern collision")
        pattern_sequences[pid] = stop_ids

    trip_rows["pattern_id"] = trip_rows["trip_id"].astype(str).map(pattern_ids)
    trip_rows = trip_rows[trip_rows["pattern_id"].notna()].copy()
    return trip_rows, pattern_sequences


def choose_pattern_weights(
    trip_patterns: pd.DataFrame,
    aggregation: str,
    directional: bool,
) -> pd.DataFrame:
    usage = (
        trip_patterns.groupby(["route_id", "direction_id", "pattern_id"], dropna=False)["service_days"]
        .sum()
        .reset_index(name="scheduled_trip_occurrences")
    )

    if aggregation == "canonical":
        usage = usage.sort_values(
            ["route_id", "direction_id", "scheduled_trip_occurrences", "pattern_id"],
            ascending=[True, True, False, True],
        )
        selected = usage.drop_duplicates(["route_id", "direction_id"], keep="first").copy()
        if directional:
            selected["pattern_weight"] = 1.0
        else:
            # Match the combined-route street analyzer: each selected direction
            # gets equal influence rather than being length/stop-count weighted.
            counts = selected.groupby("route_id")["pattern_id"].transform("count")
            selected["pattern_weight"] = 1.0 / counts
        return selected

    if directional:
        totals = usage.groupby(["route_id", "direction_id"])["scheduled_trip_occurrences"].transform("sum")
    else:
        totals = usage.groupby("route_id")["scheduled_trip_occurrences"].transform("sum")
    usage["pattern_weight"] = usage["scheduled_trip_occurrences"] / totals
    return usage


def build_stop_lookup(stops: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in stops.to_dict(orient="records"):
        stop_id = str(row["stop_id"])
        stop_name = row.get("stop_name")
        lookup[stop_id] = {
            **row,
            "streets": parse_stop_streets(stop_name),
        }
    return lookup


def analyze_summary(
    routes: pd.DataFrame,
    selected_patterns: pd.DataFrame,
    pattern_sequences: dict[str, list[str]],
    stop_lookup: dict[str, dict],
    aggregation: str,
    directional: bool,
) -> pd.DataFrame:
    """Compute service-weighted stop percentages by normalized street."""
    if directional:
        accum: dict[tuple[str, str, str], float] = defaultdict(float)
    else:
        accum: dict[tuple[str, str], float] = defaultdict(float)

    for row in selected_patterns.itertuples(index=False):
        stop_ids = pattern_sequences.get(str(row.pattern_id), [])
        if not stop_ids:
            continue
        denominator = float(len(stop_ids))
        counts: Counter[str] = Counter()
        for stop_id in stop_ids:
            streets = stop_lookup.get(str(stop_id), {}).get("streets", [UNKNOWN_STREET])
            # parse_stop_streets() already deduplicates aliases within a stop.
            for street in streets:
                counts[normalize_street_label(street)] += 1

        weight = float(row.pattern_weight)
        for street, stop_count in counts.items():
            pattern_share = float(stop_count) / denominator
            if directional:
                accum[(str(row.route_id), str(row.direction_id), street)] += pattern_share * weight
            else:
                accum[(str(row.route_id), street)] += pattern_share * weight

    records = []
    if directional:
        for (route_id, direction_id, street), share in accum.items():
            records.append(
                {
                    "route_id": route_id,
                    "direction_id": direction_id,
                    "street": normalize_street_label(street),
                    "stop_share_pct": 100.0 * share,
                }
            )
        group_keys = ["route_id", "direction_id", "street"]
    else:
        for (route_id, street), share in accum.items():
            records.append(
                {
                    "route_id": route_id,
                    "street": normalize_street_label(street),
                    "stop_share_pct": 100.0 * share,
                }
            )
        group_keys = ["route_id", "street"]

    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError("No route/stop/street results were produced.")

    # Hard-stop final normalization/coalescing, mirroring the corrected route
    # analyzer. This guarantees aliases cannot survive into the output.
    result["street"] = result["street"].map(normalize_street_label)
    result = result.groupby(group_keys, as_index=False, dropna=False)["stop_share_pct"].sum()

    unstable = result.loc[
        result["street"].map(normalize_street_label) != result["street"], "street"
    ].astype(str).tolist()
    if unstable:
        raise RuntimeError(
            "Final stop output contains non-normalized street labels: "
            + ", ".join(sorted(set(unstable)))
        )

    route_cols = [
        c for c in ["route_id", "route_short_name", "route_long_name", "route_type"]
        if c in routes.columns
    ]
    result = result.merge(routes[route_cols].drop_duplicates("route_id"), on="route_id", how="left")
    result["route_type_label"] = result["route_type"].map(ROUTE_TYPE_LABELS).fillna("other")
    result["aggregation"] = aggregation
    return result


def build_stop_detail(
    routes: pd.DataFrame,
    selected_patterns: pd.DataFrame,
    pattern_sequences: dict[str, list[str]],
    stop_lookup: dict[str, dict],
    directional: bool,
) -> pd.DataFrame:
    """List route/stop/street membership with scheduled service presence.

    ``service_presence_pct`` is the weighted share of selected service patterns
    containing that stop at least once. It is useful for distinguishing a stop
    used by nearly every run from one used only by a short variant.
    """
    if directional:
        presence: dict[tuple[str, str, str], float] = defaultdict(float)
    else:
        presence: dict[tuple[str, str], float] = defaultdict(float)

    for row in selected_patterns.itertuples(index=False):
        # Presence is per pattern, so repeated visits to the same stop in a loop
        # do not make the stop appear on >100% of service.
        for stop_id in dict.fromkeys(pattern_sequences.get(str(row.pattern_id), [])):
            if directional:
                key = (str(row.route_id), str(row.direction_id), str(stop_id))
            else:
                key = (str(row.route_id), str(stop_id))
            presence[key] += float(row.pattern_weight)

    records = []
    for key, weight in presence.items():
        if directional:
            route_id, direction_id, stop_id = key
        else:
            route_id, stop_id = key
            direction_id = None
        info = stop_lookup.get(stop_id, {})
        stop_name = info.get("stop_name", "")
        streets = info.get("streets", [UNKNOWN_STREET])
        for street in streets:
            rec = {
                "route_id": route_id,
                "stop_id": stop_id,
                "stop_name": stop_name,
                "street": normalize_street_label(street),
                "service_presence_pct": 100.0 * weight,
            }
            if directional:
                rec["direction_id"] = direction_id
            for col in ("stop_lat", "stop_lon"):
                if col in info:
                    rec[col] = info.get(col)
            records.append(rec)

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail

    detail["street"] = detail["street"].map(normalize_street_label)
    # A stop side may normalize into the same canonical street as another side.
    # Coalesce to one route/stop/street row and keep the service presence once.
    detail_keys = ["route_id"] + (["direction_id"] if directional else []) + ["stop_id", "street"]
    agg = {
        "stop_name": "first",
        "service_presence_pct": "max",
    }
    if "stop_lat" in detail.columns:
        agg["stop_lat"] = "first"
    if "stop_lon" in detail.columns:
        agg["stop_lon"] = "first"
    detail = detail.groupby(detail_keys, as_index=False, dropna=False).agg(agg)

    route_cols = [
        c for c in ["route_id", "route_short_name", "route_long_name"]
        if c in routes.columns
    ]
    detail = detail.merge(routes[route_cols].drop_duplicates("route_id"), on="route_id", how="left")
    return detail


def route_sort_key(value: object) -> tuple:
    s = "" if pd.isna(value) else str(value)
    chunks = []
    current = ""
    is_digit = None
    for ch in s:
        d = ch.isdigit()
        if is_digit is None or d == is_digit:
            current += ch
        else:
            chunks.append(int(current) if is_digit else current.lower())
            current = ch
        is_digit = d
    if current:
        chunks.append(int(current) if is_digit else current.lower())
    return tuple((0, x) if isinstance(x, int) else (1, x) for x in chunks)


def print_report(result: pd.DataFrame, min_percent: float, directional: bool) -> None:
    route_order = (
        result[["route_id", "route_short_name", "route_long_name"]]
        .drop_duplicates()
        .assign(_sort=lambda d: d["route_short_name"].map(route_sort_key))
        .sort_values("_sort")
    )
    for r in route_order.itertuples(index=False):
        short = "" if pd.isna(r.route_short_name) else str(r.route_short_name)
        long = "" if pd.isna(r.route_long_name) else str(r.route_long_name)
        title = " ".join(x for x in [short, long] if x).strip() or str(r.route_id)
        route_rows = result[result["route_id"] == r.route_id]

        if directional:
            for direction_id, sub in route_rows.groupby("direction_id", dropna=False):
                print(f"\n=== {title} | direction_id={direction_id} ===")
                sub = sub[sub["stop_share_pct"] >= min_percent].sort_values(
                    "stop_share_pct", ascending=False
                )
                for row in sub.itertuples(index=False):
                    print(f"{row.stop_share_pct:6.2f}%  {row.street}")
        else:
            print(f"\n=== {title} ===")
            sub = route_rows[route_rows["stop_share_pct"] >= min_percent].sort_values(
                "stop_share_pct", ascending=False
            )
            for row in sub.itertuples(index=False):
                print(f"{row.stop_share_pct:6.2f}%  {row.street}")


def main() -> int:
    args = parse_args()
    print(f"Stop street normalization: {NORMALIZATION_VERSION}", file=sys.stderr)
    verify_normalization_contract()

    if args.min_percent < 0:
        raise SystemExit("--min-percent must be >= 0")

    route_types = parse_route_types(args.route_types)
    cache_dir = Path(args.cache_dir)
    gtfs_zip = download_gtfs(args.api_key, cache_dir, args.refresh_gtfs)
    routes, trips, stop_times, stops = load_gtfs(gtfs_zip, route_types)
    if routes.empty:
        raise RuntimeError(f"No Muni routes matched route types {sorted(route_types)}")

    print("Building ordered GTFS stop patterns...", file=sys.stderr)
    trip_patterns, pattern_sequences = build_trip_patterns(trips, stop_times)
    selected_patterns = choose_pattern_weights(
        trip_patterns, args.aggregation, args.directional
    )
    stop_lookup = build_stop_lookup(stops)

    print(
        f"Analyzing {len(selected_patterns):,} weighted stop patterns across "
        f"{routes['route_id'].nunique():,} routes...",
        file=sys.stderr,
    )
    result = analyze_summary(
        routes,
        selected_patterns,
        pattern_sequences,
        stop_lookup,
        args.aggregation,
        args.directional,
    )
    result = result[result["stop_share_pct"] >= args.min_percent].copy()

    sort_cols = ["route_short_name"]
    if args.directional:
        sort_cols.append("direction_id")
    sort_cols.append("stop_share_pct")
    ascending = [True] * (len(sort_cols) - 1) + [False]
    result = result.sort_values(sort_cols, ascending=ascending, na_position="last")

    out = result.copy()
    out["stop_share_pct"] = out["stop_share_pct"].round(3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out):,} route/street summary rows to {output_path}", file=sys.stderr)

    if args.stops_output:
        detail = build_stop_detail(
            routes,
            selected_patterns,
            pattern_sequences,
            stop_lookup,
            args.directional,
        )
        if not detail.empty:
            detail["service_presence_pct"] = detail["service_presence_pct"].round(3)
            detail_sort = ["route_short_name"]
            if args.directional:
                detail_sort.append("direction_id")
            detail_sort += ["stop_name", "street"]
            detail = detail.sort_values(detail_sort, na_position="last")
        detail_path = Path(args.stops_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(detail_path, index=False)
        print(f"Wrote {len(detail):,} route/stop/street detail rows to {detail_path}", file=sys.stderr)

    if not args.quiet:
        print_report(result, args.min_percent, args.directional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
