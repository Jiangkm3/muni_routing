#!/usr/bin/env python3
"""Plot Muni route street composition with bar lengths proportional to route length.

The chart uses the same street normalization, route-name highlighting, tunnel/right-of-way
colors, exclusions, and directional selection rule as the percentage chart. Unlike the
percentage chart, every route is shown in natural route-label order with no separate
neighborhood section. Each stacked bar uses ``street_length_m`` for its physical width,
so the longest selected route fills the plotting row and shorter routes are proportionally
shorter.

For directional data, the script evaluates each direction separately and keeps the same
best direction used by the percentage chart: highest combined named-street share, then
lower N/A share, then stronger terminal-pair share, then lower direction_id.

Example:
    python plot_muni_route_named_streets_by_length.py \
        muni_directional_streets.csv \
        --output muni_route_named_street_length.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# ---------------------------------------------------------------------------
# Street normalization
# ---------------------------------------------------------------------------

UNKNOWN_STREET = "[UNKNOWN]"
RIGHT_OF_WAY_STREET = "[RIGHT OF WAY]"
TUNNEL_PREFIX = "[TUNNEL"

# Special infrastructure colors are deliberately very light so the red/yellow
# route-name highlights remain visually dominant.
TUNNEL_FACE = "#EAF4FF"
RIGHT_OF_WAY_FACE = "#EDF8ED"
N_A_FACE = "0.88"


def is_tunnel_street(value: object) -> bool:
    text = str(value).strip().casefold()
    return text.startswith("[tunnel") and text.endswith("]")


def is_right_of_way_street(value: object) -> bool:
    return str(value).strip().casefold() == RIGHT_OF_WAY_STREET.casefold()


def is_special_infrastructure(value: object) -> bool:
    return is_tunnel_street(value) or is_right_of_way_street(value)

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
UNKNOWN_INPUTS = {
    "[UNNAMED ROAD]",
    "[UNNAMED STREET]",
    "[OFF-STREET / UNMATCHED]",
    UNKNOWN_STREET,
}
UNKNOWN_INPUTS_CASEFOLD = {x.casefold() for x in UNKNOWN_INPUTS}


def normalize_street_label(value: object) -> str:
    """Normalize one street label, including slash-separated OSM names."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN_STREET

    text = str(value).strip()
    if not text or text.casefold() in UNKNOWN_INPUTS_CASEFOLD:
        return UNKNOWN_STREET

    # Preserve infrastructure categories verbatim instead of treating the
    # slash/bracket syntax as a composite street label.
    if is_tunnel_street(text) or is_right_of_way_street(text):
        return text

    whole = STREET_ALIASES_CASEFOLD.get(text.casefold())
    if whole is not None:
        return whole

    parts = [p.strip() for p in text.split(" / ") if p.strip()]
    normalized: list[str] = []
    for part in parts:
        if part.casefold() in UNKNOWN_INPUTS_CASEFOLD:
            item = UNKNOWN_STREET
        else:
            item = STREET_ALIASES_CASEFOLD.get(part.casefold(), part)
        if item not in normalized:
            normalized.append(item)

    named = [x for x in normalized if x != UNKNOWN_STREET]
    if not named:
        return UNKNOWN_STREET
    if len(named) == 1:
        return named[0]
    return " / ".join(named)


def coalesce_streets(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize street labels and re-sum aliases without merging directions."""
    out = df.copy()
    out["street"] = out["street"].map(normalize_street_label)

    metadata_candidates = [
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
        "aggregation",
    ]
    id_cols = [c for c in metadata_candidates if c in out.columns]
    value_cols = [c for c in ["street_share_pct", "street_length_m"] if c in out.columns]

    return (
        out.groupby(id_cols + ["street"], dropna=False, as_index=False)[value_cols]
        .sum()
    )


# ---------------------------------------------------------------------------
# Route-name -> street matching and route classification
# ---------------------------------------------------------------------------

STREET_SUFFIXES = {
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd",
    "drive", "dr", "way", "terrace", "ter", "court", "ct", "lane", "ln",
    "highway", "hwy", "freeway", "parkway", "pkwy", "circle", "cir",
    "place", "pl", "plaza", "tunnel", "ramp",
}

ROUTE_CORE_ALIASES = {
    "third": "3rd",
}

# Optional exact overrides. Values preserve route-name order.
# Set a route to None to suppress street highlighting deliberately.
ROUTE_NAME_OVERRIDES: dict[str, list[str] | None] = {
    # California Street Cable Car: keep an explicit invariant even if its
    # public long name gains/removes a service qualifier in a future GTFS feed.
    "CA": ["California Street"],
    # "49": ["Van Ness Avenue", "Mission Street"],
    # "F": ["Market Street", "The Embarcadero"],
}

# Section placement only. These routes STILL go through normal street matching.
NEIGHBORHOOD_NAMED_ROUTES = {
    "8",     # BAYSHORE
    "8AX",   # BAYSHORE A EXPRESS
    "8BX",   # BAYSHORE B EXPRESS
    "15",    # BAYVIEW HUNTERS POINT EXPRESS
    "25",    # TREASURE ISLAND
    "30X",   # MARINA EXPRESS
    "39",    # COIT
    "52",    # EXCELSIOR
    "55",    # DOGPATCH
    "57",    # PARKMERCED
    "67",    # BERNAL HEIGHTS
    "K",     # INGLESIDE
    "M",     # OCEAN VIEW
}


def is_excluded_route(route_short_name: object) -> bool:
    """Exclude replacement BUS/OWL services and route 90/91/714."""
    short = str(route_short_name).strip().upper()
    return short in ["90", "91", "714"] or short.endswith("BUS") or short.endswith("OWL")


def _basic_text(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _street_core(street: str) -> str:
    base = _basic_text(street)
    words = base.split()
    while words and words[-1] in STREET_SUFFIXES:
        words.pop()
    return " ".join(words)


def _clean_route_long_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\bEARLY\s+BIRD\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCABLE\s+CAR\b", " ", s, flags=re.IGNORECASE)
    for word in ["RAPID", "EXPRESS", "OWL", "BUS"]:
        s = re.sub(rf"\b{word}\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\b[AB]\b\s*$", "", s, flags=re.IGNORECASE)
    return " ".join(s.split())


def _route_components(long_name: str) -> list[str]:
    cleaned = _clean_route_long_name(long_name)
    parts = [p.strip() for p in re.split(r"\s*(?:-|&)\s*", cleaned) if p.strip()]
    return parts[:2]


def _component_core(component: str) -> str:
    base = _basic_text(component)
    words = base.split()
    while words and words[-1] in STREET_SUFFIXES:
        words.pop()
    core = " ".join(words)
    return ROUTE_CORE_ALIASES.get(core, core)


def is_neighborhood_named(route_short_name: object) -> bool:
    return str(route_short_name).upper() in NEIGHBORHOOD_NAMED_ROUTES


def infer_named_streets(
    route_short_name: str,
    route_long_name: str,
    streets: Iterable[str],
) -> list[str]:
    """Infer zero, one, or two literal route-name streets, preserving order."""
    short = str(route_short_name)
    available = [normalize_street_label(s) for s in streets]
    available = list(dict.fromkeys(available))

    if short in ROUTE_NAME_OVERRIDES:
        override = ROUTE_NAME_OVERRIDES[short]
        if not override:
            return []
        present = {s.casefold(): s for s in available}
        return [present[x.casefold()] for x in override if x.casefold() in present][:2]

    by_core: dict[str, list[str]] = {}
    for street in available:
        if street == UNKNOWN_STREET or is_special_infrastructure(street):
            continue
        core = _street_core(street)
        by_core.setdefault(core, []).append(street)

    matched: list[str] = []
    for component in _route_components(route_long_name):
        core = _component_core(component)
        candidates = by_core.get(core, [])
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda x: (len(x), x.casefold()))[0]
        if chosen not in matched:
            matched.append(chosen)
        if len(matched) == 2:
            break
    return matched


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

LABEL_ABBREVIATIONS = [
    (r"\bBoulevard\b", "Blvd"),
    (r"\bAvenue\b", "Ave"),
    (r"\bStreet\b", "St"),
    (r"\bRoad\b", "Rd"),
    (r"\bDrive\b", "Dr"),
    (r"\bHighway\b", "Hwy"),
    (r"\bFreeway\b", "Fwy"),
    (r"\bParkway\b", "Pkwy"),
    (r"\bTerrace\b", "Ter"),
    (r"\bCourt\b", "Ct"),
    (r"\bLane\b", "Ln"),
    (r"\bCircle\b", "Cir"),
    (r"\bPlace\b", "Pl"),
    (r"\bWay\b", "Way"),
]

# When a street-name label is tight, first abbreviate the suffix, then remove
# it entirely. If the remaining name still looks too wide for the segment, the
# chart shows only the percentage rather than squeezing or truncating the name.
LABEL_SUFFIXES = {
    "boulevard", "blvd", "avenue", "ave", "street", "st", "road", "rd",
    "drive", "dr", "highway", "hwy", "parkway", "pkwy", "terrace", "ter",
    "way",
}
LABEL_CHARS_PER_PERCENT = 2.2


def abbreviated_street_name(street: str) -> str:
    if street == UNKNOWN_STREET:
        return "N/A"
    if is_right_of_way_street(street):
        return "Right of way"
    if is_tunnel_street(street):
        match = re.fullmatch(r"\[TUNNEL(?::\s*(.*?))?\]", street, flags=re.IGNORECASE)
        if match and match.group(1):
            return match.group(1).strip()
        return "Tunnel"
    s = street
    for pattern, repl in LABEL_ABBREVIATIONS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def strip_street_suffix(street: str) -> str:
    """Drop a trailing road-type suffix (e.g. Blvd/Road) for tight labels."""
    words = abbreviated_street_name(street).split()
    if words and words[-1].casefold().rstrip(".") in LABEL_SUFFIXES:
        words = words[:-1]
    return " ".join(words).strip()


def fitted_street_name(street: str, width_pct: float) -> str | None:
    """Return the least-shortened street name likely to fit, else None.

    Highlighted labels may abbreviate a road suffix to stay legible in narrow
    segments. Tunnel labels keep the word "Tunnel" intact. Ordinary white street
    labels use ``ordinary_street_name_if_fits`` so a compact road-type suffix
    (St/Ave/Blvd/Fwy/etc.) must still fit rather than being dropped.
    """
    if street == UNKNOWN_STREET:
        return "N/A"

    max_chars = max(1, int(math.floor(width_pct * LABEL_CHARS_PER_PERCENT)))
    candidates = [abbreviated_street_name(street), strip_street_suffix(street)]
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if len(candidate) <= max_chars:
            return candidate
    return None


def ordinary_street_name_if_fits(street: str, width_pct: float) -> str | None:
    """Return an ordinary street label only when its compact full name fits.

    Road-type suffixes are retained but may use conventional abbreviations such
    as St, Ave, Blvd, Hwy, or Fwy. We never fall back to a suffix-free name.
    Ordinal suffixes are normalized for display. Tunnel labels are handled
    separately and never abbreviate the word "Tunnel".
    """
    if street == UNKNOWN_STREET or is_special_infrastructure(street):
        return None
    display = abbreviated_street_name(street)
    display = re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        lambda m: f"{m.group(1)}{m.group(2).lower()}",
        display.strip(),
        flags=re.IGNORECASE,
    )
    max_chars = max(1, int(math.floor(width_pct * LABEL_CHARS_PER_PERCENT)))
    return display if display and len(display) <= max_chars else None


def smart_title(value: object) -> str:
    """Title-case route names without producing ordinals like 3Rd/18Th."""
    text = str(value).title()
    return re.sub(
        r"\b(\d+)(St|Nd|Rd|Th)\b",
        lambda m: f"{m.group(1)}{m.group(2).lower()}",
        text,
    )


def natural_key(value: object) -> list[object]:
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x.casefold() for x in parts]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


# ---------------------------------------------------------------------------
# Direction selection and route ordering
# ---------------------------------------------------------------------------


def _direction_display(meta: dict[str, object]) -> str:
    for field in ["direction_name", "direction_label"]:
        value = meta.get(field)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    value = meta.get("direction_id")
    if value is not None and not pd.isna(value):
        try:
            return f"Direction {int(float(value))}"
        except (TypeError, ValueError):
            return f"Direction {value}"
    return ""


def _candidate_record(
    meta: dict[str, object],
    g: pd.DataFrame,
) -> dict[str, object]:
    named = infer_named_streets(
        str(meta["route_short_name"]),
        str(meta["route_long_name"]),
        g["street"].tolist(),
    )
    lookup = g.groupby("street")["street_share_pct"].sum().to_dict()
    route_length_m = float(g["street_length_m"].sum()) if "street_length_m" in g.columns else float("nan")
    primary = named[0] if len(named) >= 1 else None
    secondary = named[1] if len(named) >= 2 else None
    primary_share = float(lookup.get(primary, 0.0)) if primary else 0.0
    secondary_share = float(lookup.get(secondary, 0.0)) if secondary else 0.0

    rec = dict(meta)
    rec.update({
        "primary_name_street": primary,
        "primary_share_pct": primary_share,
        "secondary_name_street": secondary,
        "secondary_share_pct": secondary_share,
        "named_share_pct": primary_share + secondary_share,
        "na_share_pct": float(lookup.get(UNKNOWN_STREET, 0.0)),
        "route_length_m": route_length_m,
        "is_neighborhood_named": is_neighborhood_named(meta["route_short_name"]),
        "direction_display": _direction_display(meta),
    })
    return rec


def build_route_records(df: pd.DataFrame, route_cols: list[str]) -> list[dict[str, object]]:
    """Build one record per route, selecting the best direction when present."""
    directional = "direction_id" in df.columns
    records: list[dict[str, object]] = []

    for route_key, route_g in df.groupby(route_cols, dropna=False, sort=False):
        if not isinstance(route_key, tuple):
            route_key = (route_key,)
        route_meta = dict(zip(route_cols, route_key))

        if is_excluded_route(route_meta["route_short_name"]):
            continue

        candidates: list[dict[str, object]] = []
        if directional:
            for direction_id, dg in route_g.groupby("direction_id", dropna=False, sort=False):
                meta = dict(route_meta)
                meta["direction_id"] = direction_id
                for field in [
                    "direction_name",
                    "direction_label",
                    "origin_terminal",
                    "destination_terminal",
                    "representative_headsign",
                    "dominant_terminal_pair_share_pct",
                ]:
                    if field in dg.columns:
                        vals = dg[field].dropna()
                        meta[field] = vals.iloc[0] if not vals.empty else None
                candidates.append(_candidate_record(meta, dg))
        else:
            candidates.append(_candidate_record(dict(route_meta), route_g))

        # Objective rule for every route, including neighborhoods:
        # 1) highest combined literal route-name street share;
        # 2) on an exact named-share tie, lower N/A share;
        # 3) stronger dominant terminal pair;
        # 4) lowest direction_id for deterministic final tie-breaking.
        candidates.sort(
            key=lambda r: (
                -safe_float(r.get("named_share_pct")),
                safe_float(r.get("na_share_pct")),
                -safe_float(r.get("dominant_terminal_pair_share_pct")),
                safe_float(r.get("direction_id"), 999.0),
            )
        )
        records.append(candidates[0])

    # Length chart: no neighborhood/street-name sections. Put every included
    # transit route in natural route-label order (1, 2, 3, ... 8AX, 8BX, ...).
    records.sort(key=lambda r: natural_key(r["route_short_name"]))
    return records


def selected_direction_mask(
    df: pd.DataFrame,
    meta: dict[str, object],
    route_cols: list[str],
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for c in route_cols:
        mask &= df[c].astype(str) == str(meta[c])
    if "direction_id" in df.columns:
        mask &= df["direction_id"].astype(str) == str(meta.get("direction_id"))
    return mask


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(
    df: pd.DataFrame,
    output: Path,
    title: str | None = None,
    route_type_label: str | None = None,
    mapping_output: Path | None = None,
    dpi: int = 220,
    label_threshold: float = 6.0,
    highlighted_name_threshold: float = 3.0,
) -> pd.DataFrame:
    required = {
        "route_short_name",
        "route_long_name",
        "street",
        "street_share_pct",
        "street_length_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            f"{sorted(missing)}. This length-scaled chart requires street_length_m."
        )

    df = df.copy()
    df["street_share_pct"] = pd.to_numeric(df["street_share_pct"], errors="raise")
    df["street_length_m"] = pd.to_numeric(df["street_length_m"], errors="raise")
    if not df["street_length_m"].map(math.isfinite).all() or (df["street_length_m"] < 0).any():
        raise ValueError("street_length_m must contain finite non-negative numeric values")

    df = coalesce_streets(df)

    if route_type_label is not None:
        if "route_type_label" not in df.columns:
            raise ValueError("--route-type-label was given, but CSV has no route_type_label column")
        df = df[
            df["route_type_label"].astype(str).str.casefold() == route_type_label.casefold()
        ].copy()
        if df.empty:
            raise ValueError(f"No rows found for route_type_label={route_type_label!r}")

    route_cols = ["route_short_name", "route_long_name"]
    if "route_id" in df.columns:
        route_cols.insert(0, "route_id")

    directional = "direction_id" in df.columns
    total_cols = route_cols + (["direction_id"] if directional else [])
    sums = df.groupby(total_cols, dropna=False)["street_share_pct"].sum()
    bad = sums[(sums < 99.0) | (sums > 101.0)]
    if not bad.empty:
        sample = bad.head(5).to_dict()
        raise ValueError(
            "Street shares do not sum to about 100% for every route/direction. "
            "This script expects the route-street distance CSV. "
            f"Example totals: {sample}"
        )

    route_records = build_route_records(df, route_cols)
    if not route_records:
        raise ValueError("No routes remain after filtering/exclusions")

    route_lengths_m = [safe_float(r.get("route_length_m")) for r in route_records]
    max_route_length_m = max(route_lengths_m)
    if max_route_length_m <= 0:
        raise ValueError("Selected routes have zero total street_length_m")

    meters_per_mile = 1609.344
    max_route_length_mi = max_route_length_m / meters_per_mile

    n_routes = len(route_records)
    fig_height = max(8.0, 0.43 * n_routes + 3.2)
    fig, ax = plt.subplots(figsize=(17, fig_height))

    mapping_rows: list[dict[str, object]] = []
    y_positions: list[float] = []
    y_labels: list[str] = []

    for idx, meta in enumerate(route_records):
        y = float(idx)

        mask = selected_direction_mask(df, meta, route_cols)
        g = df.loc[mask].sort_values(
            ["street_length_m", "street"], ascending=[False, True]
        ).copy()

        primary = meta["primary_name_street"]
        secondary = meta["secondary_name_street"]
        named_share = float(meta["named_share_pct"])
        direction_display = str(meta.get("direction_display") or "")
        route_length_m = float(g["street_length_m"].sum())
        route_length_mi = route_length_m / meters_per_mile

        mapping_rows.append({
            "route_short_name": meta["route_short_name"],
            "route_long_name": meta["route_long_name"],
            "selected_direction_id": meta.get("direction_id", ""),
            "selected_direction": direction_display,
            "origin_terminal": meta.get("origin_terminal", ""),
            "destination_terminal": meta.get("destination_terminal", ""),
            "primary_name_street": primary or "",
            "primary_share_pct": float(meta["primary_share_pct"]),
            "secondary_name_street": secondary or "",
            "secondary_share_pct": float(meta["secondary_share_pct"]),
            "combined_named_share_pct": named_share,
            "selected_na_share_pct": float(meta.get("na_share_pct", 0.0)),
            "route_length_m": route_length_m,
            "route_length_miles": route_length_mi,
            "relative_to_longest_pct": 100.0 * route_length_m / max_route_length_m,
        })

        left_mi = 0.0
        suppress_later_regular_labels = False
        for row in g.itertuples(index=False):
            street = str(row.street)
            share_pct = float(row.street_share_pct)
            length_m = float(row.street_length_m)
            width_mi = length_m / meters_per_mile

            # Label-fit functions were tuned in percentage-of-full-row units.
            # Convert this segment's actual physical width to that same display
            # scale so a 10% segment of a short route is correctly treated as
            # narrower than a 10% segment of the longest route.
            display_width_pct = 100.0 * length_m / max_route_length_m

            is_primary = street == primary
            is_secondary = street == secondary
            is_highlighted = is_primary or is_secondary
            is_tunnel = is_tunnel_street(street)
            is_right_of_way = is_right_of_way_street(street)
            is_label_exempt = is_highlighted or is_tunnel or is_right_of_way

            if is_primary:
                face = "red"
            elif is_secondary:
                face = "gold"
            elif street == UNKNOWN_STREET:
                face = N_A_FACE
            elif is_tunnel:
                face = TUNNEL_FACE
            elif is_right_of_way:
                face = RIGHT_OF_WAY_FACE
            else:
                face = "white"

            ax.barh(
                y,
                width_mi,
                left=left_mi,
                height=0.72,
                color=face,
                edgecolor="0.45",
                linewidth=0.45,
            )

            marked = False
            if is_highlighted:
                street_label = None
                if display_width_pct >= highlighted_name_threshold:
                    street_label = fitted_street_name(street, display_width_pct)
                label = f"{street_label}\n{share_pct:.0f}%" if street_label else f"{share_pct:.0f}%"
                text_color = "white" if is_primary else "black"
                ax.text(
                    left_mi + width_mi / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.2 if not street_label else 6.5,
                    color=text_color,
                    fontweight="bold",
                    clip_on=False,
                    linespacing=0.9,
                    zorder=4,
                )
                marked = True
            elif is_tunnel or is_right_of_way:
                if display_width_pct >= label_threshold:
                    street_label = fitted_street_name(street, display_width_pct)
                    if street_label:
                        label = f"{street_label}\n{share_pct:.0f}%"
                        ax.text(
                            left_mi + width_mi / 2,
                            y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color="black",
                            fontweight="normal",
                            clip_on=True,
                            linespacing=0.9,
                        )
                        marked = True
            elif not suppress_later_regular_labels:
                if face == "white":
                    if display_width_pct >= label_threshold:
                        street_label = ordinary_street_name_if_fits(street, display_width_pct)
                        if street_label:
                            label = f"{street_label}\n{share_pct:.0f}%"
                            ax.text(
                                left_mi + width_mi / 2,
                                y,
                                label,
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                color="black",
                                fontweight="normal",
                                clip_on=True,
                                linespacing=0.9,
                            )
                            marked = True
                elif face == N_A_FACE and display_width_pct >= label_threshold:
                    street_label = fitted_street_name(street, display_width_pct)
                    if street_label:
                        label = f"{street_label}\n{share_pct:.0f}%"
                        ax.text(
                            left_mi + width_mi / 2,
                            y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color="black",
                            fontweight="normal",
                            clip_on=True,
                            linespacing=0.9,
                        )
                        marked = True

            # Segments are sorted largest-first by actual length. Once a regular
            # segment cannot be marked, later regular/N/A segments are smaller in
            # rendered width and remain blank. Named/tunnel/ROW labels are exempt.
            if not is_label_exempt and not marked:
                suppress_later_regular_labels = True

            left_mi += width_mi

        right_label = f"{route_length_mi:.1f} mi  ·  {named_share:.0f}% named"
        if directional and direction_display:
            right_label += f"  ·  {direction_display}"
        ax.text(
            route_length_mi + max_route_length_mi * 0.007,
            y,
            right_label,
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="0.25",
            clip_on=False,
        )

        short = str(meta["route_short_name"])
        long_name = smart_title(meta["route_long_name"])
        y_positions.append(y)
        y_labels.append(f"{short}  {long_name}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.8)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.set_ylim(max(y_positions) + 0.5, -0.5)
    ax.set_xlim(0, max_route_length_mi)
    ax.set_xlabel("Route distance (miles)")
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)

    legend = [
        Patch(facecolor="red", edgecolor="0.45", label="First street in route name"),
        Patch(facecolor="gold", edgecolor="0.45", label="Second street in route name"),
        Patch(facecolor=TUNNEL_FACE, edgecolor="0.45", label="Tunnel"),
        Patch(facecolor=RIGHT_OF_WAY_FACE, edgecolor="0.45", label="Right of way"),
        Patch(facecolor="white", edgecolor="0.45", label="Other street"),
        Patch(facecolor=N_A_FACE, edgecolor="0.45", label="N/A"),
    ]

    if title is None:
        title = "How long is each MUNI route, and which streets does it run on?"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.978)
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.960),
        ncol=6,
        frameon=False,
    )

    if directional:
        footer = (
            "Bars are proportional to selected-direction route length; the longest route fills the row. "
            "Directions use the same rule as the percentage chart: highest named-street share, then less N/A. "
            "All included routes are sorted by route label."
        )
    else:
        footer = (
            "Bars are proportional to route length; the longest route fills the row. "
            "All included routes are sorted by route label."
        )

    fig.text(0.5, 0.006, footer, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.956))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    mapping = pd.DataFrame(mapping_rows)
    if mapping_output is not None:
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(mapping_output, index=False)
    return mapping

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Normalized route-street CSV; directional data is auto-detected")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("muni_route_named_street_length.png"),
        help="Output image path (default: %(default)s)",
    )
    p.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Optional CSV containing selected direction, named shares, and route lengths",
    )
    p.add_argument(
        "--route-type-label",
        default=None,
        help="Optional exact route_type_label filter, e.g. bus",
    )
    p.add_argument("--title", default=None, help="Optional chart title")
    p.add_argument(
        "--label-threshold",
        type=float,
        default=6.0,
        help=(
            "Minimum rendered-width percentage for non-highlighted labels. Because bars have "
            "different total lengths, this is measured against the longest route's full row "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--highlighted-name-threshold",
        type=float,
        default=3.0,
        help="Minimum rendered-width percentage for showing a highlighted street name; smaller highlighted segments still show their route-local percentage (default: %(default)s)",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    mapping = make_plot(
        df,
        output=args.output,
        title=args.title,
        route_type_label=args.route_type_label,
        mapping_output=args.mapping_output,
        dpi=args.dpi,
        label_threshold=args.label_threshold,
        highlighted_name_threshold=args.highlighted_name_threshold,
    )
    print(f"Wrote {args.output}")
    if args.mapping_output:
        print(f"Wrote {args.mapping_output}")

    cols = [
        "route_short_name",
        "route_long_name",
        "selected_direction",
        "route_length_miles",
        "primary_name_street",
        "secondary_name_street",
        "combined_named_share_pct",
    ]
    cols = [c for c in cols if c in mapping.columns]
    print("\nRoutes in plotted order:")
    print(mapping[cols].to_string(index=False))


if __name__ == "__main__":
    main()
