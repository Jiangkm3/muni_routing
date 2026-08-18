#!/usr/bin/env python3
"""Plot the share of Muni stops on the street(s) each route is named after.

Each route is reduced to three pieces that sum to 100%:
  * first street in the route name (red)
  * second street in the route name (gold), when present
  * Other (white) = 100 - first - second

Street/other routes are ranked by combined named-street stop share. Neighborhood-
named routes are shown in a separate lower section and sorted naturally by route
label. Rail-replacement BUS/OWL services and route 714 are excluded; numeric
night routes 90 and 91 remain.

Example:
    python plot_muni_route_named_stops.py muni_stop_streets.csv \
        --output muni_route_named_stop_share.png
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
# Street normalization and route-name matching
# ---------------------------------------------------------------------------

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

STREET_SUFFIXES = {
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd",
    "drive", "dr", "way", "terrace", "ter", "court", "ct", "lane", "ln",
    "highway", "hwy", "freeway", "fwy", "parkway", "pkwy", "circle", "cir",
    "place", "pl", "plaza", "tunnel", "ramp",
}

SUFFIX_CANON = {
    "street": "street", "st": "street",
    "avenue": "avenue", "ave": "avenue",
    "boulevard": "boulevard", "blvd": "boulevard",
    "road": "road", "rd": "road",
    "drive": "drive", "dr": "drive",
    "terrace": "terrace", "ter": "terrace",
    "court": "court", "ct": "court",
    "lane": "lane", "ln": "lane",
    "highway": "highway", "hwy": "highway",
    "freeway": "freeway", "fwy": "freeway",
    "parkway": "parkway", "pkwy": "parkway",
    "circle": "circle", "cir": "circle",
    "place": "place", "pl": "place",
    "way": "way",
}

ROUTE_CORE_ALIASES = {
    "third": "3rd",
}

# Exact overrides preserve route-name order where that is clearer than inference.
ROUTE_NAME_OVERRIDES: dict[str, list[str] | None] = {
    "CA": ["California Street"],
    "49": ["Van Ness Avenue", "Mission Street"],
    "48": ["Quintara Street", "24th Street"],
    "F": ["Market Street", "The Embarcadero"],
}

# Section placement only. These routes still go through literal street matching.
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
]

LABEL_SUFFIXES = {
    "boulevard", "blvd", "avenue", "ave", "street", "st", "road", "rd",
    "drive", "dr", "highway", "hwy", "parkway", "pkwy", "terrace", "ter",
    "way",
}
LABEL_CHARS_PER_PERCENT = 2.2


def normalize_street_label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    whole = STREET_ALIASES_CASEFOLD.get(text.casefold())
    if whole is not None:
        return whole
    parts = [p.strip() for p in text.split(" / ") if p.strip()]
    normalized: list[str] = []
    for part in parts:
        item = STREET_ALIASES_CASEFOLD.get(part.casefold(), part)
        if item not in normalized:
            normalized.append(item)
    return " / ".join(normalized)


def _basic_text(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _street_core(street: str) -> str:
    words = _basic_text(street).split()
    while words and words[-1] in STREET_SUFFIXES:
        words.pop()
    core = " ".join(words)
    return ROUTE_CORE_ALIASES.get(core, core)


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
    return [p.strip() for p in re.split(r"\s*(?:-|&)\s*", cleaned) if p.strip()][:2]


def _component_core(component: str) -> str:
    words = _basic_text(component).split()
    while words and words[-1] in STREET_SUFFIXES:
        words.pop()
    core = " ".join(words)
    return ROUTE_CORE_ALIASES.get(core, core)


def _canonical_with_suffix(text: str) -> str:
    """Comparison form that preserves an explicitly supplied street suffix."""
    words = _basic_text(text).split()
    if not words:
        return ""
    if words[-1] in SUFFIX_CANON:
        words[-1] = SUFFIX_CANON[words[-1]]
    return " ".join(words)


def _component_has_explicit_suffix(component: str) -> bool:
    words = _basic_text(component).split()
    return bool(words and words[-1] in SUFFIX_CANON)


def infer_named_streets(
    route_short_name: str,
    route_long_name: str,
    streets: Iterable[str],
) -> list[str]:
    """Infer zero, one, or two literal route-name streets, preserving name order."""
    short = str(route_short_name)
    available = [normalize_street_label(s) for s in streets]
    available = [s for s in dict.fromkeys(available) if s]

    if short in ROUTE_NAME_OVERRIDES:
        override = ROUTE_NAME_OVERRIDES[short]
        if not override:
            return []
        present = {s.casefold(): s for s in available}
        return [present[x.casefold()] for x in override if x.casefold() in present][:2]

    by_core: dict[str, list[str]] = {}
    for street in available:
        by_core.setdefault(_street_core(street), []).append(street)

    matched: list[str] = []
    for component in _route_components(route_long_name):
        chosen: str | None = None

        # If the route name itself says Street/Avenue/etc., prefer that exact road
        # type before falling back to suffix-free matching. This avoids, for example,
        # confusing 24th Street with 24th Avenue.
        if _component_has_explicit_suffix(component):
            exact_form = _canonical_with_suffix(component)
            exact = [s for s in available if _canonical_with_suffix(s) == exact_form]
            if exact:
                chosen = sorted(exact, key=lambda s: (len(s), s.casefold()))[0]

        if chosen is None:
            candidates = by_core.get(_component_core(component), [])
            if candidates:
                chosen = sorted(candidates, key=lambda s: (len(s), s.casefold()))[0]

        if chosen is not None and chosen not in matched:
            matched.append(chosen)
        if len(matched) == 2:
            break
    return matched


def is_neighborhood_named(route_short_name: object) -> bool:
    return str(route_short_name).upper() in NEIGHBORHOOD_NAMED_ROUTES


def is_excluded_route(route_short_name: object) -> bool:
    short = str(route_short_name).strip().upper()
    return short == "714" or short.endswith("BUS") or short.endswith("OWL")


def natural_key(value: object) -> list[object]:
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x.casefold() for x in parts]


def smart_title(value: object) -> str:
    text = str(value).title()
    return re.sub(
        r"\b(\d+)(St|Nd|Rd|Th)\b",
        lambda m: f"{m.group(1)}{m.group(2).lower()}",
        text,
    )


def abbreviated_street_name(street: str) -> str:
    """Use the same compact street-name display as the route-distance graph."""
    s = smart_title(street)
    for pattern, repl in LABEL_ABBREVIATIONS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def strip_street_suffix(street: str) -> str:
    """Drop a trailing road-type suffix for a tighter highlighted label."""
    words = abbreviated_street_name(street).split()
    if words and words[-1].casefold().rstrip(".") in LABEL_SUFFIXES:
        words = words[:-1]
    return " ".join(words).strip()


def fitted_street_name(street: str, width_pct: float) -> str | None:
    """Try compact name, then suffix-free name; return None if neither fits."""
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


# ---------------------------------------------------------------------------
# Data reduction
# ---------------------------------------------------------------------------


def build_route_records(df: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "route_id", "route_short_name", "route_long_name", "street",
        "stop_share_pct", "route_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    work = df.copy()
    route_type_num = pd.to_numeric(work["route_type"], errors="coerce")
    work = work[route_type_num.isin({0, 3, 5})].copy()
    if work.empty:
        raise ValueError("No GTFS route_type 0/3/5 transit rows found")
    work["street"] = work["street"].map(normalize_street_label)

    records: list[dict[str, object]] = []
    for (route_id, short, long_name), g in work.groupby(
        ["route_id", "route_short_name", "route_long_name"], dropna=False, sort=False
    ):
        if is_excluded_route(short):
            continue

        named = infer_named_streets(str(short), str(long_name), g["street"].tolist())
        lookup = g.groupby("street", dropna=False)["stop_share_pct"].sum().to_dict()

        primary = named[0] if len(named) >= 1 else None
        secondary = named[1] if len(named) >= 2 else None
        primary_share = float(lookup.get(primary, 0.0)) if primary else 0.0
        secondary_share = float(lookup.get(secondary, 0.0)) if secondary else 0.0
        named_share = primary_share + secondary_share

        if named_share > 100.05:
            raise ValueError(
                f"Named stop shares exceed 100% for route {short}: {named_share:.3f}%"
            )
        other_share = max(0.0, 100.0 - named_share)

        records.append({
            "route_id": route_id,
            "route_short_name": short,
            "route_long_name": long_name,
            "primary_name_street": primary,
            "primary_share_pct": primary_share,
            "secondary_name_street": secondary,
            "secondary_share_pct": secondary_share,
            "combined_named_share_pct": named_share,
            "other_share_pct": other_share,
            "is_neighborhood_named": is_neighborhood_named(short),
        })

    street_named = [r for r in records if not r["is_neighborhood_named"]]
    street_named.sort(
        key=lambda r: (-float(r["combined_named_share_pct"]), natural_key(r["route_short_name"]))
    )
    neighborhood = [r for r in records if r["is_neighborhood_named"]]
    neighborhood.sort(key=lambda r: natural_key(r["route_short_name"]))
    return street_named + neighborhood


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _segment_label(name: str, pct: float) -> str:
    return f"{name}\n{pct:.0f}%"


def make_plot(
    df: pd.DataFrame,
    output: Path,
    title: str | None = None,
    mapping_output: Path | None = None,
    dpi: int = 220,
    highlighted_name_threshold: float = 3.0,
) -> pd.DataFrame:
    route_records = build_route_records(df)
    n_routes = len(route_records)
    n_neighborhood = sum(bool(r["is_neighborhood_named"]) for r in route_records)
    n_primary = n_routes - n_neighborhood
    separator_gap = 1.0 if n_neighborhood and n_primary else 0.0

    fig_height = max(8.0, 0.43 * n_routes + 3.2)
    fig, ax = plt.subplots(figsize=(17, fig_height))

    y_positions: list[float] = []
    y_labels: list[str] = []
    mapping_rows: list[dict[str, object]] = []

    for idx, rec in enumerate(route_records):
        y = float(idx)
        if separator_gap and idx >= n_primary:
            y += separator_gap

        primary = rec["primary_name_street"]
        secondary = rec["secondary_name_street"]
        primary_share = float(rec["primary_share_pct"])
        secondary_share = float(rec["secondary_share_pct"])
        other_share = float(rec["other_share_pct"])
        named_share = float(rec["combined_named_share_pct"])

        left = 0.0
        segments = [
            (primary, primary_share, "red", "white", True),
            (secondary, secondary_share, "gold", "black", True),
            ("Other", other_share, "white", "black", False),
        ]

        for name, width, face, text_color, is_named in segments:
            if width <= 1e-9:
                continue
            ax.barh(
                y,
                width,
                left=left,
                height=0.72,
                color=face,
                edgecolor="0.45",
                linewidth=0.45,
            )

            if name:
                if is_named:
                    # Match the route-distance graph's highlighted-label rule:
                    # compact name if it fits, then suffix-free short name, then
                    # percentage only when even the short name is too wide.
                    street_label = None
                    if width >= highlighted_name_threshold:
                        street_label = fitted_street_name(str(name), width)
                    label = (
                        f"{street_label}\n{width:.0f}%"
                        if street_label
                        else f"{width:.0f}%"
                    )
                    fs = 6.2 if not street_label else 6.5
                else:
                    label = _segment_label("Other", width)
                    fs = 6.5

                ax.text(
                    left + width / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=text_color,
                    fontweight="bold" if is_named else "normal",
                    clip_on=False if is_named else True,
                    linespacing=0.9,
                    zorder=4,
                )
            left += width

        ax.text(
            100.7,
            y,
            f"{named_share:.0f}%",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="0.25",
            clip_on=False,
        )

        short = str(rec["route_short_name"])
        long_name = smart_title(rec["route_long_name"])
        y_positions.append(y)
        y_labels.append(f"{short}  {long_name}")

        mapping_rows.append({
            "route_short_name": rec["route_short_name"],
            "route_long_name": rec["route_long_name"],
            "route_category": "neighborhood" if rec["is_neighborhood_named"] else "street/other",
            "primary_name_street": primary or "",
            "primary_share_pct": primary_share,
            "secondary_name_street": secondary or "",
            "secondary_share_pct": secondary_share,
            "combined_named_share_pct": named_share,
            "other_share_pct": other_share,
        })

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.8)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.set_ylim(max(y_positions) + 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of route stops (%)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)

    if separator_gap and n_primary < len(y_positions):
        prev_y = y_positions[n_primary - 1]
        next_y = y_positions[n_primary]
        separator_y = (prev_y + next_y) / 2
        ax.axhline(separator_y, color="0.25", linewidth=1.2)
        ax.text(
            0,
            separator_y + 0.36,
            "Neighborhood-named routes",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    legend = [
        Patch(facecolor="red", edgecolor="0.45", label="First street in route name"),
        Patch(facecolor="gold", edgecolor="0.45", label="Second street in route name"),
        Patch(facecolor="white", edgecolor="0.45", label="Other"),
    ]

    if title is None:
        title = "What percentage of each MUNI route's stops are on the street(s) it is named after?"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.978)
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.960),
        ncol=3,
        frameon=False,
    )

    footer = (
        "Street/other routes are ranked by combined red + yellow share; neighborhood routes are sorted by route name."
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
    p.add_argument("csv", type=Path, help="Muni stop-street percentage CSV")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("muni_route_named_stop_share.png"),
        help="Output image path (default: %(default)s)",
    )
    p.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Optional CSV containing inferred stop-street matches and shares",
    )
    p.add_argument("--title", default=None, help="Optional chart title")
    p.add_argument(
        "--highlighted-name-threshold",
        type=float,
        default=3.0,
        help=(
            "Minimum named-street percentage for attempting a street-name label; "
            "smaller red/yellow sections show percentage only (default: %(default)s)"
        ),
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
        mapping_output=args.mapping_output,
        dpi=args.dpi,
        highlighted_name_threshold=args.highlighted_name_threshold,
    )
    print(f"Wrote {args.output}")
    if args.mapping_output:
        print(f"Wrote {args.mapping_output}")

    cols = [
        "route_short_name",
        "route_long_name",
        "route_category",
        "primary_name_street",
        "secondary_name_street",
        "combined_named_share_pct",
        "other_share_pct",
    ]
    print("\nRoutes in plotted order:")
    print(mapping[cols].to_string(index=False))


if __name__ == "__main__":
    main()
