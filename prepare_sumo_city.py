#!/usr/bin/env python3
"""Prepare an OSM-backed SUMO city scenario.

The script downloads a small OpenStreetMap bounding box, converts it into a
SUMO network, generates simple vehicle and pedestrian demand, and writes a
gNB layout file that can be loaded by senario_multi_gnodeb.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen


SUMO_HOME = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo"))
SUMO_TOOLS = SUMO_HOME / "tools"
OSM_GET = SUMO_TOOLS / "osmGet.py"
OSM_BUILD = SUMO_TOOLS / "osmBuild.py"
RANDOM_TRIPS = SUMO_TOOLS / "randomTrips.py"


DEFAULT_CITY_BBOXES = {
    # west,south,east,north. Kept deliberately compact for fast Overpass/SUMO runs.
    "berlin_mitte": "13.3990,52.5165,13.4105,52.5235",
    "paris_center": "2.3320,48.8530,2.3650,48.8690",
    "tunis_center": "10.1690,36.7920,10.1950,36.8140",
    "madrid_center": "-3.7150,40.4080,-3.6850,40.4280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and build a SUMO city scenario from OSM.")
    parser.add_argument("--city", default="berlin_mitte", help="Scenario/city folder name.")
    parser.add_argument(
        "--bbox",
        default=None,
        help="OSM bbox as west,south,east,north. Defaults to a known bbox for --city when available.",
    )
    parser.add_argument("--output-root", default="scenario/mobility")
    parser.add_argument(
        "--overpass-url",
        default="overpass-api.de/api/interpreter",
        help="Overpass endpoint used by SUMO osmGet.py.",
    )
    parser.add_argument(
        "--download-method",
        choices=["auto", "osm-get", "osm-api"],
        default="auto",
        help="Use SUMO osmGet.py, the OSM map API, or try osmGet then OSM API.",
    )
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=1800)
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--vehicle-period", type=float, default=4.0)
    parser.add_argument("--person-period", type=float, default=8.0)
    parser.add_argument(
        "--vehicle-classes",
        choices=["passenger", "publicTransport", "road", "all"],
        default="road",
        help="Network edge classes passed to SUMO osmBuild.py.",
    )
    parser.add_argument(
        "--pedestrians",
        action="store_true",
        help="Build pedestrian infrastructure and generate person routes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gnbs", type=int, default=4)
    parser.add_argument("--fixed-ues", type=int, default=80)
    parser.add_argument("--prbs-per-gnb", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated files.")
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def download_with_osm_api(bbox: str, output_path: Path) -> None:
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox}"
    print(f"Downloading {url}")
    request = Request(
        url,
        headers={
            "User-Agent": "network-slicing-sumo-scenario-prep/1.0",
        },
    )
    with urlopen(request, timeout=240) as response:
        output_path.write_bytes(response.read())


def ensure_tools() -> None:
    missing = [str(path) for path in [OSM_GET, OSM_BUILD, RANDOM_TRIPS] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SUMO OSM tools: "
            + ", ".join(missing)
            + ". Set SUMO_HOME or install the SUMO tools package."
        )


def parse_conv_boundary(net_path: Path) -> tuple[float, float, float, float]:
    root = ET.parse(net_path).getroot()
    location = root.find("location")
    if location is None or not location.get("convBoundary"):
        raise ValueError(f"Could not read convBoundary from {net_path}")
    x_min, y_min, x_max, y_max = [float(v) for v in location.get("convBoundary").split(",")]
    return x_min, y_min, x_max, y_max


def build_gnb_layout(bounds: tuple[float, float, float, float], n_gnbs: int, prbs_per_gnb: int):
    x_min, y_min, x_max, y_max = bounds
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    radius = 0.38 * max(width, height)

    if n_gnbs <= 1:
        positions = [(cx, cy)]
    elif n_gnbs == 2:
        positions = [(x_min + 0.30 * width, cy), (x_min + 0.70 * width, cy)]
    elif n_gnbs == 3:
        positions = [
            (x_min + 0.25 * width, y_min + 0.30 * height),
            (x_min + 0.75 * width, y_min + 0.30 * height),
            (cx, y_min + 0.75 * height),
        ]
    elif n_gnbs == 4:
        positions = [
            (x_min + 0.25 * width, y_min + 0.25 * height),
            (x_min + 0.75 * width, y_min + 0.25 * height),
            (x_min + 0.25 * width, y_min + 0.75 * height),
            (x_min + 0.75 * width, y_min + 0.75 * height),
        ]
    else:
        positions = [(cx, cy)]
        ring = n_gnbs - 1
        ring_radius = 0.32 * max(width, height)
        for idx in range(ring):
            angle = 2.0 * 3.141592653589793 * idx / ring
            positions.append((cx + ring_radius * math.cos(angle), cy + ring_radius * math.sin(angle)))

    return {
        "gnb_positions": [[round(x, 3), round(y, 3)] for x, y in positions],
        "coverage_radius": [round(radius, 3)] * len(positions),
        "carrier_ids": [0] * len(positions),
        "max_prbs_per_gnb": [int(prbs_per_gnb)] * len(positions),
    }


def write_sumocfg(path: Path, net_file: str, route_files: list[str], begin: int, end: int, step_length: float) -> None:
    route_value = ",".join(route_files)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_value}"/>
    </input>
    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="{step_length}"/>
    </time>
</configuration>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_tools()

    bbox = args.bbox or DEFAULT_CITY_BBOXES.get(args.city)
    if not bbox:
        valid = ", ".join(sorted(DEFAULT_CITY_BBOXES))
        raise ValueError(f"No bbox supplied for city '{args.city}'. Known city defaults: {valid}")

    if args.n_gnbs <= 0:
        raise ValueError("--n-gnbs must be positive")
    if args.fixed_ues <= 0:
        raise ValueError("--fixed-ues must be positive")

    out_dir = Path(args.output_root) / args.city
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.city
    osm_file = out_dir / f"{prefix}_bbox.osm.xml"
    net_file = out_dir / f"{prefix}.net.xml"
    vehicles_file = out_dir / "vehicles.rou.xml"
    vehicle_trips_file = out_dir / "vehicles.trips.xml"
    persons_file = out_dir / "persons.rou.xml"
    person_trips_file = out_dir / "persons.trips.xml"
    sumocfg_file = out_dir / "sim.sumocfg"
    layout_file = out_dir / "gnb_layout.json"

    if osm_file.exists() and not args.force:
        print(f"Using existing {osm_file}")
    else:
        if args.download_method in {"auto", "osm-get"}:
            run_command([
                sys.executable,
                str(OSM_GET),
                "--bbox",
                bbox,
                "--prefix",
                prefix,
                "--output-dir",
                str(out_dir),
                "--url",
                args.overpass_url,
            ])

        if (
            args.download_method == "osm-api"
            or (
                args.download_method == "auto"
                and (not osm_file.exists() or osm_file.stat().st_size == 0)
            )
        ):
            download_with_osm_api(bbox, osm_file)

    if not osm_file.exists() or osm_file.stat().st_size == 0:
        raise RuntimeError(
            f"OSM download did not create {osm_file}. "
            "Try a smaller --bbox or another --overpass-url."
        )

    if net_file.exists() and not args.force:
        print(f"Using existing {net_file}")
    else:
        run_command([
            sys.executable,
            str(OSM_BUILD),
            "--osm-file",
            str(osm_file),
            "--prefix",
            prefix,
            "--output-directory",
            str(out_dir),
            "--vehicle-classes",
            args.vehicle_classes,
        ] + (["--pedestrians"] if args.pedestrians else []))

    if not net_file.exists() or net_file.stat().st_size == 0:
        raise RuntimeError(f"SUMO network build did not create {net_file}.")

    route_files = [vehicles_file.name]

    run_command([
        sys.executable,
        str(RANDOM_TRIPS),
        "--net-file",
        str(net_file),
        "--route-file",
        str(vehicles_file),
        "--output-trip-file",
        str(vehicle_trips_file),
        "--begin",
        str(args.begin),
        "--end",
        str(args.end),
        "--period",
        str(args.vehicle_period),
        "--seed",
        str(args.seed),
        "--validate",
        "--vehicle-class",
        "passenger",
        "--prefix",
        "veh",
    ])

    if args.pedestrians and args.person_period > 0:
        run_command([
            sys.executable,
            str(RANDOM_TRIPS),
            "--net-file",
            str(net_file),
            "--route-file",
            str(persons_file),
            "--output-trip-file",
            str(person_trips_file),
            "--begin",
            str(args.begin),
            "--end",
            str(args.end),
            "--period",
            str(args.person_period),
            "--seed",
            str(args.seed + 1),
            "--validate",
            "--pedestrians",
            "--prefix",
            "person",
        ])
        route_files.append(persons_file.name)

    write_sumocfg(
        sumocfg_file,
        net_file=net_file.name,
        route_files=route_files,
        begin=args.begin,
        end=args.end,
        step_length=args.step_length,
    )

    layout = {
        "city": args.city,
        "bbox": bbox,
        "sumo_config_path": str(sumocfg_file),
        "sumo_net_path": str(net_file),
        "n_ues": int(args.fixed_ues),
        "slices": (
            [{"type": "eMBB", "count": 3}, {"type": "URLLC", "count": 1}]
            if args.pedestrians
            else [{"type": "eMBB", "count": 4}]
        ),
        **build_gnb_layout(parse_conv_boundary(net_file), args.n_gnbs, args.prbs_per_gnb),
    }
    layout_file.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    print("\nPrepared SUMO city scenario")
    print(f"- OSM:        {osm_file}")
    print(f"- network:    {net_file}")
    print(f"- routes:     {', '.join(route_files)}")
    print(f"- SUMO config:{sumocfg_file}")
    print(f"- gNB layout: {layout_file}")
    print("\nUse in training as scenario_sumo_osm_city after this file exists.")


if __name__ == "__main__":
    main()
