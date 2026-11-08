"""Command line: `python -m meter_coverage demo` and `python -m meter_coverage plan`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .buildings import Penetration
from .links import all_techs
from .pipeline import DATA_DIR, Scenario, run_demo, run_plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="meter_coverage", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="reproduce the README numbers and figures on the bundled town")
    d.add_argument("--out", type=Path, default=Path("output"))
    d.add_argument("--data", type=Path, default=DATA_DIR)
    d.add_argument("--no-ilp", action="store_true", help="skip the exact ILP comparison")
    d.add_argument("--no-figures", action="store_true")
    d.add_argument("--ilp-seconds", type=float, default=45.0)

    p = sub.add_parser("plan", help="choose gateways for one technology")
    p.add_argument("--tech", choices=sorted(all_techs()), default="lorawan")
    p.add_argument("--target", type=float, default=0.95, help="share of meters to cover (0-1)")
    p.add_argument("--env", choices=["urban", "suburban", "open"], default="suburban")
    p.add_argument("--model", choices=["hata", "cost231", "fspl"], default="hata")
    p.add_argument("--terrain", choices=["deygout", "ep", "none"], default="deygout")
    p.add_argument("--cellar-db", type=float, default=30.0)
    p.add_argument("--indoor-db", type=float, default=15.0)
    p.add_argument("--cellar-share", type=float, default=0.5)
    p.add_argument("--buildings", default="murten_buildings.csv", help="CSV in --data with GWR columns")
    p.add_argument("--dem", default="murten_dem_10m.npz", help="npz in --data (z, x0, y0, cell)")
    p.add_argument("--data", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=Path("output"))

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        run_demo(a.out, a.data, with_ilp=not a.no_ilp, figures=not a.no_figures, ilp_seconds=a.ilp_seconds)
        return 0
    if not 0 < a.target <= 1:
        ap.error("--target must be in (0, 1]")
    scn = Scenario.load(a.data, Penetration(a.indoor_db, a.cellar_db, a.cellar_share), a.buildings, a.dem)
    if a.terrain != "none":
        scn.compute_terrain()
    res = run_plan(scn, all_techs()[a.tech], a.out, target=a.target, env=a.env, model=a.model, terrain=a.terrain)
    ev = res["evaluation"]
    print(json.dumps({"gateways": ev["gateways"], "cover1": ev["cover1"], "cover2": ev["cover2"],
                      "sites_for_target": res["sites_for_target"]}, indent=1))
    return 0
