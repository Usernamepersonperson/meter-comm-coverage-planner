"""End-to-end runs: planning for one technology and the full measured demo."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import optimize as opt
from .buildings import Penetration, load_buildings, with_penetration
from .coverage import LinkTable, build_links, diffraction_matrices, make_candidates
from .links import Tech, all_techs
from .network import evaluate
from .terrain import Dem

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = 0.95
K_MAX = 60
FADE_MARGIN_DB = 8.0
PERIOD_S = 900.0
DIFF_FREQ_MHZ = 868.5


@dataclass
class Scenario:
    dem: Dem
    b: pd.DataFrame
    cands: pd.DataFrame
    dey: np.ndarray | None = None
    ep: np.ndarray | None = None
    timings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR, pen: Penetration | None = None,
             buildings: str = "murten_buildings.csv", dem: str = "murten_dem_10m.npz") -> Scenario:
        d = Dem.load(Path(data_dir) / dem)
        b = load_buildings(Path(data_dir) / buildings, d, pen)
        return cls(d, b, make_candidates(b, d))

    def compute_terrain(self, n_profile: int = 100) -> None:
        t0 = time.perf_counter()
        self.dey, self.ep = diffraction_matrices(self.dem, self.cands, self.b, DIFF_FREQ_MHZ, n_profile)
        self.timings["terrain_diffraction_s"] = time.perf_counter() - t0

    def links(self, tech: Tech, *, model: str = "hata", env: str = "suburban", terrain: str = "deygout",
              fade_margin_db: float = FADE_MARGIN_DB, b: pd.DataFrame | None = None) -> LinkTable:
        diff = {"none": None, "deygout": self.dey, "ep": self.ep}[terrain]
        if terrain != "none" and diff is None:
            raise RuntimeError("call compute_terrain() first")
        return build_links(tech, self.cands, self.b if b is None else b, self.dem, model=model, env=env,
                           fade_margin_db=fade_margin_db, diff=diff)


def plan_sites(links: LinkTable, w: np.ndarray, target: float = TARGET, k_max: int = K_MAX, polish: bool = True):
    """Greedy curve, then swap local search at the size that first reaches the coverage target.

    Returns dict with order, curve, k1 (sites for target, 1-fold), k2 (2-fold), sites_k1 (polished).
    """
    A = links.feasible
    t0 = time.perf_counter()
    order = opt.greedy(A, w, k_max)
    t_greedy = time.perf_counter() - t0
    cv = opt.curve(A, w, order)
    k1 = opt.min_sites(cv, target, "cover1")
    k2 = opt.min_sites(cv, target, "cover2")
    out = {"order": order, "curve": cv, "k1": k1, "k2": k2, "greedy_s": t_greedy,
           "sites_k1": order[:k1] if k1 else order, "local_search_s": 0.0}
    if polish and k1:
        t0 = time.perf_counter()
        out["sites_k1"] = opt.local_search(A, w, order[:k1])
        out["local_search_s"] = time.perf_counter() - t0
    return out


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def run_plan(scn: Scenario, tech: Tech, out_dir: Path | None, *, target: float = TARGET, env: str = "suburban",
             model: str = "hata", terrain: str = "deygout") -> dict:
    """Plan one technology and (optionally) write GeoJSON, CSV and map."""
    links = scn.links(tech, model=model, env=env, terrain=terrain)
    w = scn.b["meters"].to_numpy(dtype=float)
    p = plan_sites(links, w, target)
    sites = p["sites_k1"]
    ev = evaluate(links, scn.b, sites, PERIOD_S)
    res = {"tech": tech.name, "target": target, "sites_for_target": p["k1"], "sites_2fold": p["k2"],
           "evaluation": ev, "sites": [int(s) for s in sites]}
    if out_dir is not None:
        from .export import write_gateways_csv, write_geojson  # noqa: PLC0415
        from .plots import coverage_map_png  # noqa: PLC0415

        out_dir.mkdir(parents=True, exist_ok=True)
        write_geojson(out_dir / "coverage.geojson", scn, links, sites)
        write_gateways_csv(out_dir / "gateways.csv", scn, links, sites)
        coverage_map_png(out_dir / "coverage_map.png", scn, links, sites)
        (out_dir / "plan.json").write_text(json.dumps(res, indent=1))
    return res


def _centroid_sites(scn: Scenario, k: int) -> list[int]:
    """Naive baseline: k-means centres of the meter locations, snapped to the nearest unused candidate."""
    from scipy.cluster.vq import kmeans2  # noqa: PLC0415

    xy = scn.b[["x", "y"]].to_numpy()
    cen, _ = kmeans2(xy - xy.mean(axis=0), k, minit="++", seed=7)
    cen = cen + xy.mean(axis=0)
    cx, cy = scn.cands["x"].to_numpy(), scn.cands["y"].to_numpy()
    chosen: list[int] = []
    for c in cen:
        d = np.hypot(cx - c[0], cy - c[1])
        d[chosen] = np.inf
        chosen.append(int(np.argmin(d)))
    return chosen


def run_demo(out_dir: Path, data_dir: Path = DATA_DIR, *, with_ilp: bool = True, figures: bool = True,
             ilp_seconds: float = 45.0, quiet: bool = False) -> dict:
    """The full measured run behind the README numbers."""
    from .export import write_gateways_csv, write_geojson  # noqa: PLC0415

    log = (lambda *a: None) if quiet else print
    t_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    scn = Scenario.load(data_dir)
    b = scn.b
    w = b["meters"].to_numpy(dtype=float)
    techs = all_techs()
    lora = techs["lorawan"]
    log(f"{len(b)} buildings, {int(w.sum())} meters, {len(scn.cands)} candidate sites")
    scn.compute_terrain()
    log(f"terrain diffraction {scn.timings['terrain_diffraction_s']:.0f}s")

    res: dict = {"setup": {
        "town": "Murten (FR)", "buildings": len(b), "meters": int(w.sum()),
        "candidates": len(scn.cands), "roof_candidates": int((scn.cands.kind == "roof").sum()),
        "pole_candidates": int((scn.cands.kind == "pole").sum()),
        "dem_cell_m": scn.dem.cell, "dem_min_m": float(scn.dem.z.min()), "dem_max_m": float(scn.dem.z.max()),
        "model": "Okumura-Hata suburban + Deygout 3-edge DEM diffraction", "fade_margin_db": FADE_MARGIN_DB,
        "indoor_db": 15.0, "cellar_db": 30.0, "cellar_share": 0.5, "report_period_s": PERIOD_S,
        "target": TARGET, "cellar_meter_share_realised": float(w[b["cellar"].to_numpy()].sum() / w.sum()),
    }}

    # ---- coverage vs gateways, every technology
    tech_rows = {}
    plans = {}
    for name, tech in techs.items():
        lk = scn.links(tech)
        p = plan_sites(lk, w, TARGET, K_MAX, polish=(name == "lorawan"))
        plans[name] = (lk, p)
        tech_rows[name] = {
            "mapl_db": {lv.name: round(tech.mapl_db(lv, FADE_MARGIN_DB), 1) for lv in tech.levels},
            "airtime_s": {lv.name: round(lv.airtime_s, 4) for lv in tech.levels},
            "sites_80": opt.min_sites(p["curve"], 0.80), "sites_90": opt.min_sites(p["curve"], 0.90),
            "sites_95": p["k1"], "sites_99": opt.min_sites(p["curve"], 0.99),
            "sites_95_2fold": p["k2"], "cover1_at_kmax": _round(p["curve"][-1]["cover1"]),
            "cover2_at_kmax": _round(p["curve"][-1]["cover2"]),
            "curve": [{"k": r["k"], "cover1": _round(r["cover1"]), "cover2": _round(r["cover2"])} for r in p["curve"]],
        }
        log(name, "sites for 95%:", p["k1"], "2-fold:", p["k2"], f"cover@{K_MAX}: {p['curve'][-1]['cover1']:.3f}")
    res["technologies"] = tech_rows

    lk, p = plans["lorawan"]
    A = lk.feasible
    k1 = p["k1"]
    k2 = p["k2"] if p["k2"] else len(p["order"])
    res["lorawan"] = {"k_target": k1, "k_target_2fold": p["k2"], "greedy_s": round(p["greedy_s"], 3),
                      "local_search_s": round(p["local_search_s"], 3)}

    # ---- optimiser comparison at a few sizes: greedy vs greedy+swap vs ILP
    comp = []
    for k in sorted({3, k1, k2}):
        if not k or k > len(p["order"]):
            continue
        g = p["order"][:k]
        ls = opt.local_search(A, w, g)
        row = {"k": k, "greedy": _round(opt.objective(A, w, g), 1), "swap": _round(opt.objective(A, w, ls), 1)}
        for nm, s in (("greedy", g), ("swap", ls)):
            st = opt.coverage_stats(A, w, s)
            row[f"{nm}_cover1"], row[f"{nm}_cover2"] = _round(st["cover1"]), _round(st["cover2"])
        if with_ilp:
            sites, info = opt.ilp(A, w, k, time_limit_s=ilp_seconds)
            if sites:
                st = opt.coverage_stats(A, w, sites)
                row.update({"ilp": _round(opt.objective(A, w, sites), 1), "ilp_cover1": _round(st["cover1"]),
                            "ilp_cover2": _round(st["cover2"]), "ilp_seconds": round(info["seconds"], 1),
                            "ilp_gap": _round(info["gap"], 5), "ilp_unique_rows": info["rows"],
                            "ilp_upper_bound": _round(info["upper_bound"], 1)})
        comp.append(row)
        log("optimiser", row)
    res["optimiser"] = comp

    # ---- compare plans on the network metrics
    sites_a = p["sites_k1"] if k1 else p["order"]
    sites_b = opt.local_search(A, w, p["order"][:k2])
    kk = len(sites_a)
    cand = scn.cands
    roofs = cand[cand.kind == "roof"].sort_values("h_ant", ascending=False).cid.tolist()
    tall = roofs[:kk]
    centroid = _centroid_sites(scn, kk)
    rng = np.random.default_rng(7)
    rand_cov = []
    for _ in range(300):
        s = rng.choice(len(cand), size=kk, replace=False)
        rand_cov.append(opt.coverage_stats(A, w, s))
    ev = {
        f"optimised_{kk}": evaluate(lk, b, sites_a, PERIOD_S),
        f"optimised_{len(sites_b)}_redundant": evaluate(lk, b, sites_b, PERIOD_S),
        f"tallest_roofs_{kk}": evaluate(lk, b, tall, PERIOD_S),
        f"demand_centroids_{kk}": evaluate(lk, b, centroid, PERIOD_S),
    }
    for e in ev.values():
        e.pop("gateway_load", None)
    res["plans"] = ev
    res["random_baseline"] = {
        "k": kk, "draws": 300,
        "cover1_mean": _round(np.mean([r["cover1"] for r in rand_cov])),
        "cover1_p10": _round(np.percentile([r["cover1"] for r in rand_cov], 10)),
        "cover1_p90": _round(np.percentile([r["cover1"] for r in rand_cov], 90)),
        "cover2_mean": _round(np.mean([r["cover2"] for r in rand_cov])),
    }
    full_eval = evaluate(lk, b, sites_a, PERIOD_S)
    res["lorawan"]["gateway_load"] = [{k: _round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                                      for r in full_eval["gateway_load"]]

    # ---- reporting-period sweep (airtime / duty / battery) for the target plan
    sweep = []
    for per in (300.0, 900.0, 3600.0):
        e = evaluate(lk, b, sites_a, per)
        sweep.append({"period_s": per, "pdr_mean": _round(e.get("pdr_mean")),
                      "pdr_worst_gateway": _round(e.get("pdr_worst_gateway")),
                      "duty_headroom_min": _round(e.get("duty_headroom_min")),
                      "battery_years_median": _round(e.get("battery_years_median"), 2),
                      "battery_years_p10": _round(e.get("battery_years_p10"), 2)})
    res["period_sweep"] = sweep

    # ---- validation: models against each other (no field data)
    val = {}
    d_h = scn.links(lora, model="hata", env="suburban", terrain="none").loss
    pen = b["pen_db"].to_numpy(dtype=np.float32)[None, :]
    base_loss = d_h - pen
    rows = []
    for model, env, label in (("fspl", "suburban", "free space"), ("hata", "open", "Hata open"),
                              ("hata", "suburban", "Hata suburban"), ("hata", "urban", "Hata urban"),
                              ("cost231", "urban", "COST-231 (extrapolated)")):
        L = scn.links(lora, model=model, env=env, terrain="none").loss - pen
        rows.append({"model": label, "mean_offset_vs_hata_suburban_db": _round(float((L - base_loss).mean()), 2),
                     "sd_db": _round(float((L - base_loss).std()), 2)})
    val["model_offsets"] = rows
    dd, ee = scn.dey, scn.ep
    obst = dd > 0
    val["diffraction"] = {
        "links_with_obstruction_share": _round(float(obst.mean())),
        "deygout_mean_db": _round(float(dd[obst].mean()), 2), "ep_mean_db": _round(float(ee[obst].mean()), 2),
        "mean_abs_diff_db": _round(float(np.abs(dd - ee)[obst].mean()), 2),
        "deygout_minus_ep_mean_db": _round(float((dd - ee)[obst].mean()), 2),
    }
    grid_rows = []
    for model, env, terrain, label in (
            ("hata", "suburban", "deygout", "Hata suburban + Deygout (base)"),
            ("hata", "suburban", "ep", "Hata suburban + Epstein-Peterson"),
            ("hata", "suburban", "none", "Hata suburban, no terrain"),
            ("hata", "urban", "deygout", "Hata urban + Deygout"),
            ("hata", "open", "deygout", "Hata open + Deygout"),
            ("cost231", "urban", "deygout", "COST-231 (extrapolated) + Deygout"),
            ("fspl", "suburban", "deygout", "Free space + Deygout"),
            ("fspl", "suburban", "none", "Free space only")):
        lkx = scn.links(lora, model=model, env=env, terrain=terrain)
        px = plan_sites(lkx, w, TARGET, K_MAX, polish=False)
        grid_rows.append({"scenario": label, "sites_95": px["k1"], "sites_95_2fold": px["k2"],
                          "cover1_at_5": _round(px["curve"][4]["cover1"]) if len(px["curve"]) >= 5 else None})
    val["model_scenarios"] = grid_rows
    sens = []
    for cellar_db, share, indoor in ((20, 0.5, 15), (30, 0.5, 15), (40, 0.5, 15), (30, 0.2, 15), (30, 0.8, 15),
                                     (30, 0.5, 5), (30, 0.5, 25)):
        pen_x = Penetration(indoor_db=indoor, cellar_db=cellar_db, cellar_share=share)
        bx = with_penetration(b, pen_x)
        lkx = scn.links(lora, b=bx)
        px = plan_sites(lkx, w, TARGET, K_MAX, polish=False)
        sens.append({"cellar_db": cellar_db, "cellar_share": share, "indoor_db": indoor,
                     "sites_95": px["k1"], "sites_95_2fold": px["k2"],
                     "cover1_at_5": _round(px["curve"][4]["cover1"]) if len(px["curve"]) >= 5 else None,
                     "cover1_at_10": _round(px["curve"][9]["cover1"]) if len(px["curve"]) >= 10 else None})
    val["penetration_sensitivity"] = sens
    res["validation"] = val

    # ---- outputs
    sites = sites_a
    write_geojson(out_dir / "coverage.geojson", scn, lk, sites)
    write_gateways_csv(out_dir / "gateways.csv", scn, lk, sites)
    res["runtime_s"] = {**{k: round(v, 1) for k, v in scn.timings.items()},
                        "greedy_curve_60_s": round(p["greedy_s"], 2),
                        "swap_search_s": round(p["local_search_s"], 2),
                        "total_s": round(time.perf_counter() - t_all, 1)}
    res["sites_cid"] = [int(cand.loc[s, "cid"]) for s in sites]
    if figures:
        from .plots import coverage_map_png, figure_main, figure_post  # noqa: PLC0415

        coverage_map_png(out_dir / "coverage_map.png", scn, lk, sites)
        figure_main(out_dir / "figure.png", scn, lk, sites, res)
        figure_post(out_dir / "post-figure.png", scn, lk, sites, res)
    res["runtime_s"]["total_s"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    from .report import write_report  # noqa: PLC0415

    write_report(out_dir / "results.md", res)
    log(f"done in {res['runtime_s']['total_s']}s -> {out_dir}")
    return res
