"""Markdown report from a results dict."""

from __future__ import annotations

from pathlib import Path

NAN = float("nan")


def _p(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _k(x, kmax=60):
    return f">{kmax}" if x is None else str(x)


def write_report(path: Path, r: dict) -> None:
    s, t, v = r["setup"], r["technologies"], r["validation"]
    out = ["# Results (measured by python -m meter_coverage demo)\n"]
    out.append(f"Town: {s['town']}; {s['buildings']} buildings with {s['meters']} assumed meters; "
               f"{s['candidates']} candidate sites ({s['roof_candidates']} rooftops, {s['pole_candidates']} poles). "
               f"Model: {s['model']}; fade margin {s['fade_margin_db']} dB; entry loss {s['indoor_db']} dB indoor / "
               f"{s['cellar_db']} dB cellar ({100 * s['cellar_share']:.0f}% of buildings, assumed). "
               "All coverage is modelled, nothing is field-measured.\n")
    out.append("## Gateways needed (greedy, meter-weighted)\n")
    out.append("| technology | 80% | 90% | 95% | 99% | 95% with 2-fold | coverage at 60 |\n"
               "|---|---|---|---|---|---|---|")
    for n, row in t.items():
        out.append(f"| {n} | {_k(row['sites_80'])} | {_k(row['sites_90'])} | {_k(row['sites_95'])} | "
                   f"{_k(row['sites_99'])} | {_k(row['sites_95_2fold'])} | {_p(row['cover1_at_kmax'])} |")
    out.append("\n## Plans compared (LoRaWAN, one uplink per meter every 15 min)\n")
    out.append("| plan | gateways | covered | 2-fold | median battery (y) | p10 battery (y) | mean airtime (s) | "
               "duty headroom | mean PDR (ALOHA) |\n|---|---|---|---|---|---|---|---|---|")
    for n, e in r["plans"].items():
        out.append(f"| {n} | {e['gateways']} | {_p(e['cover1'])} | {_p(e['cover2'])} | "
                   f"{e.get('battery_years_median', NAN):.1f} | {e.get('battery_years_p10', NAN):.1f} | "
                   f"{e.get('mean_airtime_s', NAN):.3f} | {_p(e.get('duty_headroom_min'))} | "
                   f"{_p(e.get('pdr_mean'), 2)} |")
    out.append("\nSpreading-factor mix (share of meters):\n")
    out.append("| plan | SF7 | SF8 | SF9 | SF10 | SF11 | SF12 | uncovered |\n|---|---|---|---|---|---|---|---|")
    for n, e in r["plans"].items():
        m = e["level_mix"]
        out.append(f"| {n} | " + " | ".join(_p(m.get(k)) for k in
                                           ("SF7", "SF8", "SF9", "SF10", "SF11", "SF12", "uncovered")) + " |")
    rb = r["random_baseline"]
    out.append(f"\nRandom {rb['k']}-site sets ({rb['draws']} draws): covered {_p(rb['cover1_mean'])} on average "
               f"(p10 {_p(rb['cover1_p10'])}, p90 {_p(rb['cover1_p90'])}), 2-fold {_p(rb['cover2_mean'])}.\n")
    out.append("## Optimiser check (objective = meters covered once + 0.5 x covered twice)\n")
    out.append("| sites | greedy | greedy+swap | ILP (60 s limit) | ILP upper bound | ILP s |\n"
               "|---|---|---|---|---|---|")
    for o in r["optimiser"]:
        out.append(f"| {o['k']} | {o['greedy']} | {o['swap']} | {o.get('ilp', 'n/a')} | "
                   f"{o.get('ilp_upper_bound', 'n/a')} | "
                   f"{o.get('ilp_seconds', 'n/a')} |")
    out.append("\n## Model against model (no field data)\n")
    out.append("| scenario | gateways for 95% | gateways for 95% 2-fold | covered by 5 |\n|---|---|---|---|")
    for g in v["model_scenarios"]:
        out.append(f"| {g['scenario']} | {_k(g['sites_95'])} | {_k(g['sites_95_2fold'])} | {_p(g['cover1_at_5'])} |")
    d = v["diffraction"]
    out.append(f"\nTerrain: {_p(d['links_with_obstruction_share'])} of gateway-building links have a DEM obstruction; "
               f"mean loss where obstructed Deygout {d['deygout_mean_db']} dB vs Epstein-Peterson {d['ep_mean_db']} dB "
               f"(mean absolute difference {d['mean_abs_diff_db']} dB).\n")
    out.append("| model | mean offset vs Hata suburban (dB) | sd (dB) |\n|---|---|---|")
    for m in v["model_offsets"]:
        out.append(f"| {m['model']} | {m['mean_offset_vs_hata_suburban_db']} | {m['sd_db']} |")
    out.append("\n## Sensitivity to entry loss\n")
    out.append("| cellar dB | cellar share | indoor dB | gateways for 95% | 95% 2-fold | covered by 5 | "
               "covered by 10 |\n"
               "|---|---|---|---|---|---|---|")
    for x in v["penetration_sensitivity"]:
        out.append(f"| {x['cellar_db']} | {x['cellar_share']} | {x['indoor_db']} | {_k(x['sites_95'])} | "
                   f"{_k(x['sites_95_2fold'])} | {_p(x['cover1_at_5'])} | {_p(x['cover1_at_10'])} |")
    out.append("\n## Period sweep (target plan)\n")
    out.append("| period (s) | mean PDR | worst gateway PDR | duty headroom | median battery (y) | p10 battery (y) |\n"
               "|---|---|---|---|---|---|")
    for x in r["period_sweep"]:
        out.append(f"| {int(x['period_s'])} | {_p(x['pdr_mean'], 2)} | {_p(x['pdr_worst_gateway'], 2)} | "
                   f"{_p(x['duty_headroom_min'])} | {x['battery_years_median']} | {x['battery_years_p10']} |")
    rt = r["runtime_s"]
    out.append(f"\n## Runtime\n\nTerrain diffraction for all {s['candidates']} x {s['buildings']} links: "
               f"{rt['terrain_diffraction_s']} s; greedy curve (60 steps): {rt['greedy_curve_60_s']} s; "
               f"swap search: {rt['swap_search_s']} s; whole demo: {rt['total_s']} s.\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")
