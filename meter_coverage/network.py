"""Evaluate a gateway set: spreading-factor mix, airtime / duty cycle, collision load, battery life."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .coverage import LinkTable
from .links import aloha_success


def evaluate(links: LinkTable, b: pd.DataFrame, sel, period_s: float = 900.0) -> dict:
    """Plan metrics for gateway rows `sel` of the link table. Meter-weighted unless noted."""
    tech = links.tech
    w = b["meters"].to_numpy(dtype=float)
    server, lv, red = links.selected(sel)
    covered = lv >= 0
    n_lv = len(tech.levels)
    names = [lv_.name for lv_ in tech.levels]

    mix = {names[i]: float(w[lv == i].sum() / w.sum()) for i in range(n_lv)}
    mix["uncovered"] = float(w[~covered].sum() / w.sum())

    airtime = np.array([lv_.airtime_s for lv_ in tech.levels])
    years_by_level = np.array([tech.battery_years(lv_, period_s) for lv_ in tech.levels])
    duty = np.array([tech.duty_used(lv_, period_s) for lv_ in tech.levels])
    out = {
        "tech": tech.name,
        "gateways": int(len(sel)),
        "cover1": float(w[red >= 1].sum() / w.sum()),
        "cover2": float(w[red >= 2].sum() / w.sum()),
        "buildings_cover1": float((red >= 1).mean()),
        "level_mix": mix,
        "airtime_s": {names[i]: float(airtime[i]) for i in range(n_lv)},
    }
    if covered.any():
        wc = w[covered]
        out["mean_airtime_s"] = float((airtime[lv[covered]] * wc).sum() / wc.sum())
        out["duty_used_max"] = float(duty[lv[covered]].max())
        out["duty_headroom_min"] = float(1.0 - duty[lv[covered]].max())
        ys = years_by_level[lv[covered]]
        order = np.argsort(ys)
        cw = np.cumsum(wc[order]) / wc.sum()
        out["battery_years_median"] = float(ys[order][np.searchsorted(cw, 0.5)])
        out["battery_years_p10"] = float(ys[order][np.searchsorted(cw, 0.1)])
        out["battery_years_min"] = float(ys.min())
    # collision load per gateway: pure ALOHA per level and channel
    per_gw = []
    for g in sorted(set(server[covered].tolist())):
        mine = covered & (server == g)
        worst, mean_pdr, nm = 1.0, 0.0, float(w[mine].sum())
        for i in range(n_lv):
            n_i = float(w[mine & (lv == i)].sum())
            if n_i == 0:
                continue
            g_load = n_i * airtime[i] / period_s / tech.channels
            pdr = float(aloha_success(g_load))
            worst = min(worst, pdr)
            mean_pdr += pdr * n_i
        per_gw.append({"row": int(g), "meters": nm, "pdr_worst_level": worst,
                       "pdr_mean": mean_pdr / nm if nm else 1.0})
    out["gateway_load"] = per_gw
    if per_gw:
        tot = sum(r["meters"] for r in per_gw)
        out["pdr_mean"] = float(sum(r["pdr_mean"] * r["meters"] for r in per_gw) / tot)
        out["pdr_worst_gateway"] = float(min(r["pdr_worst_level"] for r in per_gw))
        out["meters_per_gateway_max"] = float(max(r["meters"] for r in per_gw))
    return out
