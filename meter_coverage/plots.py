"""Matplotlib figures: coverage map, coverage-vs-gateways curve, README and LinkedIn images."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
SOURCE = "Source: swisstopo (swissALTI3D), FSO (GWR). Modelled coverage, no field measurements."


def _font():
    from matplotlib import font_manager  # noqa: PLC0415

    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _level_groups(links):
    n = len(links.tech.levels)
    if n == 6:  # SF7..SF12
        return [("SF7-8", [0, 1], GREEN), ("SF9-10", [2, 3], ACCENT), ("SF11-12", [4, 5], AMBER)]
    return [(links.tech.levels[0].name, list(range(n)), GREEN)]


def draw_map(ax, scn, links, sites, *, ms=5.0, star=230, legend=True, fs=11):
    from .export import per_building  # noqa: PLC0415

    x0, y0, x1, y1 = scn.dem.extent
    hs = scn.dem.hillshade()
    cmap = LinearSegmentedColormap.from_list("dark", [BG, "#33425F"])
    ax.imshow(hs, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, vmin=0.2, vmax=1.0, zorder=0)
    pb = per_building(links, scn.b, sites)
    names = [lv.name for lv in links.tech.levels]
    lvl = pb["level"].map({n: i for i, n in enumerate(names)}).fillna(-1).astype(int).to_numpy()
    unc = lvl < 0
    ax.scatter(pb.x[unc], pb.y[unc], s=ms, c=RED, lw=0, zorder=2)
    for _, idx, col in _level_groups(links):
        m = np.isin(lvl, idx)
        ax.scatter(pb.x[m], pb.y[m], s=ms, c=col, lw=0, zorder=2)
    sel = scn.cands.loc[list(sites)]
    ax.scatter(sel.x, sel.y, s=star, marker="*", c=TEXT, edgecolors=VIOLET, linewidths=1.3, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    sx, sy = x0 + 120, y0 + 110  # 500 m scale bar
    ax.plot([sx, sx + 500], [sy, sy], color=TEXT, lw=3, zorder=6)
    ax.text(sx + 250, sy + 40, "500 m", color=TEXT, ha="center", va="bottom", fontsize=fs - 1, zorder=6)
    if legend:
        h = [Line2D([], [], marker="*", ls="", mfc=TEXT, mec=VIOLET, ms=13, label="gateway")]
        h += [Line2D([], [], marker="o", ls="", color=c, ms=7, label=n) for n, _, c in _level_groups(links)]
        h.append(Line2D([], [], marker="o", ls="", color=RED, ms=7, label="not covered"))
        leg = ax.legend(handles=h, loc="upper left", fontsize=fs - 1, facecolor=BG, edgecolor=GRID,
                        labelcolor=TEXT, framealpha=0.9, borderpad=0.6)
        leg.set_zorder(10)


def draw_curve(ax, res, *, tech="lorawan", fs=12, others=False, target=0.95):
    _style(ax)
    cur = res["technologies"][tech]["curve"]
    k = [r["k"] for r in cur]
    ax.plot(k, [100 * r["cover1"] for r in cur], color=ACCENT, lw=3, label="heard by 1+ gateway")
    ax.plot(k, [100 * r["cover2"] for r in cur], color=GREEN, lw=3, label="heard by 2+ gateways")
    if others:
        for name, col in (("wmbus_t1", AMBER), ("mioty", VIOLET)):
            c2 = res["technologies"][name]["curve"]
            ax.plot([r["k"] for r in c2], [100 * r["cover1"] for r in c2], color=col, lw=1.8, ls="--",
                    label={"wmbus_t1": "wM-Bus T1, 1+", "mioty": "mioty (params), 1+"}[name])
    ax.axhline(100 * target, color=AMBER, lw=1.2, ls=":")
    k1 = res["technologies"][tech]["sites_95"]
    if k1:
        ax.scatter([k1], [100 * target], s=90, color=AMBER, zorder=5)
        ax.annotate(f"{k1} gateways", (k1, 100 * target), xytext=(k1 + 2.2, 100 * target - 14), color=AMBER,
                    fontsize=fs, fontweight="bold", arrowprops=dict(arrowstyle="-", color=AMBER, lw=1))
    ax.set_xlim(0, 25 if not others else 40)
    ax.set_ylim(0, 102)
    ax.set_xlabel("gateways", color=TEXT, fontsize=fs)
    ax.set_ylabel("% of meters covered", color=TEXT, fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.legend(loc="lower right", fontsize=fs - 1, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)


def coverage_map_png(path: Path, scn, links, sites) -> None:
    _font()
    fig = plt.figure(figsize=(12, 9.4), dpi=100, facecolor=BG)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.88))
    draw_map(ax, scn, links, sites, ms=7, fs=12)
    fig.text(0.02, 0.955, f"Meter coverage, {links.tech.name}, {len(sites)} gateways", color=TEXT, fontsize=16,
             fontweight="bold", va="center")
    fig.text(0.02, 0.925, SOURCE, color=MUTED, fontsize=10.5, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _mix_bars(ax, res, fs=11):
    _style(ax)
    ax.grid(axis="y", visible=False)
    keys = [k for k in res["plans"] if k.startswith(("optimised", "tallest", "demand"))]
    labels = []
    for k in keys:
        parts = k.split("_")
        n = parts[1] if k.endswith("redundant") else parts[-1]
        kind = ("redundant" if k.endswith("redundant") else "optimised" if k.startswith("optimised")
                else "demand centroids" if k.startswith("demand") else "tallest roofs")
        labels.append(f"{kind}" + chr(10) + f"{n} gw")
    cols = [("SF7", GREEN), ("SF8", GREEN), ("SF9", ACCENT), ("SF10", ACCENT), ("SF11", AMBER), ("SF12", AMBER),
            ("uncovered", RED)]
    left = np.zeros(len(keys))
    shades = {"SF7": 1.0, "SF8": 0.6, "SF9": 1.0, "SF10": 0.6, "SF11": 1.0, "SF12": 0.6, "uncovered": 1.0}
    for name, c in cols:
        v = np.array([100 * res["plans"][k]["level_mix"].get(name, 0.0) for k in keys])
        ax.barh(range(len(keys)), v, left=left, color=c, alpha=shades[name], edgecolor=BG, lw=1.2, label=name)
        left += v
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, color=TEXT, fontsize=fs)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of meters by spreading factor", color=TEXT, fontsize=fs)
    ax.tick_params(labelsize=fs - 1)


def figure_main(path: Path, scn, links, sites, res) -> None:
    _font()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    fig.text(0.02, 0.955, "Smart-meter coverage planner: gateways needed for a Swiss town (LoRaWAN 868 MHz)",
             color=TEXT, fontsize=20, fontweight="bold", va="center")
    s = res["setup"]
    fig.text(0.02, 0.918, f"Murten FR: {s['buildings']} buildings, {s['meters']} assumed meters, "
             f"{s['candidates']} candidate sites. {SOURCE}", color=MUTED, fontsize=12, va="center")
    ax = fig.add_axes((0.01, 0.03, 0.55, 0.84))
    draw_map(ax, scn, links, sites, ms=7)
    ax2 = fig.add_axes((0.62, 0.53, 0.36, 0.33))
    draw_curve(ax2, res, others=True)
    ax2.set_title("Meters covered vs number of gateways", color=TEXT, fontsize=12, loc="left")
    ax3 = fig.add_axes((0.72, 0.17, 0.26, 0.24))
    _mix_bars(ax3, res)
    ax3.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.30), fontsize=10, facecolor=BG,
               edgecolor=GRID, labelcolor=TEXT)
    fig.text(0.62, 0.455, "Spreading-factor mix per plan (LoRaWAN, 15-min reports)", color=TEXT, fontsize=12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def figure_post(path: Path, scn, links, sites, res) -> None:
    _font()
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    c1 = res["plans"][next(k for k in res["plans"] if k.startswith("optimised") and not k.endswith("redundant"))]
    fig.text(0.03, 0.947, f"{len(sites)} gateways reach {100 * c1['cover1']:.0f}% of a Swiss town's smart meters",
             color=TEXT, fontsize=14.5, fontweight="bold", va="center")
    fig.text(0.03, 0.905, "Murten FR, open data (swisstopo, FSO). Modelled LoRaWAN coverage, not measured.",
             color=MUTED, fontsize=11, va="center")
    ax = fig.add_axes((0.015, 0.04, 0.57, 0.84))
    draw_map(ax, scn, links, sites, ms=5, star=170, fs=10.5)
    ax2 = fig.add_axes((0.67, 0.50, 0.31, 0.37))
    draw_curve(ax2, res, fs=11)
    ax2.set_title("meters covered vs gateways", color=TEXT, fontsize=11.5, loc="left")
    slow = c1["level_mix"].get("SF11", 0) + c1["level_mix"].get("SF12", 0)
    k2 = res["technologies"]["lorawan"]["sites_95_2fold"]
    rows = [(f"{100 * c1['cover1']:.1f}%", f"meters heard with {len(sites)} gateways", ACCENT),
            (f"{100 * c1['cover2']:.0f}%", "heard by 2+ gateways", GREEN),
            (f"{k2}", "gateways for 95% with 2+ each", AMBER),
            (f"{100 * slow:.0f}%", "of meters need SF11-12", VIOLET)]
    for i, (big, cap, col) in enumerate(rows):
        y = 0.385 - i * 0.092
        fig.text(0.67, y, big, color=col, fontsize=21, fontweight="bold", va="center")
        fig.text(0.67 + 0.105, y, cap, color=TEXT, fontsize=10.5, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
