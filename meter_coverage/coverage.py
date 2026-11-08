"""Candidate gateway sites, path-loss matrices and per-meter best server / spreading factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .links import Tech
from .propagation import model_loss_db, terrain_diffraction
from .terrain import Dem

METER_HEIGHT_M = 1.5  # Hata mobile height; entry loss is a separate term


def make_candidates(b: pd.DataFrame, dem: Dem, roof_cell_m: float = 150.0, pole_grid_m: float = 250.0,
                    pole_h_m: float = 10.0, mast_m: float = 2.0) -> pd.DataFrame:
    """Candidate sites: the tallest building per grid cell (mast on the roof) + a regular pole grid.

    Columns: cid, kind, x, y, ground_z, h_ant (antenna height above ground, m).
    """
    if b.empty:
        raise ValueError("no buildings")
    cx = np.floor((b["x"] - b["x"].min()) / roof_cell_m).astype(int)
    cy = np.floor((b["y"] - b["y"].min()) / roof_cell_m).astype(int)
    tmp = b.assign(cx=cx, cy=cy).sort_values(["height_m", "egid"], ascending=[False, True])
    roofs = tmp.drop_duplicates(["cx", "cy"])
    rows = [{"kind": "roof", "x": r.x, "y": r.y, "ground_z": r.ground_z, "h_ant": r.height_m + mast_m}
            for r in roofs.itertuples()]
    x0, y0, x1, y1 = b["x"].min(), b["y"].min(), b["x"].max(), b["y"].max()
    for x in np.arange(x0 + pole_grid_m / 2, x1, pole_grid_m):
        for y in np.arange(y0 + pole_grid_m / 2, y1, pole_grid_m):
            rows.append({"kind": "pole", "x": float(x), "y": float(y),
                         "ground_z": float(dem.elevation(x, y)), "h_ant": pole_h_m})
    c = pd.DataFrame(rows)
    c.insert(0, "cid", np.arange(len(c)))
    return c


def diffraction_matrices(dem: Dem, cands: pd.DataFrame, b: pd.DataFrame, f_mhz: float, n_profile: int = 100):
    """Deygout and Epstein-Peterson terrain diffraction loss, float32 arrays (n_cand, n_building)."""
    rx_xy = b[["x", "y"]].to_numpy()
    rx_h = b["ground_z"].to_numpy() + METER_HEIGHT_M
    dey = np.zeros((len(cands), len(b)), dtype=np.float32)
    ep = np.zeros_like(dey)
    for i, c in enumerate(cands.itertuples()):
        d, e = terrain_diffraction(dem, (c.x, c.y), c.ground_z + c.h_ant, rx_xy, rx_h, f_mhz, n_profile)
        dey[i], ep[i] = d, e
    return dey, ep


def distances_km(cands: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    dx = cands["x"].to_numpy()[:, None] - b["x"].to_numpy()[None, :]
    dy = cands["y"].to_numpy()[:, None] - b["y"].to_numpy()[None, :]
    return np.maximum(np.hypot(dx, dy), 20.0) / 1000.0


def path_loss(cands: pd.DataFrame, b: pd.DataFrame, f_mhz: float, model: str = "hata", env: str = "suburban",
              diff: np.ndarray | None = None) -> np.ndarray:
    """Path loss (cand x building) without entry loss: median model + optional terrain diffraction."""
    d = distances_km(cands, b)
    hb = np.maximum(cands["h_ant"].to_numpy(), 3.0)
    out = np.empty(d.shape, dtype=np.float32)
    for i in range(len(cands)):  # hb differs per candidate
        out[i] = model_loss_db(model, d[i], f_mhz, float(hb[i]), METER_HEIGHT_M, env)
    if diff is not None:
        out += diff
    return out


@dataclass
class LinkTable:
    """Per cand x building: required link margin per level and feasibility."""

    tech: Tech
    loss: np.ndarray  # path loss + entry loss + (not fade margin), float32 (C, B)
    fade_margin_db: float

    def __post_init__(self):
        mapl = self.tech.mapl_vector(self.fade_margin_db)  # (L,)
        # level index = first (fastest) level whose budget holds; -1 if none
        ok = self.loss[:, :, None] <= mapl[None, None, :]
        any_ok = ok.any(axis=2)
        self.level = np.where(any_ok, ok.argmax(axis=2), -1).astype(np.int8)
        self.feasible = any_ok  # (C, B) bool: the gateway can hear that building at its slowest level

    def selected(self, sel: np.ndarray):
        """Best server and level for a set of gateway indices. Returns (server_row, level, redundancy)."""
        sel = np.asarray(sel, dtype=int)
        B = self.loss.shape[1]
        if len(sel) == 0:
            return np.full(B, -1), np.full(B, -1, dtype=np.int8), np.zeros(B, dtype=int)
        sub = self.loss[sel]
        best = sub.argmin(axis=0)
        lv = self.level[sel][best, np.arange(B)]
        red = self.feasible[sel].sum(axis=0)
        server = np.where(lv >= 0, sel[best], -1)
        return server, lv, red


def build_links(tech: Tech, cands: pd.DataFrame, b: pd.DataFrame, dem: Dem | None, *, model: str = "hata",
                env: str = "suburban", fade_margin_db: float = 8.0, diff: np.ndarray | None = None) -> LinkTable:
    loss = path_loss(cands, b, tech.freq_mhz, model, env, diff) + b["pen_db"].to_numpy(dtype=np.float32)[None, :]
    return LinkTable(tech, loss.astype(np.float32), fade_margin_db)
