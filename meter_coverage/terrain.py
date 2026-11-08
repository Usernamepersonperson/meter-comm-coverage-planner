"""Digital elevation model on a regular LV95 grid with bilinear sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Dem:
    """Regular grid. Row 0 is the southern row; z[r, c] is the height at the cell centre."""

    def __init__(self, z: np.ndarray, x0: float, y0: float, cell: float):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2 or min(z.shape) < 2:
            raise ValueError("DEM must be a 2-D array of at least 2x2 cells")
        if cell <= 0:
            raise ValueError("cell size must be positive")
        if not np.isfinite(z).all():
            raise ValueError("DEM contains NaN/inf cells")
        self.z, self.x0, self.y0, self.cell = z, float(x0), float(y0), float(cell)

    @classmethod
    def load(cls, path: str | Path) -> Dem:
        d = np.load(path)
        return cls(d["z"], float(d["x0"]), float(d["y0"]), float(d["cell"]))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        ny, nx = self.z.shape
        return self.x0, self.y0, self.x0 + nx * self.cell, self.y0 + ny * self.cell

    def elevation(self, x, y) -> np.ndarray:
        """Bilinear height; coordinates outside the grid are clamped to the border."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ny, nx = self.z.shape
        c = np.clip((x - self.x0) / self.cell - 0.5, 0.0, nx - 1.000001)
        r = np.clip((y - self.y0) / self.cell - 0.5, 0.0, ny - 1.000001)
        c0 = c.astype(np.int64)
        r0 = r.astype(np.int64)
        fc, fr = c - c0, r - r0
        z = self.z
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bot = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr

    def hillshade(self, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
        gy, gx = np.gradient(self.z.astype(np.float64), self.cell)
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(gy, -gx)
        az, alt = np.radians(360.0 - azimuth_deg + 90.0), np.radians(altitude_deg)
        hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
        return np.clip(hs, 0.0, 1.0)
