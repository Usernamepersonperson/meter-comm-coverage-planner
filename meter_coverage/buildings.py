"""Building register (GWR) points -> metering points with height, ground level and entry loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .terrain import Dem

# GWR building class codes for sheds, garages and farm outbuildings: no metering point assumed
NO_METER_CLASSES = {1274, 1278}
EXISTING = 1004  # GSTAT: building exists


@dataclass(frozen=True)
class Penetration:
    """Extra loss for the meter location, dB (assumed, see README for typical literature values)."""

    indoor_db: float = 15.0  # meter above ground, inside the building shell
    cellar_db: float = 30.0  # meter in a cellar / meter room below ground (total, not additional)
    cellar_share: float = 0.5  # share of buildings whose meter sits in a cellar
    seed: int = 1


def load_buildings(path: str | Path, dem: Dem, pen: Penetration | None = None) -> pd.DataFrame:
    """Return one row per building that has at least one assumed metering point.

    Columns: egid, x, y, floors, height_m, meters, ground_z, cellar, pen_db.
    Meters: number of dwellings if known, else 1 for non-residential buildings with a footprint
    of >= 40 m2 that are not outbuildings. All meters of a building share one location.
    """
    pen = pen or Penetration()
    df = pd.read_csv(path)
    need = {"egid", "gkode", "gkodn", "gstat", "garea", "gastw", "ganzwhg", "gklas"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"building file lacks columns: {sorted(missing)}")
    df = df[df["gstat"] == EXISTING].copy()
    dw = df["ganzwhg"].fillna(0).clip(lower=0)
    area = df["garea"].fillna(0)
    other = (dw == 0) & (area >= 40) & ~df["gklas"].isin(NO_METER_CLASSES)
    df["meters"] = np.where(dw > 0, dw, np.where(other, 1, 0)).astype(int)
    df = df[df["meters"] > 0].copy()
    floors = df["gastw"].copy()
    floors = floors.fillna(pd.Series(np.where(df["garea"].fillna(0) > 300, 3, 2), index=df.index)).clip(lower=1)
    out = pd.DataFrame({
        "egid": df["egid"].astype(int).to_numpy(),
        "x": df["gkode"].to_numpy(dtype=float),
        "y": df["gkodn"].to_numpy(dtype=float),
        "floors": floors.to_numpy(dtype=float),
        "meters": df["meters"].to_numpy(),
    })
    out["height_m"] = out["floors"] * 3.0 + 2.0  # storeys x 3 m + roof
    out["ground_z"] = dem.elevation(out["x"].to_numpy(), out["y"].to_numpy()).astype(float)
    rng = np.random.default_rng(pen.seed)
    out["cellar"] = rng.random(len(out)) < pen.cellar_share
    out["pen_db"] = np.where(out["cellar"], pen.cellar_db, pen.indoor_db)
    return out.reset_index(drop=True)


def with_penetration(b: pd.DataFrame, pen: Penetration) -> pd.DataFrame:
    """Copy of b with a different penetration assumption (same random cellar draw per seed)."""
    out = b.copy()
    rng = np.random.default_rng(pen.seed)
    out["cellar"] = rng.random(len(out)) < pen.cellar_share
    out["pen_db"] = np.where(out["cellar"], pen.cellar_db, pen.indoor_db)
    return out
