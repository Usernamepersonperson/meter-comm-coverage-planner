"""One-off fetch of the committed fixture for the demo town (needs `pip install rasterio`).

Writes data/murten_buildings.csv (GWR building points) and data/murten_dem_10m.npz (swissALTI3D, 10 m).
Both are open data from geo.admin.ch; see README "Data sources" for the licence and attribution.
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
X0, Y0, X1, Y1 = 2574000, 1195500, 2577000, 1198000  # LV95 box around Murten (FR)
CELL = 10.0
IDENT = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify?"
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items?"
KEEP = ["egid", "gkode", "gkodn", "gkat", "gklas", "gstat", "garea", "gastw", "ganzwhg", "gbauj"]


def _get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _tile(x0, y0, x1, y1):
    q = urllib.parse.urlencode(
        dict(geometryType="esriGeometryEnvelope", geometry=f"{x0},{y0},{x1},{y1}",
             layers="all:ch.bfs.gebaeude_wohnungs_register", tolerance=0, sr=2056,
             returnGeometry="false", limit=500))
    return _get(IDENT + q)["results"]


def fetch_buildings(x0, y0, x1, y1, out):
    res = _tile(x0, y0, x1, y1)
    if len(res) >= 190 and (x1 - x0) > 20:  # server caps near 200 per call: split
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        for a, b, c, d in ((x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)):
            fetch_buildings(a, b, c, d, out)
        return
    for r in res:
        a = r.get("properties") or r["attributes"]
        out[a["egid"]] = {k: a.get(k) for k in KEEP}


def main():
    out: dict = {}
    for x in range(X0, X1, 500):
        for y in range(Y0, Y1, 500):
            fetch_buildings(x, y, x + 500, y + 500, out)
    df = pd.DataFrame(out.values())
    # keep only buildings whose point is inside the box (tile edges overlap)
    df = df[(df.gkode >= X0) & (df.gkode < X1) & (df.gkodn >= Y0) & (df.gkodn < Y1)]
    df = df.sort_values("egid").reset_index(drop=True)
    df.to_csv(ROOT / "data" / "murten_buildings.csv", index=False)
    print("buildings", len(df))

    import rasterio  # noqa: PLC0415
    from rasterio.windows import from_bounds  # noqa: PLC0415

    nx, ny = int((X1 - X0) / CELL), int((Y1 - Y0) / CELL)
    dem = np.full((ny, nx), np.nan, dtype=np.float32)  # row 0 = south
    items = _get(STAC + urllib.parse.urlencode(dict(limit=100, bbox="7.05,46.89,7.17,46.97")))
    with tempfile.TemporaryDirectory() as tmp:
        for tx in range(X0, X1, 1000):
            for ty in range(Y0 - Y0 % 1000, Y1, 1000):
                key = f"{tx // 1000}-{ty // 1000}"
                cand = [f for f in items["features"] if f["id"].endswith(key)]
                if not cand:
                    print("no tile", key)
                    continue
                f = sorted(cand, key=lambda i: i["id"])[-1]
                href = next(a["href"] for n, a in f["assets"].items() if "_2_2056_" in n and n.endswith(".tif"))
                p = Path(tmp) / f"{key}.tif"
                urllib.request.urlretrieve(href, p)
                with rasterio.open(p) as ds:
                    xa, ya = max(tx, X0), max(ty, Y0)
                    xb, yb = min(tx + 1000, X1), min(ty + 1000, Y1)
                    if xb <= xa or yb <= ya:
                        continue
                    w = from_bounds(xa, ya, xb, yb, ds.transform)
                    arr = ds.read(1, window=w, boundless=True, fill_value=np.nan)
                    k = int(CELL / 2)
                    h, wd = (yb - ya) // int(CELL), (xb - xa) // int(CELL)
                    arr = arr[: h * k, : wd * k].reshape(h, k, wd, k).mean(axis=(1, 3))
                    arr = arr[::-1]  # north-up -> south-up
                    r0, c0 = int((ya - Y0) / CELL), int((xa - X0) / CELL)
                    dem[r0:r0 + h, c0:c0 + wd] = arr
                print("tile", key, f["id"])
    print("nan cells", int(np.isnan(dem).sum()), "min/max", float(np.nanmin(dem)), float(np.nanmax(dem)))
    np.savez_compressed(ROOT / "data" / "murten_dem_10m.npz", z=dem, x0=X0, y0=Y0, cell=CELL)


if __name__ == "__main__":
    sys.exit(main())
