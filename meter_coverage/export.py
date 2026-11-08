"""GeoJSON (WGS84) and CSV outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import LinkTable
from .network import evaluate


def _to_wgs84(x, y):
    from pyproj import Transformer  # noqa: PLC0415

    tr = Transformer.from_crs(2056, 4326, always_xy=True)
    lon, lat = tr.transform(np.asarray(x), np.asarray(y))
    return np.asarray(lon), np.asarray(lat)


def per_building(links: LinkTable, b: pd.DataFrame, sites) -> pd.DataFrame:
    """Best server, spreading factor / mode, link margin at that level and redundancy per building."""
    server, lv, red = links.selected(sites)
    tech = links.tech
    names = np.array([lv_.name for lv_ in tech.levels] + ["none"])
    mapl_slowest = tech.mapl_db(tech.levels[-1], links.fade_margin_db)
    out = b[["egid", "x", "y", "meters", "cellar"]].copy()
    out["server_cid"] = server
    out["level"] = names[np.where(lv >= 0, lv, len(tech.levels))]
    out["redundancy"] = red
    best_loss = links.loss[np.asarray(sites, dtype=int)].min(axis=0) if len(sites) else np.full(len(b), np.inf)
    out["margin_db"] = np.round(mapl_slowest - best_loss, 1)
    return out


def write_geojson(path: Path, scn, links: LinkTable, sites) -> None:
    pb = per_building(links, scn.b, sites)
    lon, lat = _to_wgs84(pb["x"], pb["y"])
    feats = []
    for r, lo, la in zip(pb.itertuples(), lon, lat, strict=True):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lo, 6), round(la, 6)]},
                      "properties": {"kind": "meter_location", "egid": int(r.egid), "meters": int(r.meters),
                                     "cellar": bool(r.cellar), "best_server": int(r.server_cid),
                                     "level": r.level, "redundancy": int(r.redundancy),
                                     "margin_db_at_slowest": float(r.margin_db)}})
    sel = scn.cands.loc[list(sites)]
    glon, glat = _to_wgs84(sel["x"], sel["y"])
    for r, lo, la in zip(sel.itertuples(), glon, glat, strict=True):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lo, 6), round(la, 6)]},
                      "properties": {"kind": "gateway", "cid": int(r.cid), "site_type": r.kind,
                                     "antenna_height_m": float(r.h_ant), "ground_m": round(float(r.ground_z), 1)}})
    doc = {"type": "FeatureCollection",
           "name": f"coverage_{links.tech.name}",
           "properties": {"source": "building points: Federal Statistical Office FSO (GWR) via geo.admin.ch; "
                                    "terrain: swisstopo swissALTI3D. Modelled, not measured."},
           "features": feats}
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))


def write_gateways_csv(path: Path, scn, links: LinkTable, sites) -> None:
    ev = evaluate(links, scn.b, sites)
    load = {r["row"]: r for r in ev.get("gateway_load", [])}
    sel = scn.cands.loc[list(sites)].copy()
    lon, lat = _to_wgs84(sel["x"], sel["y"])
    sel["lon"], sel["lat"] = np.round(lon, 6), np.round(lat, 6)
    sel["meters_served"] = [load.get(int(c), {}).get("meters", 0) for c in sel["cid"]]
    sel["pdr_worst_level"] = [round(load.get(int(c), {}).get("pdr_worst_level", 1.0), 4) for c in sel["cid"]]
    sel.rename(columns={"kind": "site_type", "x": "x_lv95", "y": "y_lv95"}).to_csv(path, index=False)
