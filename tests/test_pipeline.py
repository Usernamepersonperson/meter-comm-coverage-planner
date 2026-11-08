import json

import numpy as np
import pytest

from meter_coverage import pipeline
from meter_coverage.cli import main
from meter_coverage.links import lorawan_eu868
from meter_coverage.pipeline import DATA_DIR, Scenario, plan_sites


@pytest.fixture(scope="module")
def scn():
    s = Scenario.load(DATA_DIR)
    return s


def test_fixture_is_real_town(scn):
    assert 1500 < len(scn.b) < 4000
    assert scn.b.meters.sum() > 3000
    assert scn.dem.z.min() > 400 and scn.dem.z.max() < 700  # Lake Murten at ~429 m
    x0, y0, x1, y1 = scn.dem.extent
    assert scn.b.x.between(x0, x1).all() and scn.b.y.between(y0, y1).all()


def test_no_personal_columns_in_fixture():
    import pandas as pd
    cols = set(pd.read_csv(DATA_DIR / "murten_buildings.csv").columns)
    assert not cols & {"strname_deinr", "strname", "dplzname", "label", "egrid"}


def test_plan_without_terrain_reaches_target(scn):
    links = scn.links(lorawan_eu868(), terrain="none")
    w = scn.b.meters.to_numpy(float)
    p = plan_sites(links, w, 0.95, k_max=30, polish=True)
    assert p["k1"] is not None and p["k1"] <= 15
    cov = [r["cover1"] for r in p["curve"]]
    assert cov == sorted(cov)
    assert p["curve"][p["k1"] - 1]["cover1"] >= 0.95
    assert len(set(p["sites_k1"])) == p["k1"]


def test_more_cellar_loss_needs_more_gateways(scn):
    from meter_coverage.buildings import Penetration, with_penetration
    w = scn.b.meters.to_numpy(float)
    t = lorawan_eu868()
    low = plan_sites(scn.links(t, terrain="none", b=with_penetration(scn.b, Penetration(cellar_db=20.0))),
                     w, 0.95, 40, polish=False)
    high = plan_sites(scn.links(t, terrain="none", b=with_penetration(scn.b, Penetration(cellar_db=40.0))),
                      w, 0.95, 40, polish=False)
    assert (low["k1"] or 99) <= (high["k1"] or 99)
    assert high["curve"][4]["cover1"] <= low["curve"][4]["cover1"]


def test_terrain_requires_compute(scn):
    s = Scenario(scn.dem, scn.b, scn.cands)
    with pytest.raises(RuntimeError):
        s.links(lorawan_eu868(), terrain="deygout")


def test_cli_plan_writes_outputs(tmp_path):
    rc = main(["plan", "--tech", "lorawan", "--terrain", "none", "--target", "0.9", "--out", str(tmp_path)])
    assert rc == 0
    gj = json.loads((tmp_path / "coverage.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"meter_location", "gateway"}
    lon, lat = gj["features"][0]["geometry"]["coordinates"]
    assert 7.0 < lon < 7.3 and 46.8 < lat < 47.0  # Murten
    assert (tmp_path / "gateways.csv").exists() and (tmp_path / "coverage_map.png").stat().st_size > 10_000
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["evaluation"]["cover1"] >= 0.9


def test_cli_rejects_bad_target(tmp_path):
    with pytest.raises(SystemExit):
        main(["plan", "--target", "1.5", "--out", str(tmp_path)])


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])


def test_constants_sane():
    assert 0 < pipeline.TARGET <= 1 and pipeline.PERIOD_S > 0
    assert np.isfinite(pipeline.FADE_MARGIN_DB)
