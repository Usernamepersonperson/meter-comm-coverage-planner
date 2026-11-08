import numpy as np
import pandas as pd
import pytest

from meter_coverage import links as LK
from meter_coverage import optimize as O
from meter_coverage.buildings import Penetration, load_buildings, with_penetration
from meter_coverage.coverage import LinkTable, make_candidates
from meter_coverage.network import evaluate
from meter_coverage.terrain import Dem


def test_lora_airtime_reference_values():
    # widely published values for a 51-byte PHY payload, 125 kHz, CR 4/5, 8-symbol preamble
    assert LK.lora_airtime_s(7, 51) == pytest.approx(0.1027, abs=0.0005)
    assert LK.lora_airtime_s(12, 51) == pytest.approx(2.4658, abs=0.002)


def test_airtime_grows_with_sf_and_payload():
    a = [LK.lora_airtime_s(sf, 25) for sf in range(7, 13)]
    assert all(x < y for x, y in zip(a, a[1:], strict=False))
    assert LK.lora_airtime_s(9, 50) > LK.lora_airtime_s(9, 10)


def test_airtime_rejects_bad_input():
    with pytest.raises(ValueError):
        LK.lora_airtime_s(5, 10)
    with pytest.raises(ValueError):
        LK.lora_airtime_s(7, -1)


def test_lorawan_levels_and_budget():
    t = LK.lorawan_eu868()
    assert [lv.name for lv in t.levels] == [f"SF{i}" for i in range(7, 13)]
    sens = [lv.sens_dbm for lv in t.levels]
    assert sens == sorted(sens, reverse=True)
    # 14 - 2 + 5 - 1 + 137 - 8 = 145 dB
    assert t.mapl_db(t.levels[-1], 8.0) == pytest.approx(145.0)


def test_wmbus_budget_much_smaller_than_lora_sf12():
    lo, wm = LK.lorawan_eu868(), LK.wmbus_t1()
    assert wm.mapl_db(wm.levels[0], 8.0) < lo.mapl_db(lo.levels[-1], 8.0) - 25
    assert wm.levels[0].airtime_s < 0.02  # a T1 frame is milliseconds


def test_wmbus_sits_in_the_tenth_percent_subband():
    assert LK.wmbus_t1().duty_cycle == 0.001 and LK.lorawan_eu868().duty_cycle == 0.01


def test_c1_less_sensitive_than_t1():
    assert LK.wmbus_c1().levels[0].sens_dbm > LK.wmbus_t1().levels[0].sens_dbm


def test_duty_cycle_and_battery_monotonic():
    t = LK.lorawan_eu868()
    sf7, sf12 = t.levels[0], t.levels[-1]
    assert t.duty_used(sf12, 900) > t.duty_used(sf7, 900)
    assert t.duty_used(sf12, 900) < 1.0  # 15-minute reporting fits the 1% limit
    assert t.battery_years(sf7, 900) > t.battery_years(sf12, 900)
    assert t.battery_years(sf7, 3600) > t.battery_years(sf7, 900)


def test_aloha_success():
    assert float(LK.aloha_success(0.0)) == 1.0
    assert float(LK.aloha_success(0.5)) == pytest.approx(np.exp(-1.0))


def test_sensitivity_offset_shifts_budget():
    t = LK.lorawan_eu868()
    t2 = LK.with_sensitivity_offset(t, -3.0)
    assert t2.mapl_db(t2.levels[0], 8.0) == pytest.approx(t.mapl_db(t.levels[0], 8.0) + 3.0)


# ------------------------------------------------------------------ optimiser

def _toy():
    # 4 candidates, 6 buildings. weights 1 except building 5 (weight 4)
    A = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0],
    ], dtype=bool)
    w = np.array([1, 1, 1, 1, 1, 4], dtype=float)
    return A, w


def test_objective_and_stats():
    A, w = _toy()
    assert O.objective(A, w, [0], lam=0.5) == pytest.approx(3.0)
    st = O.coverage_stats(A, w, [0, 1])
    assert st["cover1"] == pytest.approx(5 / 9)
    assert st["cover2"] == pytest.approx(1 / 9)
    assert O.coverage_stats(A, w, []) == {"cover1": 0.0, "cover2": 0.0}


def test_greedy_picks_heaviest_first_and_is_monotone():
    A, w = _toy()
    order = O.greedy(A, w, 4)
    assert order[0] == 2  # covers the weight-4 building
    cv = O.curve(A, w, order)
    assert all(b["cover1"] >= a["cover1"] for a, b in zip(cv, cv[1:], strict=False))
    assert cv[-1]["cover1"] == pytest.approx(1.0)


def test_greedy_stops_when_nothing_gained():
    A = np.zeros((3, 4), dtype=bool)
    assert O.greedy(A, np.ones(4), 3) == []


def test_local_search_never_worse():
    rng = np.random.default_rng(0)
    A = rng.random((30, 200)) < 0.12
    w = rng.integers(1, 5, 200).astype(float)
    g = O.greedy(A, w, 6)
    ls = O.local_search(A, w, g)
    assert O.objective(A, w, ls) >= O.objective(A, w, g) - 1e-9
    assert len(set(ls)) == len(ls) == len(g)


def test_ilp_matches_brute_force_on_small_instance():
    from itertools import combinations
    rng = np.random.default_rng(1)
    A = rng.random((9, 40)) < 0.25
    w = rng.integers(1, 4, 40).astype(float)
    best = max(O.objective(A, w, c) for c in combinations(range(9), 3))
    sites, info = O.ilp(A, w, 3, time_limit_s=20)
    assert len(sites) == 3
    assert O.objective(A, w, sites) == pytest.approx(best)
    assert info["rows"] <= 40


def test_ilp_degenerate():
    sites, info = O.ilp(np.zeros((3, 5), dtype=bool), np.ones(5), 2)
    assert sites == [] and info["status"] == "empty"


def test_min_sites():
    rows = [{"k": 1, "cover1": 0.5}, {"k": 2, "cover1": 0.96}]
    assert O.min_sites(rows, 0.95) == 2
    assert O.min_sites(rows, 0.99) is None


# ------------------------------------------------------------------ buildings / network

def _gwr(tmp_path, rows):
    cols = ["egid", "gkode", "gkodn", "gstat", "garea", "gastw", "ganzwhg", "gklas", "gkat"]
    p = tmp_path / "b.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    return p


@pytest.fixture
def flat_dem():
    return Dem(np.full((100, 100), 400.0), 2_600_000.0, 1_200_000.0, 10.0)


def test_load_buildings_rules(tmp_path, flat_dem):
    x, y = 2_600_500.0, 1_200_500.0
    p = _gwr(tmp_path, [
        [1, x, y, 1004, 120, 3, 4, 1110, 1020],          # 4 dwellings -> 4 meters
        [2, x + 10, y, 1004, 80, np.nan, np.nan, 1230, 1060],  # shop -> 1 meter
        [3, x + 20, y, 1004, 30, 1, np.nan, 1230, 1060],  # too small -> none
        [4, x + 30, y, 1004, 90, 1, np.nan, 1274, 1060],  # shed class -> none
        [5, x + 40, y, 1002, 90, 2, 2, 1110, 1020],       # planned only -> none
    ])
    b = load_buildings(p, flat_dem)
    assert list(b.egid) == [1, 2]
    assert list(b.meters) == [4, 1]
    assert b.height_m.iloc[0] == 3 * 3 + 2
    assert b.ground_z.iloc[0] == pytest.approx(400.0)


def test_load_buildings_missing_columns(tmp_path, flat_dem):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"egid": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_buildings(p, flat_dem)


def test_penetration_share_and_determinism(tmp_path, flat_dem):
    rows = [[i, 2_600_100.0 + i, 1_200_100.0, 1004, 100, 2, 1, 1110, 1020] for i in range(400)]
    p = _gwr(tmp_path, rows)
    b1 = load_buildings(p, flat_dem, Penetration(cellar_share=0.3, seed=4))
    b2 = load_buildings(p, flat_dem, Penetration(cellar_share=0.3, seed=4))
    assert b1.cellar.equals(b2.cellar)
    assert 0.2 < b1.cellar.mean() < 0.4
    assert set(b1.pen_db) == {15.0, 30.0}
    b3 = with_penetration(b1, Penetration(cellar_db=40.0, cellar_share=0.3, seed=4))
    assert b3.pen_db.max() == 40.0 and b3.cellar.equals(b1.cellar)


def _toy_world():
    dem = Dem(np.full((100, 100), 400.0), 0.0, 0.0, 10.0)
    xs = np.array([100.0, 200.0, 800.0, 900.0])
    b = pd.DataFrame({"egid": range(4), "x": xs, "y": [100.0, 500.0, 300.0, 700.0], "floors": 2.0,
                      "meters": [1, 2, 1, 3],
                      "height_m": 8.0, "ground_z": 400.0, "cellar": False, "pen_db": 15.0})
    return dem, b


def test_candidates_cover_roofs_and_poles():
    dem, b = _toy_world()
    c = make_candidates(b, dem, roof_cell_m=150, pole_grid_m=400)
    assert set(c.kind) == {"roof", "pole"}
    assert (c.cid == np.arange(len(c))).all()
    assert (c[c.kind == "roof"].h_ant == 8.0 + 2.0).all()


def test_link_table_best_server_and_levels():
    tech = LK.lorawan_eu868()
    # budgets with 8 dB margin: SF7 131 ... SF10 140, SF12 145 dB
    loss = np.array([[100.0, 138.0, 160.0], [150.0, 141.0, 150.0]], dtype=np.float32)
    lt = LinkTable(tech, loss, 8.0)
    server, lv, red = lt.selected([0, 1])
    assert list(server) == [0, 0, -1]
    assert lv[0] == 0 and lv[1] == 3 and lv[2] == -1  # SF7, SF10, none
    assert list(red) == [1, 2, 0]
    s2, l2, r2 = lt.selected([])
    assert (s2 == -1).all() and (l2 == -1).all() and (r2 == 0).all()


def test_evaluate_reports_mix_and_battery():
    tech = LK.lorawan_eu868()
    _, b = _toy_world()
    loss = np.array([[100.0, 130.0, 160.0, 140.0]], dtype=np.float32)
    lt = LinkTable(tech, loss, 8.0)
    ev = evaluate(lt, b, [0])
    w = b.meters.to_numpy()
    assert ev["cover1"] == pytest.approx((1 + 2 + 3) / w.sum())
    assert sum(ev["level_mix"].values()) == pytest.approx(1.0)
    assert ev["level_mix"]["uncovered"] == pytest.approx(1 / w.sum())
    assert ev["battery_years_min"] <= ev["battery_years_median"]
    assert 0 < ev["pdr_mean"] <= 1.0
    assert ev["duty_headroom_min"] > 0
