import numpy as np
import pytest

from meter_coverage import propagation as P
from meter_coverage.terrain import Dem


def test_fspl_known_value():
    # 1 km at 868 MHz: 32.44 + 20log10(868) + 0 = 91.2 dB
    assert P.fspl_db(1.0, 868.0) == pytest.approx(91.2, abs=0.05)


def test_fspl_doubles_distance_adds_6db():
    assert P.fspl_db(2.0, 868.0) - P.fspl_db(1.0, 868.0) == pytest.approx(6.02, abs=0.01)


def test_hata_urban_textbook_value():
    # f=900 MHz, hb=30 m, hm=1.5 m, d=1 km -> ~126.4 dB (small/medium city)
    assert float(P.hata_db(1.0, 900.0, 30.0, 1.5, "urban")) == pytest.approx(126.4, abs=0.1)


def test_hata_environment_ordering():
    u = P.hata_db(1.0, 868.0, 20.0, 1.5, "urban")
    s = P.hata_db(1.0, 868.0, 20.0, 1.5, "suburban")
    o = P.hata_db(1.0, 868.0, 20.0, 1.5, "open")
    assert u > s > o


def test_hata_suburban_correction_formula():
    u = float(P.hata_db(1.0, 900.0, 30.0, 1.5, "urban"))
    s = float(P.hata_db(1.0, 900.0, 30.0, 1.5, "suburban"))
    assert u - s == pytest.approx(2 * np.log10(900 / 28) ** 2 + 5.4, abs=1e-9)


def test_hata_increases_with_distance_and_lower_base():
    assert P.hata_db(2.0, 868.0, 20.0) > P.hata_db(1.0, 868.0, 20.0)
    assert P.hata_db(1.0, 868.0, 10.0) > P.hata_db(1.0, 868.0, 40.0)


def test_hata_rejects_bad_input():
    with pytest.raises(ValueError):
        P.hata_db(1.0, 868.0, 20.0, 1.5, "forest")
    with pytest.raises(ValueError):
        P.hata_db(1.0, -5.0, 20.0)
    with pytest.raises(ValueError):
        P.hata_db(1.0, 868.0, 0.0)


def test_cost231_textbook_value():
    # f=1800, hb=30, hm=1.5, d=1 km, medium city -> ~136.2 dB
    assert float(P.cost231_db(1.0, 1800.0, 30.0, 1.5)) == pytest.approx(136.2, abs=0.1)
    assert float(P.cost231_db(1.0, 1800.0, 30.0, 1.5, metropolitan=True)) == pytest.approx(139.2, abs=0.1)


def test_model_loss_never_below_free_space():
    d = np.array([0.02, 0.05, 0.1])
    assert (P.model_loss_db("hata", d, 868.0, 40.0, 1.5, "open") >= P.fspl_db(d, 868.0) - 1e-9).all()


def test_model_loss_unknown_model():
    with pytest.raises(ValueError):
        P.model_loss_db("itm", 1.0, 868.0, 20.0, 1.5, "urban")


def test_knife_edge_known_points():
    assert float(P.knife_edge_j(0.0)) == pytest.approx(6.0, abs=0.1)  # grazing: ~6 dB
    assert float(P.knife_edge_j(-1.0)) == 0.0
    assert float(P.knife_edge_j(1.0)) == pytest.approx(13.97, abs=0.1)
    assert float(P.knife_edge_j(2.4)) == pytest.approx(20.5, abs=0.3)


def test_knife_edge_monotonic():
    v = np.linspace(-0.7, 5, 50)
    assert (np.diff(P.knife_edge_j(v)) > 0).all()


def test_fresnel_v_sign_and_scale():
    v = P.fresnel_v(10.0, 500.0, 500.0, 868.0)
    assert v > 0
    assert P.fresnel_v(-10.0, 500.0, 500.0, 868.0) == pytest.approx(-v)


def _ridge_dem(height=30.0):
    """Flat 100 m terrain with a north-south ridge at x=1000 m."""
    z = np.full((100, 200), 100.0, dtype=np.float32)
    z[:, 99] = 100.0 + height
    return Dem(z, 0.0, 0.0, 10.0)


def test_ridge_between_gateway_and_meter_causes_loss():
    dem = _ridge_dem(30.0)
    d, ep = P.terrain_diffraction(dem, (200.0, 500.0), 115.0, [(1800.0, 500.0)], 101.5, 868.0)
    assert d[0] > 10.0 and ep[0] > 10.0
    # a single thin ridge is one edge, so the two methods agree closely
    assert d[0] == pytest.approx(ep[0], abs=3.0)


def test_no_ridge_means_no_loss():
    dem = _ridge_dem(0.0)
    d, ep = P.terrain_diffraction(dem, (200.0, 500.0), 130.0, [(1800.0, 500.0)], 101.5, 868.0)
    assert d[0] == 0.0 and ep[0] == 0.0


def test_higher_gateway_reduces_loss():
    dem = _ridge_dem(30.0)
    low, _ = P.terrain_diffraction(dem, (200.0, 500.0), 115.0, [(1800.0, 500.0)], 101.5, 868.0)
    high, _ = P.terrain_diffraction(dem, (200.0, 500.0), 160.0, [(1800.0, 500.0)], 101.5, 868.0)
    assert high[0] < low[0]


def test_vectorised_deygout_matches_scalar_reference():
    rng = np.random.default_rng(3)
    z = (100 + 25 * rng.random((60, 60))).astype(np.float32)
    z = (z + np.roll(z, 1, 0) + np.roll(z, 1, 1)) / 3  # smooth a little
    dem = Dem(z, 0.0, 0.0, 10.0)
    rx = np.column_stack([rng.uniform(50, 550, 40), rng.uniform(50, 550, 40)])
    tx = (60.0, 60.0)
    s, D, Hf = P.terrain_profile(dem, tx, 120.0, rx, 105.0, n=60)
    vec, _ = P.diffraction_losses(s, D, Hf, 868.0)
    for i in range(len(rx)):
        ref = P.deygout_scalar(s * D[i], Hf[i], 868.0, depth=2)
        assert vec[i] == pytest.approx(ref, abs=1e-6)


def test_multi_edge_losses_finite_and_nonnegative():
    rng = np.random.default_rng(5)
    z = (100 + 40 * rng.random((80, 80))).astype(np.float32)
    dem = Dem(z, 0.0, 0.0, 10.0)
    rx = np.column_stack([rng.uniform(300, 780, 200), rng.uniform(300, 780, 200)])
    d, ep = P.terrain_diffraction(dem, (20.0, 20.0), 110.0, rx, 102.0, 868.0, n=80)
    assert (d >= 0).all() and (ep >= 0).all()
    assert np.isfinite(d).all() and np.isfinite(ep).all()
    assert d.mean() > 0


def test_dem_bilinear_and_clamp():
    z = np.array([[0.0, 10.0], [20.0, 30.0]])
    dem = Dem(z, 0.0, 0.0, 10.0)
    assert float(dem.elevation(10.0, 10.0)) == pytest.approx(15.0)  # centre of the 2x2 grid
    assert float(dem.elevation(-500.0, -500.0)) == pytest.approx(0.0)  # clamped to the border
    assert float(dem.elevation(5.0, 5.0)) == pytest.approx(0.0)


def test_dem_rejects_bad_input():
    with pytest.raises(ValueError):
        Dem(np.zeros((1, 5)), 0, 0, 10)
    with pytest.raises(ValueError):
        Dem(np.full((3, 3), np.nan), 0, 0, 10)
    with pytest.raises(ValueError):
        Dem(np.zeros((3, 3)), 0, 0, 0)
