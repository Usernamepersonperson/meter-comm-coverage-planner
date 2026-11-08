"""Published path-loss models plus a DEM knife-edge diffraction check.

Models (all in dB, frequency f in MHz, distances d in km, heights in m):

* free-space loss (FSPL)
* Okumura-Hata (Hata 1980): urban, suburban, open. Formally valid 150-1500 MHz, d 1-20 km,
  base height 30-200 m, mobile height 1-10 m. Small cells and rooftop gateways are outside the
  base-height range; the planner extrapolates and never lets the result fall below FSPL.
* COST-231 Hata extension. Formally valid 1500-2000 MHz; at 868 MHz it is an extrapolation and is
  only used as a comparison model.
* Terrain diffraction: ITU-R P.526 single knife-edge J(v), combined along the DEM profile with the
  Deygout (3-edge) and Epstein-Peterson methods. Earth curvature uses k = 4/3.

The DEM is a bare-earth model: buildings and trees are not in it, their average effect is the job
of the Hata clutter class.
"""

from __future__ import annotations

import numpy as np

from .terrain import Dem

K_FACTOR = 4.0 / 3.0
EARTH_R = 6_371_000.0
ENVIRONMENTS = ("urban", "suburban", "open")


def wavelength_m(f_mhz: float) -> float:
    return 299.792458 / f_mhz


def fspl_db(d_km, f_mhz):
    d = np.maximum(np.asarray(d_km, dtype=np.float64), 1e-6)
    return 32.44 + 20.0 * np.log10(f_mhz) + 20.0 * np.log10(d)


def _a_hm(f_mhz: float, hm: float) -> float:
    lf = np.log10(f_mhz)
    return (1.1 * lf - 0.7) * hm - (1.56 * lf - 0.8)


def hata_db(d_km, f_mhz: float, hb_m: float, hm_m: float = 1.5, env: str = "urban"):
    """Okumura-Hata median path loss (small/medium city 'urban' correction)."""
    if env not in ENVIRONMENTS:
        raise ValueError(f"env must be one of {ENVIRONMENTS}")
    if f_mhz <= 0 or hb_m <= 0 or hm_m <= 0:
        raise ValueError("frequency and antenna heights must be positive")
    d = np.maximum(np.asarray(d_km, dtype=np.float64), 1e-3)
    hb = max(hb_m, 3.0)
    lf = np.log10(f_mhz)
    pl = (69.55 + 26.16 * lf - 13.82 * np.log10(hb) - _a_hm(f_mhz, hm_m)
          + (44.9 - 6.55 * np.log10(hb)) * np.log10(d))
    if env == "suburban":
        pl = pl - 2.0 * np.log10(f_mhz / 28.0) ** 2 - 5.4
    elif env == "open":
        pl = pl - 4.78 * lf**2 + 18.33 * lf - 40.94
    return pl


def cost231_db(d_km, f_mhz: float, hb_m: float, hm_m: float = 1.5, metropolitan: bool = False):
    """COST-231 Hata. Formally 1500-2000 MHz; extrapolated when used at 868 MHz."""
    d = np.maximum(np.asarray(d_km, dtype=np.float64), 1e-3)
    hb = max(hb_m, 3.0)
    return (46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(hb) - _a_hm(f_mhz, hm_m)
            + (44.9 - 6.55 * np.log10(hb)) * np.log10(d) + (3.0 if metropolitan else 0.0))


def model_loss_db(model: str, d_km, f_mhz: float, hb_m: float, hm_m: float, env: str):
    """Median loss for a named model, floored at free space."""
    free = fspl_db(d_km, f_mhz)
    if model == "fspl":
        return free
    if model == "hata":
        return np.maximum(hata_db(d_km, f_mhz, hb_m, hm_m, env), free)
    if model == "cost231":
        return np.maximum(cost231_db(d_km, f_mhz, hb_m, hm_m, metropolitan=(env == "urban")), free)
    raise ValueError(f"unknown model {model!r}")


# ---------------------------------------------------------------- knife edge

def knife_edge_j(v):
    """ITU-R P.526 approximation of the single knife-edge loss J(v) in dB (0 for v <= -0.78)."""
    v = np.asarray(v, dtype=np.float64)
    out = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.where(v > -0.78, out, 0.0)


def fresnel_v(h, d1, d2, f_mhz):
    """Fresnel-Kirchhoff parameter for an edge h metres above the line of sight."""
    lam = wavelength_m(f_mhz)
    return h * np.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))


def deygout_scalar(s, H, f_mhz, depth=2):
    """Reference Deygout loss on one profile.

    s: distances (m) from the transmitter, H: effective heights (m, curvature included),
    both including the end points. depth=2 gives main edge plus one edge per side (3 edges).
    """
    s = np.asarray(s, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    def rec(a, b, level):
        if b - a < 2:
            return 0.0
        idx = np.arange(a + 1, b)
        line = H[a] + (s[idx] - s[a]) / (s[b] - s[a]) * (H[b] - H[a])
        v = fresnel_v(H[idx] - line, s[idx] - s[a], s[b] - s[idx], f_mhz)
        k = int(np.argmax(v))
        if v[k] <= -0.78:
            return 0.0
        m = idx[k]
        loss = float(knife_edge_j(v[k]))
        if level > 1:
            loss += rec(a, m, level - 1) + rec(m, b, level - 1)
        return loss

    return rec(0, len(s) - 1, depth)


def _gather(arr, idx):
    return np.take_along_axis(arr, idx[:, None], axis=1)[:, 0]


def _search(Hf, s, D, a, b, f_mhz):
    """Most obstructing point strictly between node a and node b of every link.

    Returns (v_max, index, valid). valid is False when the sub-path has no interior node.
    """
    n = Hf.shape[1]
    ia = np.arange(n)[None, :]
    Ha, Hb = _gather(Hf, a), _gather(Hf, b)
    sa, sb = s[a], s[b]
    span = np.where(sb > sa, sb - sa, 1.0)
    line = Ha[:, None] + (s[None, :] - sa[:, None]) / span[:, None] * (Hb - Ha)[:, None]
    d1 = np.maximum((s[None, :] - sa[:, None]) * D[:, None], 1e-6)
    d2 = np.maximum((sb[:, None] - s[None, :]) * D[:, None], 1e-6)
    v = fresnel_v(Hf - line, d1, d2, f_mhz)
    mask = (ia > a[:, None]) & (ia < b[:, None])
    v = np.where(mask, v, -np.inf)
    k = np.argmax(v, axis=1)
    return np.maximum(_gather(v, k), -10.0), k, (b - a) >= 2


def terrain_profile(dem: Dem, tx_xy, tx_h_abs: float, rx_xy, rx_h_abs, n: int = 100):
    """Effective-height profiles (curvature folded in) from one transmitter to many receivers.

    Returns (s, D, Hf): node fractions (n+2,), path lengths (N,), heights (N, n+2).
    """
    rx_xy = np.atleast_2d(np.asarray(rx_xy, dtype=np.float64))
    rx_h = np.broadcast_to(np.asarray(rx_h_abs, dtype=np.float64), (len(rx_xy),))
    s = np.linspace(0.0, 1.0, n + 2)
    dx, dy = rx_xy[:, 0] - tx_xy[0], rx_xy[:, 1] - tx_xy[1]
    D = np.maximum(np.hypot(dx, dy), 1.0)
    X = tx_xy[0] + s[None, :] * dx[:, None]
    Y = tx_xy[1] + s[None, :] * dy[:, None]
    z = dem.elevation(X, Y)
    bulge = s[None, :] * (1.0 - s[None, :]) * D[:, None] ** 2 / (2.0 * K_FACTOR * EARTH_R)
    Hf = z + bulge
    Hf[:, 0] = tx_h_abs
    Hf[:, -1] = rx_h
    return s, D, Hf


def diffraction_losses(s, D, Hf, f_mhz: float):
    """Deygout (3 edges) and Epstein-Peterson losses in dB for many links. Returns (deygout, ep)."""
    N, n = Hf.shape
    zero = np.zeros(N, dtype=np.int64)
    last = np.full(N, n - 1, dtype=np.int64)
    vm, m, ok = _search(Hf, s, D, zero, last, f_mhz)
    main = ok & (vm > -0.78)
    lm = knife_edge_j(vm) * main
    vl, il, okl = _search(Hf, s, D, zero, m, f_mhz)
    vr, ir, okr = _search(Hf, s, D, m, last, f_mhz)
    hl = main & okl & (vl > -0.78)
    hr = main & okr & (vr > -0.78)
    deygout = lm + knife_edge_j(vl) * hl + knife_edge_j(vr) * hr

    # Epstein-Peterson: same edges, each one judged against its neighbouring edges.
    def edge(e, a, b):
        Ha, Hb, He = _gather(Hf, a), _gather(Hf, b), _gather(Hf, e)
        sa, sb, se = s[a], s[b], s[e]
        span = np.where(sb > sa, sb - sa, 1.0)
        line = Ha + (se - sa) / span * (Hb - Ha)
        d1 = np.maximum((se - sa) * D, 1e-6)
        d2 = np.maximum((sb - se) * D, 1e-6)
        return knife_edge_j(fresnel_v(He - line, d1, d2, f_mhz))

    left_nb = np.where(hl, il, zero)
    right_nb = np.where(hr, ir, last)
    ep = (edge(m, left_nb, right_nb) * main
          + edge(il, zero, m) * hl
          + edge(ir, m, last) * hr)
    return deygout, ep


def terrain_diffraction(dem: Dem, tx_xy, tx_h_abs: float, rx_xy, rx_h_abs, f_mhz: float, n: int = 100):
    s, D, Hf = terrain_profile(dem, tx_xy, tx_h_abs, rx_xy, rx_h_abs, n)
    return diffraction_losses(s, D, Hf, f_mhz)
