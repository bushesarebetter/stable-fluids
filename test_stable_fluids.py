import numpy as np

import stable_fluids as sf


def smooth_field(n):
    xs = (np.arange(n + 2) - 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    env = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.03)
    u = (np.sin(5 * X) + np.cos(7 * Y)) * env
    v = (np.cos(3 * X * Y) + X) * env
    sf.set_bnd(1, u)
    sf.set_bnd(2, v)
    return u, v


def rms(a):
    return np.sqrt((a ** 2).mean())


def test_projection_removes_divergence():
    u, v = smooth_field(32)
    before = rms(sf.divergence(u, v))
    sf.project(u, v, iters=1000)
    after = rms(sf.divergence(u, v))
    assert after < 0.1 * before


def test_projection_with_few_iterations_still_helps():
    u, v = smooth_field(32)
    before = rms(sf.divergence(u, v))
    sf.project(u, v, iters=20)
    assert rms(sf.divergence(u, v)) < 0.6 * before


def test_zero_velocity_advection_is_identity():
    rng = np.random.default_rng(0)
    d0 = rng.random((34, 34))
    zero = np.zeros_like(d0)
    d = sf.advect(0, d0, zero, zero, dt=0.1)
    np.testing.assert_allclose(d[1:-1, 1:-1], d0[1:-1, 1:-1], atol=1e-12)


def test_uniform_flow_shifts_dye():
    # u = 1 cell per step to the right: the blob should move exactly one cell
    n = 32
    d0 = np.zeros((n + 2, n + 2))
    d0[10, 15] = 1.0
    u = np.full_like(d0, 1.0 / n)
    d = sf.advect(0, d0, u, np.zeros_like(d0), dt=1.0)
    assert d[11, 15] == 1.0 and d.sum() == 1.0
