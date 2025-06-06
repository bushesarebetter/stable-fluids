import argparse
from functools import lru_cache

import numpy as np
import matplotlib
from PIL import Image

# Grid arrays are (n+2, n+2) with one ghost cell on each side, indexed [i, j] = [x, y].
# The domain is the unit square, so the cell size is h = 1/n.


def set_bnd(b, x):
    # b=1: x-velocity (flipped at the left/right walls), b=2: y-velocity, b=0: scalar
    x[0, 1:-1] = -x[1, 1:-1] if b == 1 else x[1, 1:-1]
    x[-1, 1:-1] = -x[-2, 1:-1] if b == 1 else x[-2, 1:-1]
    x[1:-1, 0] = -x[1:-1, 1] if b == 2 else x[1:-1, 1]
    x[1:-1, -1] = -x[1:-1, -2] if b == 2 else x[1:-1, -2]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])


@lru_cache
def grid(n):
    return np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")


@lru_cache
def checkerboard(n):
    i, j = grid(n)
    return [(i + j) % 2 == 0, (i + j) % 2 == 1]


def lin_solve(b, x, x0, a, c, iters):
    # Solves c*x - a*(sum of 4 neighbours) = x0 with red-black Gauss-Seidel.
    # Each colour only depends on the other one, so a half-sweep is a single vectorized update.
    for _ in range(iters):
        for mask in checkerboard(x.shape[0] - 2):
            nb = x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
            x[1:-1, 1:-1][mask] = ((x0[1:-1, 1:-1] + a * nb) / c)[mask]
        set_bnd(b, x)


def diffuse(b, x0, rate, dt, iters):
    n = x0.shape[0] - 2
    a = dt * rate * n * n
    x = x0.copy()
    lin_solve(b, x, x0, a, 1 + 4 * a, iters)
    return x


def advect(b, d0, u, v, dt):
    # Semi-Lagrangian: trace each cell centre back along the velocity and sample d0 there.
    n = d0.shape[0] - 2
    i, j = grid(n)
    x = np.clip(i - dt * n * u[1:-1, 1:-1], 0.5, n + 0.5)
    y = np.clip(j - dt * n * v[1:-1, 1:-1], 0.5, n + 0.5)
    i0 = np.floor(x).astype(int)
    j0 = np.floor(y).astype(int)
    s, t = x - i0, y - j0
    d = np.zeros_like(d0)
    d[1:-1, 1:-1] = ((1 - s) * ((1 - t) * d0[i0, j0] + t * d0[i0, j0 + 1])
                     + s * ((1 - t) * d0[i0 + 1, j0] + t * d0[i0 + 1, j0 + 1]))
    set_bnd(b, d)
    return d


def divergence(u, v):
    n = u.shape[0] - 2
    return 0.5 * n * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])


def project(u, v, iters):
    # Solve laplace(p) = div(u) and subtract grad(p), in place.
    n = u.shape[0] - 2
    h = 1.0 / n
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = -h * h * divergence(u, v)
    set_bnd(0, div)
    p = np.zeros_like(u)
    lin_solve(0, p, div, 1, 4, iters)
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
    set_bnd(1, u)
    set_bnd(2, v)


def step(u, v, dens, visc, diff, dt, iters):
    u, v = diffuse(1, u, visc, dt, iters), diffuse(2, v, visc, dt, iters)
    project(u, v, iters)
    u, v = advect(1, u, u, v, dt), advect(2, v, u, v, dt)
    project(u, v, iters)
    dens = advect(0, diffuse(0, dens, diff, dt, iters), u, v, dt)
    return u, v, dens








