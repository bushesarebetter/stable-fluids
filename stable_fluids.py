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


def add_sources(u, v, dens, t, dt, n):
    # Three emitters orbit the centre and push dye tangentially, which spins up a vortex.
    xs = (np.arange(n + 2) - 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    for k in range(3):
        ang = 0.6 * t + 2 * np.pi * k / 3
        cx, cy = 0.5 + 0.22 * np.cos(ang), 0.5 + 0.22 * np.sin(ang)
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.02 ** 2))
        # mostly tangential, slightly inward, wobbling over time so the vortex never settles
        jet = ang + np.pi / 2 + 0.4 + 0.5 * np.sin(1.7 * t + k)
        u += dt * 4.0 * np.cos(jet) * blob
        v += dt * 4.0 * np.sin(jet) * blob
        dens += dt * 6.0 * blob


def render(dens, scale):
    img = matplotlib.colormaps["inferno"](np.clip(dens[1:-1, 1:-1].T, 0, 1))[::-1, :, :3]
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--visc", type=float, default=1e-5)
    ap.add_argument("--diff", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="fluid.gif")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n, args.steps = 64, 150

    n = args.n
    u, v, dens = (np.zeros((n + 2, n + 2)) for _ in range(3))
    frames = []
    for s in range(args.steps):
        add_sources(u, v, dens, s * args.dt, args.dt, n)
        u, v, dens = step(u, v, dens, args.visc, args.diff, args.dt, args.iters)
        dens *= 0.995
        if s % 3 == 0:
            frames.append(render(dens, max(1, 256 // n)))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"{args.steps} steps on a {n}x{n} grid, wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
