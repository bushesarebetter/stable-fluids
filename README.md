# Stable Fluids

A 2D implementation of Jos Stam's [Stable Fluids](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) (SIGGRAPH 1999) in plain numpy. Three emitters circle the middle of a closed box and push dye sideways, and the solver turns that into a slowly churning vortex.

![dye density](fluid.gif)

Each step diffuses the velocity, projects it to be divergence-free, advects it along itself, projects again, then diffuses and advects the dye. Advection is semi-Lagrangian with vectorized bilinear lookups. Diffusion and the pressure Poisson solve both use red-black Gauss-Seidel, which updates half the grid at a time so each sweep is one numpy expression.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python stable_fluids.py           # 128x128, 360 steps, about 35 s, writes fluid.gif
python stable_fluids.py --quick   # 64x64, 150 steps
```

Grid size, time step, viscosity, diffusion and solver iterations are command-line flags (`--help`).

## Test

```
python -m pytest -q
```

The tests check that projection cuts the divergence of a smooth field, that advecting with zero velocity returns the field unchanged, and that a uniform flow of one cell per step moves dye by exactly one cell.

## Results

On a 32x32 smooth test field, projection with 20 Gauss-Seidel iterations (the default) brings the RMS divergence down to about 40% of where it started, and 1000 iterations bring it to about 3.5%. It doesn't reach zero because the pressure solve uses the 5-point Laplacian while divergence is measured with central differences. Stam's method has the same mismatch. Gauss-Seidel is slow on large-scale modes, so a multigrid or conjugate-gradient solve would be the next step if you needed tighter incompressibility.
