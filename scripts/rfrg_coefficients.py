#!/usr/bin/env python3
"""Central definition of the rFRG coefficients c_d.

The renormalized viscosity used by the plot scripts is

    etaR = sqrt(eta^2 + 2 c_d T rho Lam^(d-2)),

equivalently etaR/eta = sqrt(1 + 2 c_d Re^2) with Re = sqrt(T rho Lam^(d-2))/eta.
For each dimension two finite-size values (coeff1, coeff2) bracket the band that
is drawn as a shaded region, and coeff_inf is the L=infinity value drawn as a
line. In 2D no L=infinity value is available, so coeff_inf is None there and
callers either skip the corresponding curve or fall back to the mean of the two
finite-size values.

Every plot script imports these numbers from here; do not copy them into the
scripts again.
"""

# Two-dimensional grid (nz == 1).
COEFF1_2D = 0.0521183#0.0522796
COEFF2_2D = 0.0411101#0.0414326
COEFF_INF_2D = None  # no L = infinity value available in 2D

# Three-dimensional grid.
COEFF1_3D = 0.0202115#0.0212045
COEFF2_3D = 0.0183903#0.0201104
COEFF_INF_3D = 0.0236416  # L = infinity


def coefficients(dim):
    """Return (coeff1, coeff2, coeff_inf) for `dim` spatial dimensions."""
    if dim == 2:
        return COEFF1_2D, COEFF2_2D, COEFF_INF_2D
    if dim == 3:
        return COEFF1_3D, COEFF2_3D, COEFF_INF_3D
    raise ValueError(f"no rFRG coefficients available for dim = {dim}")
