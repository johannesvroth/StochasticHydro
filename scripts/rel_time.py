#!/usr/bin/env python3
r"""Print the relaxation time of a single lattice mode.

The dissipative step of the simulation evolves every Fourier mode as an exact
Ornstein-Uhlenbeck step with the damping rate

    damp = eta (1 + eta_reg_uv(k_hat^2)) k_hat^2/rho,

so the mode relaxes, and its time correlator decays, as exp(-damp t) with the
relaxation time tau = 1/damp. The momentum that enters is the lattice momentum

    k_hat_i = 2 sin(k_i/2),   k_i = 2 pi nk_i/N_i,

the square root of the eigenvalue of the nearest-neighbour Laplacian, not the
continuum k: the two part company towards the edge of the Brillouin zone, and
it is k_hat that the measured damping rate follows (see etaR_fit.py). The
regulator is imported from etaR_fit so there is one definition of it.

    rel_time.py --nx 32 --ny 32 --nz 32 \
        --eta 0.0707107 --eta-uv-cutoff 0.4 --nkx 0 --nky 1 --nkz 0

The slowest and the fastest mode of the same lattice are printed underneath,
which is what a time step has to resolve: the fastest mode sits at the corner
of the Brillouin zone, k_hat^2 = 4 d, and the slowest at the smallest nonzero
momentum, 2 pi/max(N)."""

import argparse

import numpy as np

from etaR_fit import eta_reg_uv


def damping_rate(k_hat2, eta, lam, mass_density):
    """The rate exp(-damp t) at which a mode of lattice momentum k_hat decays."""
    return (1 + eta_reg_uv(k_hat2, lam))*eta*k_hat2/mass_density


def k_hat(nk, n_sites):
    """Lattice momentum 2 sin(k/2) of mode index nk on n_sites sites."""
    return 2.0*np.sin(np.pi*nk/n_sites)


def mode_k_hat2(nk, shape):
    """Squared lattice momentum of the mode (nkx, nky, nkz)."""
    return sum(k_hat(nk_i, n_i)**2 for nk_i, n_i in zip(nk, shape))


def report(label, k_hat2, eta, lam, mass_density):
    damp = damping_rate(k_hat2, eta, lam, mass_density)
    k_hat_abs = np.sqrt(k_hat2)
    tau = np.inf if damp == 0 else 1/damp
    print(f"{label:>32}= {tau:<12.6g} (k_hat={k_hat_abs:g}, damp={damp:g})")
    return tau


def main():
    parser = argparse.ArgumentParser(
        description="Print the relaxation time 1/damp of a single lattice mode.")
    parser.add_argument("--nx", type=int, required=True,
                        help="Number of lattice sites in x")
    parser.add_argument("--ny", type=int, required=True,
                        help="Number of lattice sites in y")
    parser.add_argument("--nz", type=int, required=True,
                        help="Number of lattice sites in z")
    parser.add_argument("--nkx", type=int, default=0, help="Mode index in x")
    parser.add_argument("--nky", type=int, default=0, help="Mode index in y")
    parser.add_argument("--nkz", type=int, default=0, help="Mode index in z")
    parser.add_argument("--eta", type=float, default=1.0, help="Shear viscosity")
    parser.add_argument("--eta-uv-cutoff", type=float, default=1.25,
                        help="UV cutoff Lam for the shear viscosity")
    parser.add_argument("--mass-density", type=float, default=1.0,
                        help="Mass density rho")
    args = parser.parse_args()

    shape = (args.nx, args.ny, args.nz)
    nk = (args.nkx, args.nky, args.nkz)

    if any(n < 1 for n in shape):
        raise SystemExit("Lattice sizes must be at least 1")

    k = [2*np.pi*nk_i/n_i for nk_i, n_i in zip(nk, shape)]
    print(f"Mode nk=({args.nkx}, {args.nky}, {args.nkz}) on a "
          f"{args.nx}x{args.ny}x{args.nz} lattice, eta={args.eta:g}, "
          f"Lam={args.eta_uv_cutoff:g}, rho={args.mass_density:g}")
    print(f"                               k= ({k[0]:g}, {k[1]:g}, {k[2]:g}), "
          f"|k|={np.linalg.norm(k):g}")

    k_hat2 = mode_k_hat2(nk, shape)
    if k_hat2 == 0:
        print("\nThis mode has k_hat=0: it is the conserved total momentum, "
              "which is not damped at all.")

    tau = report("Relaxation time of this mode", k_hat2, args.eta,
                 args.eta_uv_cutoff, args.mass_density)

    # The slowest mode is the smallest nonzero momentum, which lives in the
    # longest direction; the fastest is the corner of the Brillouin zone,
    # k_hat_i = 2 sin(pi floor(N_i/2)/N_i), i.e. k_hat^2 = 4 d for even N.
    slow_dir = shape.index(max(shape))
    slow_nk = tuple(1 if i == slow_dir else 0 for i in range(3))
    fast_nk = tuple(n_i//2 for n_i in shape)

    report("Relaxation time of slowest mode",
           mode_k_hat2(slow_nk, shape), args.eta, args.eta_uv_cutoff,
           args.mass_density)
    report("Relaxation time of fastest mode",
           mode_k_hat2(fast_nk, shape), args.eta, args.eta_uv_cutoff,
           args.mass_density)

    return tau


if __name__ == "__main__":
    main()
