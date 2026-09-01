#!/usr/bin/env python3
"""Plot the renormalized viscosity ratio etaR/eta as a function of a*Lam,
with a = 1 the lattice spacing.

For every input directory the renormalized shear viscosity etaR is measured
by the standard procedure in etaR_fit.py: the constant -dampR is fitted,
unweighted, to the per-run logarithmic derivative of the time correlator over
the window [tau, 2 tau]. The plotted error bar is the statistical error
(run-to-run scatter) and the systematic one (the shift when the same window is
moved half a tau later, to [1.5 tau, 2.5 tau]) added in quadrature.

The bare eta and the cutoff Lam are parsed from the directory name, so the
measured ratio etaR/eta is drawn against a*Lam with one point per directory,
e.g.

    plot_eta_ratio_vs_lam.py \\
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk1 \\
        avg-jp-time-corr-Nx48Ny48Nz48dt10eta0.057735Lam0.266667nk1 \\
        avg-jp-time-corr-Nx64Ny64Nz64dt10eta0.05Lam0.2nk1

The theory expectation etaR/eta = sqrt(1 + 2 c_d T rho Lam / eta^2) is drawn
on top, as the L=infinity curve and as a band spanning the two finite-volume
coefficients. It depends on eta and Lam separately, not on a*Lam alone, so it
is evaluated at each directory's own (eta, Lam) and the values are simply
connected in order of increasing a*Lam; for a scan at fixed Reynolds number
Re = sqrt(T rho Lam)/eta (as in the example above, where Lam ~ 1/N keeps both
Re and the physical volume Lam*N fixed) the expectation is a constant, and
any slope in the measured points is a lattice artifact."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style
import etaR_fit
from rfrg_coefficients import coefficients

plot_style.use()

AUTO_OUTPUT = Path("<auto>")

# The lattice spacing is 1 in the units the simulation works in.
LATTICE_SPACING = 1.0

COEFF1, COEFF2, COEFF_INF = coefficients(3)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dirs", type=Path, nargs="+",
                         help="One or more directories containing the *.dat correlator "
                              "files written by compute_jpx_time_correlator.py, named "
                              "like avg-jp-time-corr-Nx8Ny8Nz8dt0.1eta0.1Lam1.25nk7 (eta "
                              "and Lam are parsed from the name); each contributes one "
                              "point to the plot")
    etaR_fit.add_fit_arguments(parser)
    parser.add_argument("--no-annotate", dest="annotate", action="store_false",
                         help="Do not label the points with their lattice size")
    parser.add_argument("-o", "--output", type=Path, nargs="?", default=None,
                         const=AUTO_OUTPUT,
                         help="Save the plot to this file instead of showing it; "
                              "without an argument, save to figs/eta-ratio-vs-lam.pdf")
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                         metavar=("XMIN", "XMAX"),
                         help="x-axis limits of the plot")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                         metavar=("YMIN", "YMAX"),
                         help="y-axis limits of the plot")
    parser.add_argument("--xlog", action="store_true",
                         help="Use a logarithmic x-axis")
    parser.add_argument("--ylog", action="store_true",
                         help="Use a logarithmic y-axis")
    args = parser.parse_args()

    if args.output is AUTO_OUTPUT:
        args.output = Path("figs") / "eta-ratio-vs-lam.pdf"
        args.output.parent.mkdir(exist_ok=True)

    results = etaR_fit.fit_dirs(args.input_dirs, args, with_ratio=True)
    results.sort(key=lambda r: LATTICE_SPACING*r.lam)

    a_lam = np.array([LATTICE_SPACING*r.lam for r in results])
    eta = np.array([r.eta for r in results])
    ratio = np.array([r.ratio for r in results])
    ratio_err = np.array([r.ratio_err for r in results])

    # The theory ratio depends on eta and Lam separately, so it can only be
    # evaluated at the (eta, Lam) of the directories themselves.
    def theory(coeff):
        return np.sqrt(1.0 + 2.0*coeff*args.temp*args.mass_density
                       * np.array([r.lam for r in results]) / eta**2)

    fig, ax = plt.subplots()
    line, = ax.plot(a_lam, theory(COEFF_INF), marker=".",
                    label=r"$\sqrt{1 + 2c_d \mathrm{Re}^2}$, $L=\infty$")
    ax.fill_between(a_lam, theory(COEFF2), theory(COEFF1),
                    color=line.get_color(), alpha=0.3,
                    label=r"$\sqrt{1 + 2c_d \mathrm{Re}^2}$")
    ax.errorbar(a_lam, ratio, yerr=ratio_err, fmt="o", capsize=3, color="black",
                label="simulation")
    if args.annotate:
        for x, y, r in zip(a_lam, ratio, results):
            ax.annotate(f"$N={r.nx}$", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize="small")
    ax.set_xlabel(r"$a\Lambda$")
    ax.set_ylabel(r"$\eta_R/\eta$")
    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.xlog:
        ax.set_xscale('log')
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    if args.ylog:
        ax.set_yscale('log')
    ax.legend()
    fig.tight_layout(pad=0.2)

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
