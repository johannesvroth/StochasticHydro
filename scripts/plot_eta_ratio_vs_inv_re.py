#!/usr/bin/env python3
"""Plot etaR/eta against 1/Re, with Re = sqrt(rho T Lam^(d-2))/eta.

This is the same comparison as plot_inv_re_renormalized.py, drawn as the
enhancement of the viscosity over its bare value rather than as the
renormalized inverse Reynolds number: with

    etaR/eta = sqrt(1 + 2 c_d Re^2),

the curve as a function of x = 1/Re is sqrt(1 + 2 c_d/x^2), i.e. the curve
sqrt(x^2 + 2 c_d) of that script divided by x. It approaches 1 for large x
(large bare viscosity, where the renormalization is a small correction) and
diverges as 1/x for x -> 0, where etaR saturates at sqrt(2 c_d rho T Lam)
while eta itself goes to zero.

Since the ratio grows without bound towards small 1/Re, while the points of
interest sit at 1 for large 1/Re, the plot normally wants --xlog --ylog; on
linear axes the divergence at the left edge flattens everything else.

The measured points are fitted from the correlator directories given on the
command line, one point per directory, by the standard procedure in
etaR_fit.py; eta and Lam come from each directory name, so 1/Re = eta/sqrt(rho
T Lam). Without any directory only the curves are drawn, e.g.

    plot_eta_ratio_vs_inv_re.py \\
        avg-jp-time-corr-Nx32Ny32Nz32dt0.2eta0.5Lam0.4nk1 \\
        avg-jp-time-corr-Nx32Ny32Nz32dt1eta0.2Lam0.4nk1 \\
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk1 \\
        --etaR-table etaR-d3g1rho1T1Lam0.4.dat
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style
import etaR_fit
from plot_inv_re_renormalized import read_etaR_table
from rfrg_coefficients import coefficients

plot_style.use()

AUTO_OUTPUT = Path("<auto>")

# Smallest 1/Re the curves are drawn down to by default. Unlike 1/Re_R the
# ratio diverges at x = 0, so the default range starts away from it (and is
# extended when a data point or a tabulated eta reaches further down).
DEFAULT_INV_RE_MIN = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dirs", type=Path, nargs="*",
                         help="Directories containing the *.dat correlator files "
                              "written by compute_avg_jp_time_correlator.py, named "
                              "like avg-jp-time-corr-Nx8Ny8Nz8dt0.1eta0.1Lam1.25nk7 "
                              "(eta and Lam are parsed from the name); each is "
                              "fitted for etaR and contributes one point")
    etaR_fit.add_fit_arguments(parser)
    parser.add_argument("--inv-re-range", type=float, nargs=2, default=None,
                         metavar=("XMIN", "XMAX"),
                         help="Range of 1/Re over which to plot the curve "
                              f"(default: --xlim if given, else "
                              f"{DEFAULT_INV_RE_MIN:g} to 0.2, extended when a "
                              "data point reaches further)")
    parser.add_argument("--num-points", type=int, default=500,
                         help="Number of sample points")
    parser.add_argument("--etaR-table", type=Path, default=None,
                         metavar="FILE",
                         help="Tabulated etaR(eta) to draw as a curve, as two "
                              "columns eta, etaR preceded by a header line "
                              "'# d = 3  g = 1  rho = 1  T = 1  Lam = 0.4' "
                              "fixing the parameters it was made with, e.g. "
                              "etaR-d3g1rho1T1Lam0.4.dat; without this option "
                              "no such curve is drawn")
    parser.add_argument("-o", "--output", type=Path, nargs="?", default=None,
                         const=AUTO_OUTPUT,
                         help="Save the plot to this file instead of showing it; "
                              "without an argument, save to figs/eta-ratio-vs-inv-re.pdf")
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
        args.output = Path("figs") / "eta-ratio-vs-inv-re.pdf"
        args.output.parent.mkdir(exist_ok=True)

    # coeff1, coeff2, coeff_inf = coefficients(3)
    _, _, coeff_inf = coefficients(3)
    
    fits = etaR_fit.fit_dirs(args.input_dirs, args, with_ratio=True)
    inv_re_data = np.array([f.inv_reynolds for f in fits])
    ratio_data = np.array([f.ratio for f in fits])
    ratio_err = np.array([f.ratio_err for f in fits])

    if args.etaR_table is not None:
        if not args.etaR_table.exists():
            raise SystemExit(f"No such etaR table: {args.etaR_table}")
        inv_re_table, inv_reR_table = read_etaR_table(args.etaR_table)
        # Both columns carry the same scale sqrt(rho T Lam^(d-2)), so their
        # quotient is etaR/eta regardless of it.
        ratio_table = inv_reR_table/inv_re_table
    else:
        inv_re_table = ratio_table = np.array([])

    if args.inv_re_range is not None:
        inv_re_range = args.inv_re_range
    elif args.xlim is not None:
        # Sample the whole visible range, so the curve is not cut off.
        inv_re_range = (min(args.xlim), max(args.xlim))
    else:
        inv_re_max = max([*inv_re_data, *inv_re_table], default=0.0)
        inv_re_min = min([*inv_re_data, *inv_re_table],
                         default=DEFAULT_INV_RE_MIN)
        inv_re_range = (0.9*min(inv_re_min, DEFAULT_INV_RE_MIN),
                        max(0.2, 1.1*inv_re_max))

    if args.xlog and inv_re_range[0] > 0.0:
        inv_re = np.geomspace(*inv_re_range, args.num_points)
    else:
        inv_re = np.linspace(*inv_re_range, args.num_points)
    # The ratios diverge at 1/Re = 0, which only a hand-picked range reaches;
    # the point is then dropped from the plot rather than warned about.
    with np.errstate(divide="ignore"):
        inv_re2 = inv_re**2
        # curve1 = np.sqrt(1.0 + 2.0*coeff1/inv_re2)
        # curve2 = np.sqrt(1.0 + 2.0*coeff2/inv_re2)
        curve_inf = np.sqrt(1.0 + 2.0*coeff_inf/inv_re2)
        # One-loop perturbation theory, 1 + c_d Re^2: the small-Re expansion of
        # sqrt(1 + 2 c_d Re^2) to first order in c_d Re^2. It follows the full
        # curve while the correction is small and runs away from it once
        # c_d Re^2 is of order one, growing as Re^2 instead of Re.
        curve_1loop = 1.0 + coeff_inf/inv_re2

    fig, ax = plt.subplots()
    # Free theory etaR = eta, the line the curves approach for Re -> 0.
    ax.axhline(1.0, ls="--", color="black", alpha=0.7, label=r"$\eta_R = \eta$")
    ax.plot(inv_re, curve_inf, color="black",
            label=r"rFRG $\sqrt{1 + 2c_d\mathrm{Re}^2}$")
    # ax.fill_between(inv_re, curve2, curve1, color="black", alpha=0.3,
    #                 label=r"rFRG $\sqrt{1 + 2c_d\mathrm{Re}^2}$")
    ax.plot(inv_re, curve_1loop, "--", color="tab:green",
            label=r"1-loop perturbation theory ($1 + c_d\,\mathrm{Re}^2$)")
    if len(inv_re_table):
        # Straight segments between the tabulated eta, without markers.
        ax.plot(inv_re_table, ratio_table, "-", color="tab:red",
                label=r"rFRG flow, self-consistent $k$-dep.")
    if len(fits):
        ax.errorbar(inv_re_data, ratio_data, yerr=ratio_err, fmt="o",
                    capsize=3, color="C0", label="simulation")
    ax.set_xlabel(r"$1/\mathrm{Re}$")
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
