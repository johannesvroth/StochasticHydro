#!/usr/bin/env python3
"""Plot 1/Re_R against 1/Re, with Re_R = sqrt(rho T Lam^(d-2))/etaR.

The bare Reynolds number is Re = sqrt(rho T Lam^(d-2))/eta, so with
etaR/eta = sqrt(1 + 2 c_d Re^2) the renormalized one follows as

    1/Re_R = etaR/sqrt(rho T Lam^(d-2)) = sqrt(1/Re^2 + 2 c_d),

i.e. as a function of x = 1/Re the curve is sqrt(x^2 + 2 c_d), which
saturates at sqrt(2 c_d) in the limit of vanishing bare viscosity.

The measured points are fitted from the correlator directories given on the
command line, one point per directory, by the standard procedure in
etaR_fit.py; eta and Lam come from each directory name, so 1/Re = eta/sqrt(rho
T Lam) and 1/Re_R = etaR/sqrt(rho T Lam). Without any directory only the
curves are drawn, e.g.

    plot_inv_re_renormalized.py \\
        avg-jp-time-corr-Nx32Ny32Nz32dt0.2eta0.5Lam0.4nk1 \\
        avg-jp-time-corr-Nx32Ny32Nz32dt1eta0.2Lam0.4nk1 \\
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk1 \\
        --etaR-table etaR-d3g1rho1T1Lam0.4.dat
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style
import etaR_fit
from rfrg_coefficients import coefficients

plot_style.use()

AUTO_OUTPUT = Path("<auto>")

FLOAT = r"[-+0-9.]+(?:[eE][+-]?[0-9]+)?"

def read_etaR_table(path: Path) -> tuple:
    """Read a tabulated etaR(eta) file and return (1/Re, 1/Re_R).

    The parameters of the table are taken from its header rather than from
    the file name, so the conversion uses the same rho, T and Lam the table
    was made with: with the scale sqrt(rho T Lam^(d-2)) both columns are
    divided by, eta becomes 1/Re and etaR becomes 1/Re_R."""
    with open(path) as f:
        header = f.readline()
    if not header.startswith("#"):
        raise SystemExit(f"{path}: expected a '# d = ... Lam = ...' header line")
    params = {key: float(value) for key, value
              in re.findall(rf"(\w+)\s*=\s*({FLOAT})", header)}
    missing = {"d", "rho", "T", "Lam"} - params.keys()
    if missing:
        raise SystemExit(f"{path}: header is missing {', '.join(sorted(missing))}")
    dim = int(params["d"])
    if dim != 3:
        raise SystemExit(f"{path}: d = {dim}, but this script draws the 3D curve")

    eta, etaR = np.loadtxt(path, unpack=True)
    # Re = sqrt(rho T Lam^(d-2))/eta and Re_R the same with etaR, so both
    # inverse Reynolds numbers are the viscosities in units of that scale.
    scale = np.sqrt(params["rho"]*params["T"]*params["Lam"]**(dim - 2))
    # Sorted by eta, so the caller can spline over it without checking again.
    order = np.argsort(eta)
    return eta[order]/scale, etaR[order]/scale


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
                              "(default: --xlim if given, else 0 to 0.2, "
                              "extended when a data point reaches further)")
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
                              "without an argument, save to figs/inv-re-renormalized.pdf")
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
        args.output = Path("figs") / "inv-re-renormalized.pdf"
        args.output.parent.mkdir(exist_ok=True)

    coeff1, coeff2, coeff_inf = coefficients(3)

    fits = etaR_fit.fit_dirs(args.input_dirs, args)
    inv_re_data = np.array([f.inv_reynolds for f in fits])
    inv_reR_data = np.array([f.inv_reynolds_r for f in fits])
    inv_reR_err = np.array([f.inv_reynolds_r_err for f in fits])

    if args.etaR_table is not None:
        if not args.etaR_table.exists():
            raise SystemExit(f"No such etaR table: {args.etaR_table}")
        inv_re_table, inv_reR_table = read_etaR_table(args.etaR_table)
    else:
        inv_re_table = inv_reR_table = np.array([])

    if args.inv_re_range is not None:
        inv_re_range = args.inv_re_range
    elif args.xlim is not None:
        # Sample the whole visible range, so the curve is not cut off.
        inv_re_range = (min(args.xlim), max(args.xlim))
    else:
        inv_re_max = max([*inv_re_data, *inv_re_table], default=0.0)
        inv_re_range = (0.0, max(0.2, 1.1*inv_re_max))

    if args.xlog and inv_re_range[0] > 0.0:
        inv_re = np.geomspace(*inv_re_range, args.num_points)
    else:
        inv_re = np.linspace(*inv_re_range, args.num_points)
    curve1 = np.sqrt(inv_re**2 + 2.0*coeff1)
    curve2 = np.sqrt(inv_re**2 + 2.0*coeff2)
    curve_inf = np.sqrt(inv_re**2 + 2.0*coeff_inf)
    # One-loop perturbation theory, 1/Re + c_d Re: the large-x expansion of
    # sqrt(x^2 + 2 c_d) = x sqrt(1 + 2 c_d/x^2) to first order in c_d/x^2.
    # It follows the full curve where the correction is small and runs away
    # from it once c_d Re^2 is of order one, diverging as 1/Re -> 0, where
    # the expansion parameter is no longer small.
    curve_1loop = inv_re + coeff_inf/inv_re

    fig, ax = plt.subplots()
    # Free theory etaR = eta, i.e. the diagonal the curves approach for Re -> 0.
    ax.plot(inv_re, inv_re, "--", color="gray",
            label=r"$1/\mathrm{Re}_R = 1/\mathrm{Re}$")
    line, = ax.plot(inv_re, curve_inf,
                    label=r"rFRG $\sqrt{\mathrm{Re}^{-2} + 2c_d}$, $L=\infty$")
    ax.fill_between(inv_re, curve2, curve1, color=line.get_color(), alpha=0.3,
                    label=r"rFRG $\sqrt{\mathrm{Re}^{-2} + 2c_d}$")
    ax.plot(inv_re, curve_1loop, "--", color="tab:green",
            label=r"1-loop perturbation theory ($\mathrm{Re}^{-1} + c_d\,\mathrm{Re}$)")
    if len(inv_re_table):
        # Straight segments between the tabulated eta, without markers.
        ax.plot(inv_re_table, inv_reR_table, "-", color="tab:red",
                label=r"rFRG flow, self-consistent p-dep")
    if len(fits):
        ax.errorbar(inv_re_data, inv_reR_data, yerr=inv_reR_err, fmt="o",
                    capsize=3, color="black", label="simulation")
    ax.set_xlabel(r"$1/\mathrm{Re}$")
    ax.set_ylabel(r"$1/\mathrm{Re}_R$")
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
