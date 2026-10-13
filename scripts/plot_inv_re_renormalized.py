#!/usr/bin/env python3
"""Plot 1/Re_R against 1/Re, with Re_R = sqrt(rho T Lam^(d-2))/etaR.

The bare Reynolds number is Re = sqrt(rho T Lam^(d-2))/eta, so with
etaR/eta = sqrt(1 + 2 c_d Re^2) the renormalized one follows as

    1/Re_R = etaR/sqrt(rho T Lam^(d-2)) = sqrt(1/Re^2 + 2 c_d),

i.e. as a function of x = 1/Re the curve is sqrt(x^2 + 2 c_d), which
saturates at sqrt(2 c_d) in the limit of vanishing bare viscosity.

Below the plot sits a small panel with the relative deviation of the measured
points from the rFRG, carrying the same error bar divided by the same
reference. Since 1/Re_R is etaR in units of sqrt(rho T Lam^(d-2)), that
deviation is etaR/etaR^rFRG - 1, which is what the panel is labelled with.
Which rFRG it is follows what is drawn above: the tabulated flow of
--etaR-table when it is given, interpolated at the 1/Re of each point, and
otherwise the analytic sqrt(1/Re^2 + 2 c_d) at L = infinity.

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


def reference_inv_reR(inv_re, coeff_inf, inv_re_table, inv_reR_table):
    """The rFRG 1/Re_R the deviation panel measures the points against, as an
    array over the 1/Re of the points.

    With a tabulated flow it is that flow at the 1/Re of each point, nan where
    a point lies outside the tabulated range; with none it is the analytic
    sqrt(1/Re^2 + 2 c_d) drawn above. The table is read in the log of both
    axes, which is also how such a curve is looked at."""
    inv_re = np.asarray(inv_re, dtype=float)
    if len(inv_re_table):
        good = (inv_re_table > 0.0) & (inv_reR_table > 0.0)
        xs, ys = inv_re_table[good], inv_reR_table[good]
        out = np.full(inv_re.shape, np.nan)
        inside = (inv_re > 0.0) & (inv_re >= xs[0]) & (inv_re <= xs[-1])
        out[inside] = np.exp(np.interp(np.log(inv_re[inside]), np.log(xs),
                                       np.log(ys)))
        return out
    return np.sqrt(inv_re**2 + 2.0*coeff_inf)


def lattice_label(nx: int, ny: int, nz: int) -> str:
    r"""LaTeX for the lattice size: $32^3$ for a cubic lattice, $32^2 \times 10$
    when one direction differs from the two others."""
    dims = (nx, ny, nz)
    if nx == ny == nz:
        return rf"${nx}^3$"
    for d in dims:
        if dims.count(d) == 2:
            odd = next(o for o in dims if o != d)
            return rf"${d}^2 \times {odd}$"
    return rf"${nx} \times {ny} \times {nz}$"


def scan_key(fit):
    """The parameters a legend entry stands for: the lattice and the cutoff
    (and the 'x' suffix); eta varies along a scan and is the x axis."""
    return (fit.nx, fit.ny, fit.nz, fit.lam, fit.suffix)


def scan_label(fit) -> str:
    r"""Legend entry for the simulation points, e.g. "$32^3$, $a\Lambda = 0.4$"."""
    label = rf"{lattice_label(fit.nx, fit.ny, fit.nz)}, $a\Lambda = {fit.lam:g}$"
    if fit.suffix == "x":
        label += ", no ideal step"
    return label


# Colors of the simulation scans, skipping C1-C3, which the curves use.
SCAN_COLORS = ["C0", "C4", "C5", "C6", "C7", "C8", "C9"]


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
    parser.add_argument("--no-ratio", dest="ratio_panel",
                         action="store_false",
                         help="Do not draw the simulation/rFRG deviation "
                              "panel below the plot")
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
    parser.add_argument("--rlim", type=float, nargs=2, default=None,
                         metavar=("RMIN", "RMAX"),
                         help="y-axis limits of the deviation panel; without "
                              "it the panel is centred on zero and follows "
                              "its own points")
    parser.add_argument("--xlog", action="store_true",
                         help="Use a logarithmic x-axis")
    parser.add_argument("--ylog", action="store_true",
                         help="Use a logarithmic y-axis")
    args = parser.parse_args()

    if args.output is AUTO_OUTPUT:
        args.output = Path("figs") / "inv-re-renormalized.pdf"
        args.output.parent.mkdir(exist_ok=True)

    # Only the L = infinity coefficient is drawn; the band between the two
    # finite-volume ones is not.
    _, _, coeff_inf = coefficients(3)

    # Only the 3D theory is drawn, so only 3D lattices are accepted; checked
    # before any fit, as etaR_fit tells them apart the same way, by Nz = 1.
    flat = [d for d in args.input_dirs
            if etaR_fit.parse_obs_name(d)["nz"] == 1]
    if flat:
        raise SystemExit("Only d = 3 is supported, but these directories are "
                         "2D lattices: " + ", ".join(str(d) for d in flat))

    fits = etaR_fit.fit_dirs(args.input_dirs, args)
    inv_re_data = np.array([f.inv_reynolds for f in fits])
    inv_reR_data = np.array([f.inv_reynolds_r for f in fits])
    inv_reR_err = np.array([f.inv_reynolds_r_err for f in fits])
    # 1/Re and 1/Re_R in terms of the bare and the renormalized viscosity,
    # eta/sqrt(rho T Lam^(d-2)) and etaR/sqrt(rho T Lam^(d-2)) at d = 3.
    inv_re_label = r"$1/Re$"
    inv_reR_label = r"$\eta_R/\sqrt{\rho T \Lambda}$"

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
    curve_inf = np.sqrt(inv_re**2 + 2.0*coeff_inf)
    # One-loop perturbation theory, 1/Re + c_d Re: the large-x expansion of
    # sqrt(x^2 + 2 c_d) = x sqrt(1 + 2 c_d/x^2) to first order in c_d/x^2.
    # It follows the full curve where the correction is small and runs away
    # from it once c_d Re^2 is of order one, diverging as 1/Re -> 0, where
    # the expansion parameter is no longer small.
    curve_1loop = inv_re + coeff_inf/inv_re

    # The deviation panel shares the x axis of the plot proper, so the two are
    # read as one figure: the panel is only ever looked at at the 1/Re of a
    # point above it. Without points there is nothing to draw in it.
    if args.ratio_panel and len(fits):
        fig, ax, rax = plot_style.deviation_panel()
    else:
        fig, ax = plt.subplots()
        rax = None

    # Free theory etaR = eta, i.e. the diagonal the curves approach for Re -> 0.
    ax.plot(inv_re, inv_re, "--", color="black", alpha=0.7)
    ax.plot(inv_re, curve_1loop, color="tab:green",
            label=r"1-loop") # ($\mathrm{Re}^{-1} + c_d\,\mathrm{Re}$) $\eta_R = \eta + c_3 T\rho\Lambda/\eta$
    ax.plot(inv_re, curve_inf, color="tab:orange",
            label=r"RG improved 1-loop") # $\eta_R = \sqrt{\eta^2 + 2c_3 T\rho\Lambda}$
    if len(inv_re_table):
        # Straight segments between the tabulated eta, without markers.
        ax.plot(inv_re_table, inv_reR_table, "-", color="tab:red",
                label=r"RG improved 1-loop (self-consistent)")
    # One legend entry and color per scan, i.e. per lattice and cutoff.
    keys = [scan_key(f) for f in fits]
    scans = []
    for key in dict.fromkeys(keys):
        idx = np.array([k == key for k in keys])
        color = SCAN_COLORS[len(scans) % len(SCAN_COLORS)]
        scans.append((idx, color))
        ax.errorbar(inv_re_data[idx], inv_reR_data[idx], yerr=inv_reR_err[idx],
                    fmt="o", capsize=3, color=color,
                    label=scan_label(fits[keys.index(key)]))
    if rax is not None:
        # Both the point and its error bar are divided by the same reference,
        # so the panel is the plot above rescaled point by point: the
        # deviation is in units of the rFRG at that 1/Re.
        ref = reference_inv_reR(inv_re_data, coeff_inf, inv_re_table,
                                inv_reR_table)
        for idx, color in scans:
            rax.errorbar(inv_re_data[idx], inv_reR_data[idx]/ref[idx] - 1.0,
                         yerr=inv_reR_err[idx]/ref[idx], fmt="o", capsize=3,
                         color=color)
        missing = np.isnan(ref)
        if missing.any():
            print("warning: no deviation drawn at 1/Re = "
                  + ", ".join(f"{v:g}" for v in inv_re_data[missing])
                  + f"; {args.etaR_table} tabulates no etaR there")
        rax.axhline(0.0, color="black", linewidth=0.6)
        rax.set_xlabel(inv_re_label)
        rax.set_ylabel(r"$\eta_R/\eta_R^{\mathrm{RG}} - 1$", fontsize=7)
        # Few ticks: the panel is a fifth of the height of the plot above and
        # its labels would otherwise run into each other. (Only --rlim gets
        # this far; the centred limits below place their own two ticks.)
        rax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    else:
        ax.set_xlabel(inv_re_label)
    ax.set_ylabel(inv_reR_label)
    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.xlog:
        ax.set_xscale('log')
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    if args.ylog:
        ax.set_yscale('log')
    if rax is not None:
        if args.rlim is not None:
            rax.set_ylim(*args.rlim)
        else:
            plot_style.center_panel(rax)
    ax.legend()
    if rax is not None:
        # The layout of a figure with a panel is the shared one: a square plot
        # over the panel, the gap of the style file between them, and the
        # ticks of both tidied once they have their final size.
        plot_style.finish_panel_figure(fig, ax, rax)
    else:
        fig.tight_layout(pad=0.2)

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
