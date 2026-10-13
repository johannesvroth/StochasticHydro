#!/usr/bin/env python3
r"""Plot the viscosity ratio etaR/eta against k/Lambda.

For every input directory etaR is measured by the standard procedure in
etaR_fit.py: the constant -dampR = -etaR (1 + eta_reg_uv) k_hat^2/rho is
fitted, unweighted, to the per-run logarithmic derivative of the time
correlator over the window [tau, 2 tau]. The plotted error bar is the
statistical error (run-to-run scatter) and the systematic one (the shift when
the same window is moved half a tau later) added in quadrature. Everything is
drawn as the ratio to the bare eta.

The mode index nk and the cutoff Lam are parsed from the directory name, so
each directory contributes one point, at the momentum k = 2 pi nk/N of its
mode in units of the cutoff, e.g.

    plot_eta_ratio_vs_k.py \
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk1 \
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk2 \
        ...

All directories are one scan: they share the lattice size, eta, Lam and the
trailing "x" of a --no-ideal-step run, and differ only in nk (and possibly in
dt, which a scan may lower for the faster modes).

The expectation etaR = sqrt(eta^2 + 2 c_d T rho Lam^(d-2)) has no
momentum dependence at all, so it is a horizontal line, and the bare eta is a
second horizontal line, at 1 on this axis: any k dependence of the measured
points is either the true momentum dependence of etaR or a lattice artifact
of the mode. The damping rate the fit
measures goes with the lattice momentum k_hat = 2 sin(pi nk/N) rather than
with k, and the two part company towards the edge of the Brillouin zone; k is
what the plot shows, because it is the momentum the theory curves are a
function of.

Momentum-dependent theory curves are drawn with --curves, each named in the
legend by the matching entry of --curve-labels and optionally colored by the
matching entry of --curve-colors, e.g.

    --curves out-d3eta0.1Lam0.4mode2 out-d3eta0.1Lam0.4mode0 \
    --curve-labels "1-loop" "RG improved 1-loop (self-consistent)" \
    --curve-colors tab:green tab:orange

Each directory holds the tabulated etaR(k) in etaIR.dat and names the
dimension, bare eta and cutoff it was computed at, which is the only place
they are recorded. The k of a curve is divided by its own cutoff, so a curve
computed at another Lam still lands where it belongs on the same
dimensionless axis. From back to front the plot holds the bare eta, the
momentum-independent expectation, the curves in the order they are given and
the measured points. The tables span decades in k while the lattice
offers a handful of modes, so they are kept out of the autoscaling: the view
still follows the measured points unless --xlim/--ylim say otherwise."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_style
import etaR_fit

plot_style.use()

AUTO_OUTPUT = Path("<auto>")

# Below this many sample points in the fit window the fitted constant is not
# meaningfully constrained: the point is still drawn, but flagged. A mode whose
# autocorrelation time tau falls below the sampling interval of the correlator
# has no resolved decay left to fit, which is what happens at large nk unless
# the run wrote its correlator often enough.
MIN_FIT_POINTS = 3

# Turns a correlator directory name into the run parameters the output file
# is named after: avg-jp-time-corr-Nx64Ny64Nz64dt10eta0.05Lam0.2nk3 becomes
# Nx64Ny64Nz64dt10eta0.05Lam0.2, with the "x" of a --no-ideal-step run kept.
OUTPUT_STEM_RE = re.compile(r"^avg-jp-time-corr-|nk\d+(x?)$")

# A theory curve directory is named like out-d3eta0.1Lam0.4mode0 and holds the
# tabulated etaR(k) in etaIR.dat; its parameters appear nowhere else. What the
# curve is is the caller's business and named only in its legend label.
CURVE_TABLE = "etaIR.dat"
CURVE_NAME_RE = re.compile(rf"d(?P<dim>\d+)eta(?P<eta>{etaR_fit.FLOAT})"
                          rf"Lam(?P<lam>{etaR_fit.FLOAT})")


def read_curve(curve_dir: Path):
    """Read the tabulated etaR(k) of a theory curve directory, returning
    (k, etaR, params) with k sorted.

    The directory is named out-d3eta0.1Lam0.4mode0 and holds the two-column
    "# k etaR" file etaIR.dat. The dimension, the bare eta and the cutoff of
    the curve are taken from that name: the table itself records none of them,
    and the curve has to be divided by the eta it was computed at, not by the
    one of the scan."""
    path = curve_dir / CURVE_TABLE
    if not path.exists():
        raise SystemExit(f"No {CURVE_TABLE} in {curve_dir}")
    m = CURVE_NAME_RE.search(curve_dir.name)
    if m is None:
        raise SystemExit(f"Cannot parse curve directory name '{curve_dir.name}': "
                         "expected the form out-d3eta0.1Lam0.4mode0")
    params = {"dim": int(m["dim"]), "eta": float(m["eta"]),
              "lam": float(m["lam"])}
    k, etaR = np.loadtxt(path, unpack=True)
    order = np.argsort(k)
    return k[order], etaR[order], params


def check_curve_params(params, ref, curve_dir: Path) -> None:
    """Report every parameter the curve was computed at that the scan does not
    share: the curve is then a different theory rather than its prediction,
    which is easy to miss once it is drawn next to the points."""
    for name, value, field in (("d", params["dim"], "dim"),
                               ("eta", params["eta"], "eta"),
                               ("Lam", params["lam"], "lam")):
        if value != getattr(ref, field):
            print(f"warning: {curve_dir} was computed at {name} = {value:g}, the "
                  f"scan at {getattr(ref, field):g}")


def draw_curve(ax, curve_dir: Path, ref, color, label: str) -> None:
    """Draw one tabulated curve as etaR/eta against k/Lam, in its own color.

    Both the ratio and the axis are taken relative to the parameters of the
    curve itself rather than of the scan, so a curve computed at another eta
    or Lam is still drawn where it belongs. The curve is added as a bare artist,
    which leaves the data limits alone: the table spans decades in k, and
    letting it into the autoscaling would squeeze the measured points into a
    corner."""
    k, etaR, params = read_curve(curve_dir)
    check_curve_params(params, ref, curve_dir)
    ax.add_artist(Line2D(k/params["lam"], etaR/params["eta"], color=color,
                         label=label))


def scan_key(fit):
    """Everything a point of the scan shares with all the others: all the
    directory name carries but nk and dt."""
    return (fit.nx, fit.ny, fit.nz, fit.eta, fit.lam, fit.suffix)


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


def scan_label(fit) -> str:
    r"""Legend entry for the measured points, e.g. "$32^3$, $a\Lambda = 0.4$".
    The bare eta is left out: the whole plot is drawn as etaR/eta, so its
    value says nothing about where the points sit."""
    label = rf"{lattice_label(fit.nx, fit.ny, fit.nz)}, $a\Lambda = {fit.lam:g}$"
    if fit.suffix == "x":
        label += ", no ideal step"
    return label


def x_values(fits):
    """The x coordinate of each fit: the momentum k = 2 pi nk/N of its mode
    in units of that run's cutoff Lam, which is dimensionless and therefore
    comparable across lattice sizes and cutoffs."""
    return np.array([2*np.pi*f.nk/(f.nx*f.lam) for f in fits])


def fit_all(input_dirs, args):
    """Fit every directory, printing one summary line each.

    A scan over nk runs into modes whose decay the correlator does not resolve:
    once the autocorrelation time drops below the sampling interval, the fit
    window holds no point at which every run is still positive and etaR_fit
    gives up. Such a directory is reported and skipped rather than taking the
    whole plot down with it, and one that is only barely resolved is drawn with
    a warning."""
    fits = []
    for input_dir in input_dirs:
        try:
            fit = etaR_fit.fit_etaR(input_dir, mass_density=args.mass_density,
                                    temp=args.temp, fit_tmin=args.fit_tmin,
                                    fit_tmax=args.fit_tmax,
                                    use_theory_tau=args.tau_theory)
        except SystemExit as exc:
            print(f"skipping {Path(input_dir).name}: {exc}")
            continue
        print(fit.summary(with_ratio=True))
        n_points = int(fit.fit_sel.sum())
        if n_points < MIN_FIT_POINTS:
            print(f"  warning: only {n_points} point(s) of the correlator lie "
                  f"in the fit window [{fit.fit_tmin:g}, {fit.fit_tmax:g}]; "
                  f"this mode decays within about {fit.tau:g}, which the "
                  f"correlator samples every {fit.time_diff[1]:g}")
        fits.append(fit)
    if not fits:
        raise SystemExit("No directory could be fitted")
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dirs", type=Path, nargs="+",
                        help="One or more directories containing the *.dat "
                             "correlator files written by "
                             "compute_avg_jp_time_correlator.py, named like "
                             "avg-jp-time-corr-Nx8Ny8Nz8dt0.1eta0.1Lam1.25nk7 "
                             "(nk, eta and Lam are parsed from the name); each "
                             "contributes one point")
    etaR_fit.add_fit_arguments(parser)
    parser.add_argument("--curves", type=Path, nargs="+", default=[],
                        metavar="DIR",
                        help="Theory curves to draw: directories named like "
                             "out-d3eta0.1Lam0.4mode0, each holding the "
                             f"two-column '# k etaR' file {CURVE_TABLE}. "
                             "They are drawn against k/Lam of their own "
                             "cutoff and do not take part in the autoscaling")
    parser.add_argument("--curve-labels", nargs="+", default=[],
                        metavar="NAME",
                        help="Legend entries of the --curves, one each, in "
                             "the same order")
    parser.add_argument("--curve-colors", nargs="+", default=None,
                        metavar="COLOR",
                        help="Colors of the --curves, one each, in the same "
                             "order, as any matplotlib color (tab:green, "
                             "#ff0000, ...); without it they take the default "
                             "color cycle after the color of the measured "
                             "points")
    parser.add_argument("--no-theory", dest="theory", action="store_false",
                        help="Do not draw the momentum-independent "
                             "expectation and the bare eta")
    parser.add_argument("-o", "--output", type=Path, nargs="?", default=None,
                        const=AUTO_OUTPUT,
                        help="Save the plot to this file instead of showing "
                             "it; without an argument, save to "
                             "figs/eta-ratio-vs-k-<run parameters>.pdf, the "
                             "parameters being those of the first input "
                             "directory without its nk")
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
    if len(args.curve_labels) != len(args.curves):
        parser.error(f"{len(args.curves)} --curves but "
                     f"{len(args.curve_labels)} --curve-labels")
    if args.curve_colors is None:
        # The measured points take C0, so the curves start at the next color.
        args.curve_colors = [f"C{i + 1}" for i in range(len(args.curves))]
    elif len(args.curve_colors) != len(args.curves):
        parser.error(f"{len(args.curves)} --curves but "
                     f"{len(args.curve_colors)} --curve-colors")

    if args.output is AUTO_OUTPUT:
        # The run parameters go into the file name verbatim from the first
        # directory, minus its nk: this plot scans the modes, so the mode
        # index is the one thing the name must not carry. A trailing "x" is
        # kept, a --no-ideal-step scan is a different figure.
        stem = OUTPUT_STEM_RE.sub(r"\1", args.input_dirs[0].name)
        args.output = Path("figs") / f"eta-ratio-vs-k-{stem}.pdf"
        args.output.parent.mkdir(exist_ok=True)

    fits = fit_all(args.input_dirs, args)
    if len({scan_key(fit) for fit in fits}) > 1:
        raise SystemExit("The directories are not one scan: they must agree in "
                         "everything but nk and dt")
    fits.sort(key=lambda f: f.nk)
    ref = fits[0]
    # The whole plot is divided by the bare eta, so the bare line sits at 1
    # and the expectation at etaR/eta.
    x = x_values(fits)
    y = np.array([f.ratio for f in fits])
    yerr = np.array([f.ratio_err for f in fits])

    fig, ax = plt.subplots()

    # Back to front, which is also the order of the legend: the bare eta, the
    # momentum-independent expectation etaR = sqrt(eta^2 + 2 c_d T rho
    # Lam^(d-2)) (in 2D no L=infinity value exists and it is left out), the
    # theory curves in the order given and, on top, the measured points.
    if args.theory:
        ax.axhline(1.0, color="black", ls="--", alpha=0.7, linewidth=1.2)
        # if ref.etaR_inf is not None:
        #     ax.axhline(ref.etaR_inf/ref.eta, color="tab:red", ls="--",
        #                linewidth=1.2,
        #                label=r"$\eta_R = \sqrt{\eta^2 + 2c_d T\rho\Lambda}$")
    for curve_dir, label, color in zip(args.curves, args.curve_labels,
                                       args.curve_colors):
        draw_curve(ax, curve_dir, ref, color, label)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color="C0",
                label=scan_label(ref))

    ax.set_xlabel(r"$p/\Lambda$")
    ax.set_ylabel(r"$\eta_R/\eta$")
    if args.xlog:
        ax.set_xscale('log')
    if args.ylog:
        ax.set_yscale('log')

    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    ax.legend()
    fig.tight_layout(pad=0.2)

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
