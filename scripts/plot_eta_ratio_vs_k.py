#!/usr/bin/env python3
r"""Plot the viscosity ratio etaR/eta against k/Lambda.

For every input directory etaR is measured by the standard procedure in
etaR_fit.py: the constant -dampR = -etaR (1 + eta_reg_uv) k_hat^2/rho is
fitted, unweighted, to the per-run logarithmic derivative of the time
correlator over the window [tau, 2 tau]. The plotted error bar is the
statistical error (run-to-run scatter) and the systematic one (the shift when
the same window is moved half a tau later) added in quadrature. Everything is
drawn as the ratio to the bare eta of its own run, which is what makes series
at different bare viscosities comparable at all.

The mode index nk and the cutoff Lam are parsed from the directory name, so
each directory contributes one point, at the momentum k = 2 pi nk/N of its
mode in units of the cutoff, e.g.

    plot_eta_ratio_vs_k.py \
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk1 \
        avg-jp-time-corr-Nx32Ny32Nz32dt10eta0.0707107Lam0.4nk2 \
        ...

Directories that differ in anything but nk (lattice size, dt, eta, Lam, or the
trailing "x" of a --no-ideal-step run) are drawn as separate series, each with
its own color and its own theory band, and the legend names only the
parameters that actually differ between them.

The rFRG expectation etaR = sqrt(eta^2 + 2 c_d T rho Lam^(d-2)) has no
momentum dependence at all, so it is a horizontal line (with a band spanning
the two finite-volume coefficients) and the bare eta is a second horizontal
line, at 1 on this axis: any k dependence of the measured points is either the true momentum
dependence of etaR or a lattice artifact of the mode. The damping rate the fit
measures goes with the lattice momentum k_hat = 2 sin(pi nk/N) rather than
with k, and the two part company towards the edge of the Brillouin zone; k is
what the plot shows, because it is the momentum the rFRG flow is a function
of.

The momentum dependence the flow itself predicts is drawn on top with
--rFRG out-d3eta0.1Lam0.4mode0, and the one-loop truncation of the same flow
with --PT out-d3eta0.1Lam0.4mode2: each directory holds the tabulated etaR(k)
in etaIR.dat and names the flow's dimension, bare eta and cutoff, which is the
only place they are recorded. The k of a curve is divided by the cutoff of the
flow it came from, so a flow run at another Lam still lands where it belongs
on the same dimensionless axis. The tables span decades in k while the lattice
offers a handful of modes, so they are kept out of the autoscaling: the view
still follows the measured points unless --xlim/--ylim say otherwise."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# A flow directory is named out-d3eta0.1Lam0.4mode0, with the trailing mode
# selecting the truncation it was run in, and holds the tabulated etaR(k) in
# etaIR.dat; the flow's parameters appear nowhere else. Which truncation a
# directory holds is the caller's business: --rFRG and --PT differ only in the
# curve they draw, not in how the directory is read.
# Turns a correlator directory name into the run parameters the output file
# is named after: avg-jp-time-corr-Nx64Ny64Nz64dt10eta0.05Lam0.2nk3 becomes
# Nx64Ny64Nz64dt10eta0.05Lam0.2, with the "x" of a --no-ideal-step run kept.
OUTPUT_STEM_RE = re.compile(r"^avg-jp-time-corr-|nk\d+(x?)$")

FLOW_TABLE = "etaIR.dat"
FLOW_NAME_RE = re.compile(rf"d(?P<dim>\d+)eta(?P<eta>{etaR_fit.FLOAT})"
                          rf"Lam(?P<lam>{etaR_fit.FLOAT})")

# What identifies one series: everything a directory name carries except nk.
# Directories agreeing in all of these are points of the same curve.
GROUP_FIELDS = ("lattice", "dt", "eta", "lam", "suffix")


def read_flow(flow_dir: Path):
    """Read the tabulated etaR(k) of a flow directory, returning
    (k, etaR, params) with k sorted.

    The directory is named out-d3eta0.1Lam0.4mode0 and holds the two-column
    "# k etaR" file etaIR.dat. The dimension, the bare eta and the cutoff of
    the flow are taken from that name: the table itself records none of them,
    and the curve has to be divided by the eta the flow was run at, not by the
    one of any series."""
    flow_dir = Path(flow_dir)
    path = flow_dir / FLOW_TABLE
    if not path.exists():
        raise SystemExit(f"No {FLOW_TABLE} in {flow_dir}")
    m = FLOW_NAME_RE.search(flow_dir.name)
    if m is None:
        raise SystemExit(f"Cannot parse flow directory name '{flow_dir.name}': "
                         "expected the form out-d3eta0.1Lam0.4mode0")
    params = {"dim": int(m["dim"]), "eta": float(m["eta"]),
              "lam": float(m["lam"])}
    k, etaR = np.loadtxt(path, unpack=True)
    order = np.argsort(k)
    return k[order], etaR[order], params


def check_flow_params(params, fits, flow_dir: Path) -> None:
    """Report every parameter the flow was run at that no plotted series
    shares: the curve is then a different theory rather than their
    prediction, which is easy to miss once it is drawn next to the points."""
    for name, value, field in (("d", params["dim"], "dim"),
                               ("eta", params["eta"], "eta"),
                               ("Lam", params["lam"], "lam")):
        values = {getattr(fit, field) for fit in fits}
        if value not in values:
            print(f"warning: {flow_dir} was run at {name} = {value:g}, the "
                  "plotted series at "
                  + ", ".join(f"{v:g}" for v in sorted(values)))


def flow_x(k, params):
    """Place the momenta of a flow on the x axis: its own k in units of the
    cutoff it was run at."""
    return k/params["lam"]


def draw_flow(ax, flow_dir: Path, fits, color: str, label: str,
              style: str = "-") -> None:
    """Draw one tabulated flow as etaR/eta against k/Lam, in its own color.

    Both the ratio and the axis are taken relative to the parameters of the
    flow itself rather than of any measured series, so a flow run at another
    eta or Lam is still drawn where it belongs."""
    k, etaR, params = read_flow(flow_dir)
    check_flow_params(params, fits, flow_dir)
    ax.plot(flow_x(k, params), etaR/params["eta"], style, color=color,
            label=label)


def group_value(fit, field):
    if field == "lattice":
        return (fit.nx, fit.ny, fit.nz)
    return getattr(fit, field)


def group_key(fit):
    return tuple(group_value(fit, f) for f in GROUP_FIELDS)


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


def field_label(fit, field: str) -> str:
    if field == "lattice":
        return lattice_label(fit.nx, fit.ny, fit.nz)
    if field == "dt":
        return f"dt = {fit.dt:g}"
    if field == "eta":
        return rf"$\eta = {fit.eta:g}$"
    if field == "lam":
        return rf"$\Lambda = {fit.lam:g}$"
    if field == "suffix":
        return "no ideal step" if fit.suffix == "x" else "ideal step"
    raise ValueError(f"unknown field {field}")


def series_label(fit, fields) -> str:
    """Legend entry naming `fields` of this series, e.g.
    "$32^3$, dt = 10, $\\eta = 0.0707107$, $\\Lambda = 0.4$"."""
    return ", ".join(field_label(fit, f) for f in fields)


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
                                    fit_tmax=args.fit_tmax)
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
    parser.add_argument("--rFRG", dest="rfrg", type=Path, default=None,
                        metavar="DIR",
                        help="Draw the tabulated etaR(k) of an rFRG flow on "
                             "top: a directory named like "
                             "out-d3eta0.1Lam0.4mode0 holding the two-column "
                             f"'# k etaR' file {FLOW_TABLE}. It is drawn "
                             "against k/Lam of its own cutoff and does not "
                             "take part in the autoscaling")
    parser.add_argument("--PT", dest="pt", type=Path, default=None,
                        metavar="DIR",
                        help="The same for the one-loop truncation of the "
                             "flow, a directory named like "
                             "out-d3eta0.1Lam0.4mode2, drawn as one-loop "
                             "perturbation theory")
    parser.add_argument("--no-theory", dest="theory", action="store_false",
                        help="Do not draw the rFRG expectation and the bare "
                             "eta")
    parser.add_argument("-o", "--output", type=Path, nargs="?", default=None,
                        const=AUTO_OUTPUT,
                        help="Save the plot to this file instead of showing "
                             "it; without an argument, save to "
                             "figs/eta-ratio-vs-k.pdf")
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
        args.output = Path("figs") / "eta-ratio-vs-k.pdf"
        args.output.parent.mkdir(exist_ok=True)

    fits = fit_all(args.input_dirs, args)

    # One series per set of directories agreeing in everything but nk, in the
    # order their first directory appeared on the command line.
    groups = {}
    for fit in fits:
        groups.setdefault(group_key(fit), []).append(fit)
    # Name only what actually differs between the series; with a single series
    # there is nothing to distinguish, so it carries its full parameters.
    if len(groups) > 1:
        varying = [f for f in GROUP_FIELDS
                   if len({group_value(fit, f) for fit in fits}) > 1]
    else:
        varying = [f for f in GROUP_FIELDS if f != "suffix"]
        if fits[0].suffix == "x":
            varying.append("suffix")

    group_list = list(groups.values())
    for members in group_list:
        members.sort(key=lambda f: f.nk)
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    colors = [colors[i % len(colors)] for i in range(len(group_list))]

    fig, ax = plt.subplots()

    # eta and Lam are fixed within a series, so the rFRG expectation
    # etaR = sqrt(eta^2 + 2 c_d T rho Lam^(d-2)) is a single number per series:
    # a horizontal line at the L=infinity coefficient and a band between the
    # two finite-volume ones (in 2D no L=infinity value exists and only the
    # band is drawn). It has no momentum dependence whatsoever, which is the
    # point of comparison here. When every series shares (eta, Lam) there is
    # only one expectation, drawn neutrally so it is not mistaken for one
    # series' own; otherwise each gets its own in its color. Either way the
    # lines go down first, so the measured points sit on top of them.
    shared_theory = len({(f.eta, f.lam, f.dim) for f in fits}) == 1
    if args.theory:
        for i, members in enumerate(group_list):
            ref = members[0]
            color = "black" if shared_theory else colors[i]
            first = i == 0
            # The whole plot is divided by the series' own bare eta, so the
            # bare line sits at 1 and the expectation at etaR/eta.
            scale = ref.eta
            if ref.etaR_inf is not None:
                ax.axhline(ref.etaR_inf/scale, color=color, linewidth=0.8,
                           label=r"rFRG $\sqrt{\eta^2 + 2c_d T\rho\Lambda}$, "
                                 if first else None)
            # ax.axhspan(min(ref.etaR1, ref.etaR2)/scale,
            #            max(ref.etaR1, ref.etaR2)/scale,
            #            color=color, alpha=0.3,
            #            label=r"rFRG $\sqrt{\eta^2 + 2c_d T\rho\Lambda}$"
            #                  if first else None)
            ax.axhline(ref.eta/scale, color=color, ls="--", alpha=0.7,
                       linewidth=0.8,
                       label=r"bare $\eta$" if first else None)
            if shared_theory:
                break

    for members, color in zip(group_list, colors):
        x = x_values(members)
        y = np.array([f.ratio for f in members])
        yerr = np.array([f.ratio_err for f in members])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=color,
                    label=series_label(members[0], varying)
                          if len(group_list) > 1 or varying else "simulation")

    ax.set_xlabel(r"$k/\Lambda$")
    ax.set_ylabel(r"$\eta_R/\eta$")
    if args.xlog:
        ax.set_xscale('log')
    if args.ylog:
        ax.set_yscale('log')

    # The tabulated curve runs over decades in k, orders of magnitude past the
    # few modes the lattice has, so letting it into the autoscaling would
    # squeeze the measured points into a corner. The view is fixed to those
    # points first and restored afterwards; an explicit --xlim/--ylim below
    # still moves it, which is how the rest of the curve is looked at.
    if args.rfrg is not None or args.pt is not None:
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # Perturbation theory goes down first, so the rFRG flow it is meant
        # to be compared against is drawn over it rather than hidden beneath
        # it; the same green and dashes the other eta-ratio plots give the
        # one-loop expression they draw analytically.
        if args.pt is not None:
            draw_flow(ax, args.pt, fits, "tab:green",
                      "1-loop perturbation theory", style="--")
        if args.rfrg is not None:
            draw_flow(ax, args.rfrg, fits, "tab:red",
                      r"rFRG flow, self-consistent $k$-dep.")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

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
