#!/usr/bin/env python3
"""Central definition of the standard etaR measurement.

The renormalized shear viscosity etaR is measured from the time correlator
f(t) = Re sum_T jp(t) jp(t+T) by fitting the constant

    -dampR = -etaR (1 + eta_reg_uv(k_hat^2))/rho k_hat^2

to the per-run logarithmic derivative d ln f(t)/dt. For an exponential decay
f(t) ~ exp(-Gamma t) the log-derivative is the constant -Gamma, so the
amplitude of the correlator drops out entirely and etaR is the only fit
parameter.

The fit window is [tau, 2 tau]: it starts past the early non-exponential
transient and stops before the correlator decays into the noise, where f'/f
develops poles. The window scale tau is not taken from theory but from the
data, through the self-consistency condition

    tau = 1/dampR(tau),

with dampR(tau) the damping rate fitted over [tau, 2 tau]: tau is varied
upward from zero until the correlation time the fit yields coincides with the
tau that defined the window, and the first crossing is taken. At small tau
the window still sits in the transient, where f'/f is small, so the fitted
correlation time is longer than the window and tau dampR(tau) < 1; it grows
past 1 once the window reaches the exponential plateau. --tau-theory falls
back to the old window scale, the theory autocorrelation time
1/(etaR_tau damp_per_eta) with etaR_tau the L=infinity (3D) or mean
finite-size (2D) renormalized viscosity, which is now only a reference value.

The fit is unweighted, so the window bounds, not the error bars, decide what
enters, which makes the extent of the window the dominant systematic. It is
estimated by running the whole procedure a second time over the same window
shifted half a tau later, [1.5 tau, 2.5 tau], with the same condition
tau = 1/dampR solved over that window. The difference of the two results is
the systematic. Each pass carries its own statistical error, and the second
one is reported alongside its result. The second pass enters the error only:
the quoted etaR is always the one from [tau, 2 tau]. It reaches further into
the data, so it can fail the sample-count requirement where the first pass
succeeded; there is then no systematic, and the summary says so. The
statistical error is the run-to-run scatter, not the covariance of a single
least-squares fit: neighboring points of the correlator are strongly
correlated, so a fit that assumes independent points vastly underestimates the
error, while the runs themselves are independent. The scatter is taken over
the per-run solutions of the self-consistency condition -- each run gets its
own tau and its own etaR = 1/(tau damp_per_eta) -- so that the uncertainty of
tau is part of the error rather than held fixed across the runs; with
--tau-theory or a hand-picked window, where there is no condition to solve, it
is the scatter of the per-run fits over the common window instead.

Every plot script that measures etaR calls `fit_etaR` from here; do not copy
the procedure into the scripts again. `add_fit_arguments` gives them the
matching command line options, so --mass-density, --temp, --fit-tmin and
--fit-tmax mean the same thing everywhere.

Note on Lam: etaR = sqrt(eta^2 + 2 c_d T rho Lam) is the 3D form, and it is
used here in 2D as well, exactly as the scripts did before this module
existed. In 2D the general expression carries Lam^(d-2) = 1 instead; that
discrepancy predates this file and is deliberately left alone here rather
than changed silently.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, curve_fit

from rfrg_coefficients import coefficients

# Default fit window, in units of the theory autocorrelation time tau, and the
# window it is compared against for the systematic error: the same width,
# shifted half a tau later. The shifted window enters the error only, never
# the central value.
TMIN_FACTOR = 1.0
TMAX_FACTOR = 2.0
SYST_TMIN_FACTOR = 1.5
SYST_TMAX_FACTOR = 2.5

# The self-consistent window scale is found by scanning tau over a geometric
# grid of SELF_CONSISTENT_SCAN_POINTS values, starting at the shortest window
# that still holds SELF_CONSISTENT_MIN_POINTS samples, and refining the first
# bracketed crossing of tau dampR(tau) = 1 by bisection. The window only
# changes when one of its bounds crosses a sample of the correlator, so the
# root is located to a fraction SELF_CONSISTENT_XTOL_FACTOR of the sampling
# interval and no further.
SELF_CONSISTENT_MIN_POINTS = 4
SELF_CONSISTENT_SCAN_POINTS = 64
SELF_CONSISTENT_XTOL_FACTOR = 0.1

FLOAT = r"[0-9.]+(?:[eE][+-]?[0-9]+)?"
OBS_NAME_RE = re.compile(
    rf"avg-jp-time-corr-Nx(?P<nx>\d+)Ny(?P<ny>\d+)Nz(?P<nz>\d+)"
    rf"dt(?P<dt>{FLOAT})eta(?P<eta>{FLOAT})Lam(?P<lam>{FLOAT})nk(?P<nk>\d+)(?P<suffix>x?)$"
)


def parse_obs_name(input_dir: Path) -> dict:
    """Parse the lattice size, dt, eta, Lam and nk out of a directory name of
    the form avg-jp-time-corr-Nx8Ny8Nz8dt0.1eta0.1Lam1.25nk7."""
    m = OBS_NAME_RE.match(Path(input_dir).name)
    if m is None:
        raise SystemExit(f"Cannot parse input directory name '{Path(input_dir).name}': "
                         "expected the form avg-jp-time-corr-Nx8Ny8Nz8dt0.1eta0.1Lam1.25nk7")
    return {"nx": int(m["nx"]), "ny": int(m["ny"]), "nz": int(m["nz"]),
            "dt": float(m["dt"]), "eta": float(m["eta"]), "lam": float(m["lam"]),
            "nk": int(m["nk"]), "suffix": m["suffix"]}


def read_header(path: Path) -> dict:
    """Read the '# nx=.. ny=.. ..' header line of a correlator file."""
    with open(path) as f:
        first_line = f.readline().lstrip("#").strip()
    return {key: float(value) for key, value in
            (item.split("=") for item in first_line.split())}


def eta_reg_uv(k2, lam):
    """The UV regulator (k^2/Lam^2 - 1) above the cutoff, 0 below it."""
    x = k2/(lam*lam)
    return x - 1 if x > 1 else 0.0


def load_correlators(input_dir: Path):
    """Read every *.dat correlator in `input_dir` and return
    (time_diff, correlators, meta), with correlators of shape (n_runs, n) the
    real parts truncated to the shortest run."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.dat"))
    if not paths:
        raise SystemExit(f"No correlator files found in {input_dir}")

    meta = read_header(paths[0])
    time_diff = None
    correlators = []
    for path in paths:
        t, re_part, im_part = np.loadtxt(path, unpack=True)
        if time_diff is None:
            time_diff = t
        correlators.append(re_part)

    n_min = min(len(c) for c in correlators)
    return time_diff[:n_min], np.stack([c[:n_min] for c in correlators]), meta


def fit_log_derivative(time_diff, log_derivatives, positive, damp_per_eta,
                       eta, tmin, tmax, name=""):
    """Fit the constant -etaR damp_per_eta to the per-run log-derivatives over
    [tmin, tmax] and return (etaR, stat, sel, n_dropped).

    The fit is unweighted: every point in the window enters with the same
    weight. Points where any run has f <= 0 are dropped, f'/f has a pole
    there, and their number is returned.

    Neighboring points of the correlator are strongly correlated, so the
    covariance a single least-squares fit reports (which assumes independent
    points) vastly underestimates the error on etaR. The runs, in contrast,
    are independent, so each is fitted separately and the scatter of the
    fitted values is taken as the error. This is done for the error only: the
    model is linear in etaR, so the mean of the per-run fits equals the fit of
    the mean curve exactly."""
    n_runs = len(log_derivatives)
    sel = (time_diff >= tmin) & (time_diff <= tmax)
    n_dropped = int(np.sum(sel & ~positive))
    sel = sel & positive
    if not sel.any():
        raise SystemExit(f"No usable fit points in {name}: the window "
                         f"[{tmin:g}, {tmax:g}] contains no point at which "
                         "all runs are positive")

    model = lambda t, etaR: np.full_like(t, -etaR * damp_per_eta)
    p0 = (eta,)
    fits = np.array([curve_fit(model, time_diff[sel], run_logd[sel],
                               p0=p0, maxfev=10000)[0][0]
                     for run_logd in log_derivatives])
    if n_runs > 1:
        return fits.mean(), fits.std(ddof=1)/np.sqrt(n_runs), sel, n_dropped
    popt, pcov = curve_fit(model, time_diff[sel], log_derivatives[0][sel],
                           p0=p0, maxfev=10000)
    return popt[0], np.sqrt(pcov[0, 0]), sel, n_dropped


def usable_tau_range(time_diff, positive, tmin_factor=TMIN_FACTOR,
                     tmax_factor=TMAX_FACTOR,
                     min_points: int = SELF_CONSISTENT_MIN_POINTS):
    """Smallest and largest window scale tau for which
    [tmin_factor tau, tmax_factor tau] is a usable fit window: wide enough to
    hold `min_points` samples of the correlator, and early enough that no run
    has crossed zero inside it, where f'/f has a pole.

    Only the contiguous positive stretch at the start of the correlator
    counts: once a run has decayed into the noise and crossed zero, the
    points beyond that crossing are noise whether they are positive or not.

    The window spans (tmax_factor - tmin_factor) tau in time, so on the
    uniform grid of the correlator it holds that over dt, plus one, samples;
    the shortest one still holding `min_points` of them follows."""
    dt_grid = time_diff[1] - time_diff[0]
    crossings = np.flatnonzero(~positive)
    t_end = (time_diff[crossings[0] - 1] if crossings.size and crossings[0] > 0
             else time_diff[-1])
    return ((min_points - 1)*dt_grid/(tmax_factor - tmin_factor),
            t_end/tmax_factor)


def solve_self_consistent_tau(time_diff, log_derivatives, positive,
                              damp_per_eta, eta, tmin_factor=TMIN_FACTOR,
                              tmax_factor=TMAX_FACTOR, name=""):
    """Solve tau = 1/dampR(tau) for the correlation time tau, with dampR(tau)
    the damping rate fitted over [tmin_factor tau, tmax_factor tau], and
    return tau.

    The condition is always the same -- the correlation time that scales the
    fit window is the one the fit over that window yields -- and the factors
    say where the window sits relative to it. With the default factors the
    window is [tau, 2 tau], starting at the correlation time; with [1.5, 2.5]
    it is the same window half a tau later, which is how the systematic is
    estimated.

    tau is varied upward from the shortest usable window and the first
    crossing is taken, as the condition has one: tau dampR(tau) - 1 starts out
    negative, because a window inside the early transient sees a
    log-derivative smaller in magnitude than the asymptotic decay rate and so
    fits a correlation time longer than tau, and turns positive once the
    window has moved onto the exponential plateau, where dampR(tau) stops
    changing while tau keeps growing. The crossing is bracketed on a geometric
    grid and refined by bisection.

    The fitted rate is a step function of tau -- the window changes only when
    one of its bounds passes a sample -- so the root is located to a fraction
    of the sampling interval, no finer.

    Windows quoted in the failure messages are in units of the bare
    correlation time of the mode, tau_bare = rho/(eta (1+eta_reg_uv) k_hat^2),
    the scale the plots use, so that they can be read off the axis directly;
    the same numbers in t, which is what --fit-tmin/--fit-tmax take, follow in
    parentheses."""
    tau_min, tau_max = usable_tau_range(time_diff, positive, tmin_factor,
                                        tmax_factor)
    # The bare correlation time of the mode, the unit the windows are quoted
    # in below and the one the plots scale their time axis by.
    tau_bare = 1.0/(eta*damp_per_eta)
    dt_grid = time_diff[1] - time_diff[0]
    factors = f"[{tmin_factor:g}, {tmax_factor:g}] tau"

    def window(tau):
        """One window as the messages quote it:
        '[a, b] tau_bare (t = [c, d])'."""
        lo, hi = tmin_factor*tau, tmax_factor*tau
        return (f"[{lo/tau_bare:g}, {hi/tau_bare:g}] tau_bare "
                f"(t = [{lo:g}, {hi:g}])")

    if not tau_min < tau_max:
        raise SystemExit(
            f"No usable fit window in {name}: the correlator stays positive "
            f"only out to {tmax_factor*tau_max/tau_bare:g} tau_bare "
            f"(t = {tmax_factor*tau_max:g}), which does not hold a window "
            f"{factors} of {SELF_CONSISTENT_MIN_POINTS} samples -- the "
            f"shortest one is {window(tau_min)}")

    def mismatch(tau):
        """tau dampR(tau) - 1: negative while the window still sits in the
        transient, positive once it has reached the exponential plateau."""
        etaR = fit_log_derivative(time_diff, log_derivatives, positive,
                                  damp_per_eta, eta, tmin_factor*tau,
                                  tmax_factor*tau, name=name)[0]
        return tau*etaR*damp_per_eta - 1.0

    taus = np.geomspace(tau_min, tau_max, SELF_CONSISTENT_SCAN_POINTS)
    mismatches = np.array([mismatch(tau) for tau in taus])
    brackets = np.flatnonzero((mismatches[:-1] < 0) & (mismatches[1:] >= 0))
    if not brackets.size:
        if mismatches[0] >= 0:
            raise SystemExit(
                f"No self-consistent fit window in {name}: already the "
                f"shortest usable window {window(tau_min)} fits a "
                "correlation time shorter than itself, so the self-consistent "
                "tau lies below the sampling interval of the correlator "
                f"({dt_grid/tau_bare:g} tau_bare, t = {dt_grid:g}); sample "
                "the correlator more finely, or set the window by hand with "
                "--fit-tmin/--fit-tmax")
        raise SystemExit(
            f"No self-consistent fit window in {name}: out to the longest "
            f"usable window {window(tau_max)} the fitted "
            "correlation time stays longer than the window, so the correlator "
            "has not decayed within the data; run longer, or set the window "
            "by hand with --fit-tmin/--fit-tmax")
    lo, hi = taus[brackets[0]], taus[brackets[0] + 1]
    # The bounds move by tmax_factor dtau when tau moves by dtau, so this is
    # the step in tau that shifts the window by a fraction of a sample.
    xtol = SELF_CONSISTENT_XTOL_FACTOR*dt_grid/tmax_factor
    return brentq(mismatch, lo, hi, xtol=xtol)


def solve_self_consistent_etaR_per_run(time_diff, log_derivatives, positive,
                                      damp_per_eta, eta,
                                      tmin_factor=TMIN_FACTOR,
                                      tmax_factor=TMAX_FACTOR, name=""):
    """Solve the self-consistency condition for each run on its own and return
    (etaR_per_run, n_unsolved): the per-run viscosities
    etaR = 1/(tau damp_per_eta), with NaN where a run has no solution, and how
    many those are.

    This is what the statistical error is taken from. Fitting the runs over
    one common window would hold tau fixed and so leave the uncertainty of tau
    itself out of the scatter; solving the condition run by run puts it in,
    since a run that decays a little faster picks a correspondingly earlier
    window.

    The usable range of tau stays the common one, from the points at which
    every run is still positive, so that the runs are compared over the same
    stretch of data and the scatter measures their decay rates rather than
    how far each of them happens to be resolvable. A run whose mismatch never
    crosses zero inside that range contributes no value and is counted
    instead."""
    etaRs = []
    n_unsolved = 0
    for run_logd in log_derivatives:
        try:
            tau = solve_self_consistent_tau(
                time_diff, run_logd[None, :], positive, damp_per_eta, eta,
                tmin_factor, tmax_factor, name=name)
        except SystemExit:
            etaRs.append(np.nan)
            n_unsolved += 1
            continue
        etaRs.append(1.0/(tau*damp_per_eta))
    return np.array(etaRs), n_unsolved


def self_consistent_fit(time_diff, log_derivatives, positive, damp_per_eta,
                        eta, tmin_factor, tmax_factor, name=""):
    """One complete self-consistent measurement over the window
    [tmin_factor tau, tmax_factor tau]: solve the condition on all runs
    together for the window, fit etaR over it, and take the statistical error
    from the per-run solutions of the same condition.

    Returns (etaR, stat, tmin, tmax, fit_sel, n_pole, etaR_per_run,
    n_unsolved). The whole procedure is run twice with different factors, and
    the difference of the two results is the systematic."""
    tau = solve_self_consistent_tau(time_diff, log_derivatives, positive,
                                    damp_per_eta, eta, tmin_factor,
                                    tmax_factor, name=name)
    tmin, tmax = tmin_factor*tau, tmax_factor*tau
    etaR, stat, fit_sel, n_pole = fit_log_derivative(
        time_diff, log_derivatives, positive, damp_per_eta, eta, tmin, tmax,
        name=name)

    etaR_per_run = None
    n_unsolved = 0
    if len(log_derivatives) > 1:
        etaR_per_run, n_unsolved = solve_self_consistent_etaR_per_run(
            time_diff, log_derivatives, positive, damp_per_eta, eta,
            tmin_factor, tmax_factor, name=name)
        n_solved = int(np.count_nonzero(~np.isnan(etaR_per_run)))
        if n_solved > 1:
            stat = float(np.nanstd(etaR_per_run, ddof=1)/np.sqrt(n_solved))
    return (etaR, stat, tmin, tmax, fit_sel, n_pole, etaR_per_run,
            n_unsolved)


@dataclass
class EtaRFit:
    """One directory's etaR measurement, with everything the plot scripts need
    to draw it: the parsed run parameters, the loaded correlators and their
    log-derivatives, the theory reference values and the fit result."""

    name: str
    path: Path
    # Parsed from the directory name.
    nx: int
    ny: int
    nz: int
    dt: float
    eta: float
    lam: float
    nk: int
    suffix: str
    dim: int
    # Physical parameters the fit was run with.
    mass_density: float
    temp: float
    # Data.
    time_diff: np.ndarray
    correlators: np.ndarray
    log_derivatives: np.ndarray
    logd_mean: np.ndarray
    logd_sem: np.ndarray
    positive: np.ndarray
    # Mode and theory reference values.
    k_hat: float
    damp_per_eta: float
    tau_bare: float
    etaR1: float
    etaR2: float
    etaR_inf: float
    etaR_tau: float
    # Window scale actually used, the theory value it is compared against,
    # and which of the two `tau` is.
    tau: float
    tau_theory: float
    self_consistent: bool
    # Fit result.
    fit_tmin: float
    fit_tmax: float
    fit_sel: np.ndarray
    n_pole: int
    etaR: float
    # Per-run solutions of the self-consistency condition, the scatter the
    # statistical error is taken from (None where the window did not come
    # from the condition), and how many runs had no solution of their own.
    etaR_per_run: np.ndarray
    n_unsolved: int
    stat: float
    syst: float
    # The second pass: same procedure over [SYST_TMIN_FACTOR, SYST_TMAX_FACTOR]
    # tau, its own statistical error and its own per-run solutions. Their
    # difference from the first pass is the systematic. None where there is no
    # second pass: a hand-picked window, or one the condition could not solve.
    etaR_alt: float
    stat_alt: float
    fit_tmin_alt: float
    fit_tmax_alt: float
    etaR_alt_per_run: np.ndarray
    n_unsolved_alt: int
    chi2_dof: float

    @property
    def n_runs(self) -> int:
        return len(self.correlators)

    @property
    def err(self) -> float:
        """Total error: statistical and systematic added in quadrature."""
        return float(np.hypot(self.stat, self.syst))

    @property
    def n_solved(self) -> int:
        """Runs that solved the self-consistency condition on their own."""
        if self.etaR_per_run is None:
            return 0
        return int(np.count_nonzero(~np.isnan(self.etaR_per_run)))

    @property
    def stat_from_per_run(self) -> bool:
        """Whether the statistical error is the scatter of the per-run
        self-consistent solutions rather than of fits over one common
        window."""
        return self.n_solved > 1


    @property
    def ratio(self) -> float:
        """etaR/eta."""
        return self.etaR/self.eta

    @property
    def ratio_err(self) -> float:
        return self.err/self.eta

    @property
    def scale(self) -> float:
        """sqrt(rho T Lam), the viscosity scale the Reynolds numbers use."""
        return float(np.sqrt(self.mass_density*self.temp*self.lam))

    @property
    def reynolds(self) -> float:
        """Bare Reynolds number Re = sqrt(rho T Lam)/eta."""
        return self.scale/self.eta

    @property
    def inv_reynolds(self) -> float:
        """1/Re = eta/sqrt(rho T Lam)."""
        return self.eta/self.scale

    @property
    def inv_reynolds_r(self) -> float:
        """1/Re_R = etaR/sqrt(rho T Lam)."""
        return self.etaR/self.scale

    @property
    def inv_reynolds_r_err(self) -> float:
        return self.err/self.scale

    @property
    def dampR(self) -> float:
        return self.etaR*self.damp_per_eta

    @property
    def dampR1(self) -> float:
        return self.etaR1*self.damp_per_eta

    @property
    def dampR2(self) -> float:
        return self.etaR2*self.damp_per_eta

    def summary(self, with_ratio: bool = False) -> str:
        """The one-line report the scripts print for each directory."""
        if self.etaR_alt is None:
            syst_note = (" (no syst: the second pass has no self-consistent "
                         "window)" if self.self_consistent
                         else " (no syst: explicit fit window)")
            alt_note = ""
        else:
            syst_note = f" +- {self.syst:g} (syst)"
            alt_note = (f", [{SYST_TMIN_FACTOR:g}, {SYST_TMAX_FACTOR:g}] tau "
                        f"over [{self.fit_tmin_alt:g}, {self.fit_tmax_alt:g}] "
                        f"gives {self.etaR_alt:g} +- {self.stat_alt:g}")
        pole_note = (f", dropped {self.n_pole} points with f <= 0"
                     if self.n_pole else "")
        stat_label = "stat, per-run tau" if self.stat_from_per_run else "stat"
        unsolved_note = "".join(
            f", {n} of {self.n_runs} runs without a self-consistent tau of "
            f"their own in the {which} pass"
            for n, which in ((self.n_unsolved, "first"),
                             (self.n_unsolved_alt, "second")) if n)
        chi2_note = ("" if self.chi2_dof is None
                     else f", chi2/dof = {self.chi2_dof:.2f}")
        tau_note = (f", tau = {self.tau:g} "
                    f"({'self-consistent' if self.self_consistent else 'theory'}"
                    f", theory {self.tau_theory:g})")
        line = (f"{self.name}: etaR = {self.etaR:g} +- {self.stat:g} "
                f"({stat_label}){syst_note} "
                f"(bare eta = {self.eta:g}, fit window "
                f"[{self.fit_tmin:g}, {self.fit_tmax:g}]"
                f"{tau_note}{alt_note}{pole_note}{unsolved_note}){chi2_note}")
        if with_ratio:
            line += f"; etaR/eta = {self.ratio:g} +- {self.ratio_err:g}"
        return line


def fit_etaR(input_dir: Path, mass_density: float = 1.0, temp: float = 1.0,
             fit_tmin: float = None, fit_tmax: float = None,
             use_theory_tau: bool = False) -> EtaRFit:
    """Measure etaR in one avg-jp-time-corr directory by the standard
    procedure described in this module's docstring.

    `fit_tmin`/`fit_tmax` override the default window in units of t. Giving
    either one replaces the central window, and the systematic -- the shift of
    the result over the standard window moved half a tau later -- is then not
    comparable to it, so it is reported as 0 and `etaR_alt` is None. It also
    leaves nothing for the self-consistency condition to fix, so the window
    scale reported as `tau` is the theory one.

    `use_theory_tau` scales the default window by the theory autocorrelation
    time instead of solving the self-consistency condition, i.e. it restores
    the procedure used before that condition replaced it."""
    input_dir = Path(input_dir)
    obs = parse_obs_name(input_dir)
    eta, lam = obs["eta"], obs["lam"]

    time_diff, correlators, meta = load_correlators(input_dir)
    n_runs = len(correlators)

    # Per-run log-derivative f'/f, then mean and SEM across runs.
    log_derivatives = np.gradient(correlators, time_diff, axis=1) / correlators
    logd_mean = log_derivatives.mean(axis=0)
    logd_sem = (log_derivatives.std(axis=0, ddof=1)/np.sqrt(n_runs)
                if n_runs > 1 else np.zeros(len(time_diff)))

    k_hat = 2.0*np.sin(2*np.pi*meta["nk"]/meta["nx"]*0.5)
    damp_per_eta = (1 + eta_reg_uv(k_hat**2, lam))/mass_density * k_hat**2
    # The bare correlation time of the mode, used by the scripts to scale the
    # time axis so that the bare decay reaches 1/e at t/tau_bare = 1.
    tau_bare = 1.0/(eta*damp_per_eta)

    dim = 2 if obs["nz"] == 1 else 3
    coeff1, coeff2, coeff_inf = coefficients(dim)
    etaR1 = np.sqrt(eta**2 + 2.0*coeff1*temp*mass_density*lam)
    etaR2 = np.sqrt(eta**2 + 2.0*coeff2*temp*mass_density*lam)
    # Reference viscosity for the theory autocorrelation time tau that sets
    # the default fit window: the L=infinity value in 3D, the mean of the two
    # finite-size values in 2D, where no L=infinity coefficient exists.
    if coeff_inf is not None:
        etaR_inf = np.sqrt(eta**2 + 2.0*coeff_inf*temp*mass_density*lam)
        etaR_tau = etaR_inf
    else:
        etaR_inf = None
        etaR_tau = 0.5*(etaR1 + etaR2)
    tau_theory = 1.0/(etaR_tau*damp_per_eta)

    # f'/f has a pole wherever a run's correlator crosses zero, so keep only
    # points at which every run is still positive.
    positive = np.all(correlators > 0, axis=0)

    # The measurement. With the self-consistent window it is made twice, over
    # [TMIN_FACTOR, TMAX_FACTOR] tau and over [SYST_TMIN_FACTOR,
    # SYST_TMAX_FACTOR] tau, each pass solving the condition tau = 1/dampR
    # over its own window and each carrying its own statistical error from the
    # per-run solutions. The second window is the first one moved half a tau
    # later, so the difference of the two results is what the placement of the
    # window is worth: that is the systematic. The first pass is the quoted
    # result.
    #
    # With --tau-theory or a hand-picked window there is no condition to
    # solve: the window is placed from theory and the second pass is the same
    # window moved half a tau later, as it was before the condition existed.
    etaR_per_run = None
    n_unsolved = 0
    etaR_alt_per_run = None
    n_unsolved_alt = 0
    if use_theory_tau or fit_tmin is not None or fit_tmax is not None:
        tau = tau_theory
        self_consistent = False
        tmin = fit_tmin if fit_tmin is not None else TMIN_FACTOR*tau
        tmax = fit_tmax if fit_tmax is not None else TMAX_FACTOR*tau
        etaR, stat, fit_sel, n_pole = fit_log_derivative(
            time_diff, log_derivatives, positive, damp_per_eta, eta, tmin,
            tmax, name=input_dir.name)
        if fit_tmin is None and fit_tmax is None:
            tmin_alt = SYST_TMIN_FACTOR*tau
            tmax_alt = SYST_TMAX_FACTOR*tau
            etaR_alt, stat_alt = fit_log_derivative(
                time_diff, log_derivatives, positive, damp_per_eta, eta,
                tmin_alt, tmax_alt, name=input_dir.name)[:2]
            syst = abs(etaR - etaR_alt)
        else:
            etaR_alt = None
            stat_alt = None
            tmin_alt = tmax_alt = None
            syst = 0.0
    else:
        self_consistent = True
        (etaR, stat, tmin, tmax, fit_sel, n_pole, etaR_per_run,
         n_unsolved) = self_consistent_fit(
            time_diff, log_derivatives, positive, damp_per_eta, eta,
            TMIN_FACTOR, TMAX_FACTOR, name=input_dir.name)
        tau = tmin/TMIN_FACTOR
        # The second pass, over the same window half a tau later. It reaches
        # further into the data, so it can fail where the first succeeded --
        # there is then nothing to compare against and no systematic, which
        # the summary says.
        try:
            (etaR_alt, stat_alt, tmin_alt, tmax_alt, _, _, etaR_alt_per_run,
             n_unsolved_alt) = self_consistent_fit(
                time_diff, log_derivatives, positive, damp_per_eta, eta,
                SYST_TMIN_FACTOR, SYST_TMAX_FACTOR, name=input_dir.name)
            syst = abs(etaR - etaR_alt)
        except SystemExit:
            etaR_alt = None
            stat_alt = None
            tmin_alt = tmax_alt = None
            syst = 0.0

    # chi^2/dof of the mean log-derivative against the fitted constant, using
    # the SEM error bars; clip exact zeros to keep 1/sigma finite. The window
    # holds only a few tens of points whose residuals are strongly correlated,
    # so chi2/dof scatters much more widely than 1 +- sqrt(2/dof) even when
    # the constant describes the data well: it is reported, not used as a cut.
    if n_runs > 1:
        sigma = np.maximum(logd_sem, logd_sem[logd_sem > 0].min())[fit_sel]
        resid = (logd_mean[fit_sel] + etaR*damp_per_eta)/sigma
        chi2_dof = float(np.sum(resid**2)/(fit_sel.sum() - 1))
    else:
        chi2_dof = None

    return EtaRFit(
        name=input_dir.name, path=input_dir,
        nx=obs["nx"], ny=obs["ny"], nz=obs["nz"], dt=obs["dt"], eta=eta,
        lam=lam, nk=obs["nk"], suffix=obs["suffix"], dim=dim,
        mass_density=mass_density, temp=temp,
        time_diff=time_diff, correlators=correlators,
        log_derivatives=log_derivatives, logd_mean=logd_mean,
        logd_sem=logd_sem, positive=positive,
        k_hat=k_hat, damp_per_eta=damp_per_eta, tau_bare=tau_bare,
        etaR1=etaR1, etaR2=etaR2, etaR_inf=etaR_inf, etaR_tau=etaR_tau,
        tau=tau, tau_theory=tau_theory, self_consistent=self_consistent,
        fit_tmin=tmin, fit_tmax=tmax, fit_sel=fit_sel, n_pole=n_pole,
        etaR=etaR, etaR_per_run=etaR_per_run,
        n_unsolved=n_unsolved, stat=stat, syst=syst, etaR_alt=etaR_alt,
        stat_alt=stat_alt, fit_tmin_alt=tmin_alt, fit_tmax_alt=tmax_alt,
        etaR_alt_per_run=etaR_alt_per_run, n_unsolved_alt=n_unsolved_alt,
        chi2_dof=chi2_dof)


def add_fit_arguments(parser) -> None:
    """Add the options every etaR fit shares, so they mean the same thing in
    every script: --mass-density, --temp, --fit-tmin and --fit-tmax."""
    parser.add_argument("--mass-density", type=float, default=1.0,
                        help="Mass density")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--tau-theory", action="store_true",
                        help="Scale the default fit window by the theory "
                             "autocorrelation time instead of solving the "
                             "self-consistency condition tau = 1/dampR(tau) "
                             "for it")
    parser.add_argument("--fit-tmin", type=float, default=None,
                        help="Only fit data with t >= this value, to cut away "
                             f"the early non-exponential transient (default: "
                             f"{TMIN_FACTOR:g} tau, with tau the "
                             "self-consistent window scale, or the theory "
                             "autocorrelation time with --tau-theory)")
    parser.add_argument("--fit-tmax", type=float, default=None,
                        help="Only fit data with t <= this value (default: "
                             f"{TMAX_FACTOR:g} tau, before the correlator "
                             "decays into the noise, where f'/f develops "
                             "poles). Giving either bound explicitly "
                             "suppresses the systematic error, which is "
                             "otherwise the shift of the result over the "
                             f"window [{SYST_TMIN_FACTOR:g}, "
                             f"{SYST_TMAX_FACTOR:g}] tau, and leaves the "
                             "self-consistency condition nothing to fix")


def fit_dirs(input_dirs, args, verbose: bool = True, with_ratio: bool = False):
    """Fit every directory in `input_dirs` with the options `add_fit_arguments`
    put on `args`, printing one summary line each."""
    fits = []
    for input_dir in input_dirs:
        fit = fit_etaR(input_dir, mass_density=args.mass_density,
                       temp=args.temp, fit_tmin=args.fit_tmin,
                       fit_tmax=args.fit_tmax,
                       use_theory_tau=getattr(args, "tau_theory", False))
        if verbose:
            print(fit.summary(with_ratio=with_ratio))
        fits.append(fit)
    return fits
