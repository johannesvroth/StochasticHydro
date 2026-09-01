#!/usr/bin/env python3
"""Central definition of the standard etaR measurement.

The renormalized shear viscosity etaR is measured from the time correlator
f(t) = Re sum_T jp(t) jp(t+T) by fitting the constant

    -dampR = -etaR (1 + eta_reg_uv(k_hat^2))/rho k_hat^2

to the per-run logarithmic derivative d ln f(t)/dt. For an exponential decay
f(t) ~ exp(-Gamma t) the log-derivative is the constant -Gamma, so the
amplitude of the correlator drops out entirely and etaR is the only fit
parameter.

The fit window is [tau, 2 tau] with tau = 1/dampR_tau the theory
autocorrelation time: it starts past the early non-exponential transient and
stops before the correlator decays into the noise, where f'/f develops poles.

The fit is unweighted, so the window bounds, not the error bars, decide what
enters, which makes the placement of the window the dominant systematic. It is
estimated by repeating the fit over the window shifted half a tau later,
[1.5 tau, 2.5 tau], and taking the difference. The shift enters the error
only: the quoted etaR is always the one fitted over [tau, 2 tau]. The
statistical error is the run-to-run scatter (SEM of the per-run fits), not the
covariance of a single least-squares fit: neighboring points of the correlator
are strongly correlated, so a fit that assumes independent points vastly
underestimates the error, while the runs themselves are independent.

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
from scipy.optimize import curve_fit

from rfrg_coefficients import coefficients

# Default fit window, in units of the theory autocorrelation time tau, and the
# window it is compared against for the systematic error: the same width,
# shifted half a tau later. The shifted window enters the error only, never
# the central value.
TMIN_FACTOR = 1.0
TMAX_FACTOR = 2.0
SYST_TMIN_FACTOR = 1.5
SYST_TMAX_FACTOR = 2.5

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
    tau: float
    # Fit result.
    fit_tmin: float
    fit_tmax: float
    fit_sel: np.ndarray
    n_pole: int
    etaR: float
    stat: float
    syst: float
    etaR_alt: float
    chi2_dof: float

    @property
    def n_runs(self) -> int:
        return len(self.correlators)

    @property
    def err(self) -> float:
        """Total error: statistical and systematic added in quadrature."""
        return float(np.hypot(self.stat, self.syst))

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
            syst_note = " (no syst: explicit fit window)"
            alt_note = ""
        else:
            syst_note = f" +- {self.syst:g} (syst)"
            alt_note = (f", [{SYST_TMIN_FACTOR:g}, {SYST_TMAX_FACTOR:g}] tau "
                        f"gives {self.etaR_alt:g}")
        pole_note = (f", dropped {self.n_pole} points with f <= 0"
                     if self.n_pole else "")
        chi2_note = ("" if self.chi2_dof is None
                     else f", chi2/dof = {self.chi2_dof:.2f}")
        line = (f"{self.name}: etaR = {self.etaR:g} +- {self.stat:g} (stat)"
                f"{syst_note} "
                f"(bare eta = {self.eta:g}, fit window "
                f"[{self.fit_tmin:g}, {self.fit_tmax:g}]"
                f"{alt_note}{pole_note}){chi2_note}")
        if with_ratio:
            line += f"; etaR/eta = {self.ratio:g} +- {self.ratio_err:g}"
        return line


def fit_etaR(input_dir: Path, mass_density: float = 1.0, temp: float = 1.0,
             fit_tmin: float = None, fit_tmax: float = None) -> EtaRFit:
    """Measure etaR in one avg-jp-time-corr directory by the standard
    procedure described in this module's docstring.

    `fit_tmin`/`fit_tmax` override the default window in units of t. Giving
    either one replaces the central window, and the systematic -- the shift of
    the result over the standard window moved half a tau later -- is then not
    comparable to it, so it is reported as 0 and `etaR_alt` is None."""
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
    tau = 1.0/(etaR_tau*damp_per_eta)

    # f'/f has a pole wherever a run's correlator crosses zero, so keep only
    # points at which every run is still positive.
    positive = np.all(correlators > 0, axis=0)

    tmin = fit_tmin if fit_tmin is not None else TMIN_FACTOR*tau
    tmax = fit_tmax if fit_tmax is not None else TMAX_FACTOR*tau
    etaR, stat, fit_sel, n_pole = fit_log_derivative(
        time_diff, log_derivatives, positive, damp_per_eta, eta, tmin, tmax,
        name=input_dir.name)

    # The systematic is the shift of the result when the whole window moves
    # half a tau later, not when it is extended: the central value above, from
    # [TMIN_FACTOR, TMAX_FACTOR] tau, is left untouched. With a hand-picked
    # window there is nothing to shift against, so none is formed.
    if fit_tmin is None and fit_tmax is None:
        etaR_alt = fit_log_derivative(
            time_diff, log_derivatives, positive, damp_per_eta, eta,
            SYST_TMIN_FACTOR*tau, SYST_TMAX_FACTOR*tau,
            name=input_dir.name)[0]
        syst = abs(etaR - etaR_alt)
    else:
        etaR_alt = None
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
        tau=tau,
        fit_tmin=tmin, fit_tmax=tmax, fit_sel=fit_sel, n_pole=n_pole,
        etaR=etaR, stat=stat, syst=syst, etaR_alt=etaR_alt,
        chi2_dof=chi2_dof)


def add_fit_arguments(parser) -> None:
    """Add the options every etaR fit shares, so they mean the same thing in
    every script: --mass-density, --temp, --fit-tmin and --fit-tmax."""
    parser.add_argument("--mass-density", type=float, default=1.0,
                        help="Mass density")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--fit-tmin", type=float, default=None,
                        help="Only fit data with t >= this value, to cut away "
                             f"the early non-exponential transient (default: "
                             f"{TMIN_FACTOR:g} tau, with tau the theory "
                             "autocorrelation time)")
    parser.add_argument("--fit-tmax", type=float, default=None,
                        help="Only fit data with t <= this value (default: "
                             f"{TMAX_FACTOR:g} tau, before the correlator "
                             "decays into the noise, where f'/f develops "
                             "poles). Giving either bound explicitly "
                             "suppresses the systematic error, which is "
                             "otherwise the shift of the result over the "
                             f"window [{SYST_TMIN_FACTOR:g}, "
                             f"{SYST_TMAX_FACTOR:g}] tau")


def fit_dirs(input_dirs, args, verbose: bool = True, with_ratio: bool = False):
    """Fit every directory in `input_dirs` with the options `add_fit_arguments`
    put on `args`, printing one summary line each."""
    fits = []
    for input_dir in input_dirs:
        fit = fit_etaR(input_dir, mass_density=args.mass_density,
                       temp=args.temp, fit_tmin=args.fit_tmin,
                       fit_tmax=args.fit_tmax)
        if verbose:
            print(fit.summary(with_ratio=with_ratio))
        fits.append(fit)
    return fits
