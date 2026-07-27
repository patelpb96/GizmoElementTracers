'''
MCMC parameter estimation for the Maoz SNe Ia (WDSN) delay-time-distribution
model, built on top of the GizmoElementTracers element-tracer forward model.

Overview
--------
The FIRE age-tracer ("element tracer") machinery maps a set of per-stellar-age-bin
mass weights into elemental abundances by integrating nucleosynthetic rate + yield
models across each age bin (see gizmo_agetracer.FIREYieldClass2 and
gizmo_agetracer.ElementAgeTracerClass).  The Maoz Ia rate model (used in FIRE-3.x)
has two physically interesting free parameters:

    n_ia  : normalization of the Ia rate            (gizmo_agetracer.NIA_DEFAULT = 2.6e-7)
    t_dd  : power-law exponent of the delay-time distribution
            (gizmo_agetracer.TDD_DEFAULT = -1.1)

This module lets you *infer* those two parameters with an affine-invariant MCMC
(emcee) from a stellar abundance data set in the [Mg/Fe]-[Fe/H] (or
[alpha/Fe]-[Fe/H]) plane.

The intended demo workflow (see mcmc_maoz_demo.py) is:

    1. Draw a set of reasonable, random age-tracer mass weights (one row of
       per-age-bin weights per mock star).
    2. Feed those weights through the element tracer module for a chosen set of
       "true" Maoz parameters to get the mean model abundances, then scatter each
       star by a Gaussian whose width is set by the observed standard deviations.
       This is the "fake"/mock data set.
    3. Run MCMC over (n_ia, t_dd) -- the same weights and element-tracer forward
       model are used to predict abundances for each trial parameter set -- and
       recover the input parameters.

For numerical convenience the (strictly positive, orders-of-magnitude-spanning)
normalization is sampled in log10 space, so the MCMC parameter vector is

    theta = (log10(n_ia), t_dd)

Units follow gizmo_model / gizmo_agetracer: ages in Myr, abundances as linear
mass fractions, and "metallicity" := log10(mass_fraction / mass_fraction_Solar),
so that [Fe/H] = metallicity.iron and [Mg/Fe] = metallicity.mg - metallicity.fe,
matching gizmo_io.ParticleDictionaryClass.prop('metallicity.agetracer...').
'''

import numpy as np

from . import gizmo_agetracer
from . import gizmo_model


# default "true"/fiducial Maoz parameters (from gizmo_agetracer)
NIA_DEFAULT = gizmo_agetracer.NIA_DEFAULT  # 2.6e-7
TDD_DEFAULT = gizmo_agetracer.TDD_DEFAULT  # -1.1

# elements that make up the "alpha" abundance, matching gizmo_io.prop('...alpha')
ALPHA_ELEMENTS = ['oxygen', 'magnesium', 'silicon', 'calcium']

# labels for the sampled parameters
PARAM_LABELS = [r'$\log_{10} n_{\mathrm{Ia}}$', r'$t_{\mathrm{dd}}$']

# default (broad but physical) flat-prior bounds on (log10 n_ia, t_dd)
DEFAULT_BOUNDS = np.array([[-7.5, -6.0], [-1.6, -0.5]])


def default_age_bins(age_bin_number=10, age_min=1.0, age_max=13700.0):
    '''
    Return age-tracer stellar age bin edges [Myr], equally spaced in log age.
    Has age_bin_number + 1 values (left edges plus the right edge of the last bin),
    matching the convention used throughout gizmo_agetracer.

    Parameters
    ----------
    age_bin_number : int
        number of age bins
    age_min, age_max : float
        minimum / maximum stellar age [Myr]
    '''
    return np.logspace(np.log10(age_min), np.log10(age_max), age_bin_number + 1)


def generate_agetracer_weights(
    n_star, n_age_bin, total_weight=1.0, older_bias=2.0, concentration=1.0, seed=None
):
    '''
    Generate a set of reasonable, random age-tracer mass weights.

    Each row is the per-age-bin mass deposited into one (mock) star or gas cell,
    as a fraction of the particle mass -- i.e. the quantity stored in
    part[species]['massfraction'][:, element.index.start:] that the element tracer
    module multiplies by the per-bin nucleosynthetic yields.  Weights are drawn
    from a Dirichlet-like distribution (per-star positive weights normalized to
    sum to total_weight), with an optional bias toward older age bins to mimic a
    star-formation history that peaks at early cosmic times.

    Parameters
    ----------
    n_star : int
        number of mock stars (rows)
    n_age_bin : int
        number of stellar age bins (columns); should equal len(age_bins) - 1
    total_weight : float
        sum of the weights for each star (mass fraction from all formation epochs);
        1.0 is a reasonable default
    older_bias : float
        ratio of the mean weight of the oldest bin to the youngest bin.  1.0 gives
        a flat expectation across bins; > 1 biases toward older (earlier) bins.
    concentration : float
        Dirichlet concentration.  Smaller -> more star-to-star scatter (spikier
        individual histories); larger -> smoother, more similar histories.
    seed : int or None
        seed for the random number generator (for reproducibility)

    Returns
    -------
    weights : 2-D array (n_star x n_age_bin)
        age-tracer mass weights
    '''
    rng = np.random.default_rng(seed)
    # per-bin mean weight profile: linear ramp from youngest to oldest bin
    profile = np.linspace(1.0, older_bias, n_age_bin)
    profile /= profile.sum()
    # Dirichlet with per-bin concentration proportional to the desired profile
    alpha = np.clip(profile * n_age_bin * concentration, 1e-3, None)
    weights = rng.dirichlet(alpha, size=n_star)
    return weights * total_weight


def _dirichlet_from_profile(profile, concentration, n_star, rng):
    '''Draw n_star Dirichlet weight rows whose expectation follows `profile`.'''
    profile = np.clip(np.asarray(profile, dtype=float), 1e-4, None)
    profile = profile / profile.sum()
    alpha = np.clip(profile * profile.size * concentration, 1e-3, None)
    return rng.dirichlet(alpha, size=n_star)


def generate_bimodal_weights(
    n_star,
    n_age_bin,
    high_alpha_frac=0.4,
    concentration=3.0,
    high_alpha_total=(0.85, 0.35),
    low_alpha_total=(2.4, 0.30),
    seed=None,
):
    '''
    Generate age-tracer mass weights for a *bimodal* stellar population, mimicking
    the Milky Way's two sequences in the [alpha/Fe]-[Fe/H] plane (the high-alpha
    "thick disk" and low-alpha "thin disk").

    The bimodality lives in the star-formation histories encoded by the weights
    (NOT in the global Maoz Ia parameters, which are shared by all stars).  Both
    sequences carry substantial weight in the old, delayed-Ia age bins, so *both*
    respond to the Maoz Ia parameters -- consistent with a gas-accretion origin of
    the alpha bimodality (e.g. a large influx of near-pristine gas; cf. dilution
    scenarios such as Barry et al. 2024) rather than an Ia-timing origin:

    * high-alpha ("thick disk"): a strong young/CCSN (alpha) contribution sits on
      top of a diluted, Ia-enriched base.  A large influx of near-pristine gas
      lowers the overall metallicity (lower total weight -> lower [Fe/H]) while
      continued prompt CCSN enrichment keeps [alpha/Fe] elevated.  It still holds
      real weight in the delayed-Ia bins, so it *does* shift with the Ia model.
    * low-alpha ("thin disk"): relatively more weight in the old (delayed-Ia) bins
      and higher total weight -> lower [alpha/Fe], higher [Fe/H]; most Ia-sensitive.

    Scatter in the per-star total weight spreads each population out along [Fe/H].

    Parameters
    ----------
    n_star : int
        total number of mock stars
    n_age_bin : int
        number of stellar age bins (columns); should equal len(age_bins) - 1
    high_alpha_frac : float
        fraction of stars in the high-alpha (thick-disk) sequence
    concentration : float
        Dirichlet concentration (star-to-star scatter in the age-bin shape)
    high_alpha_total, low_alpha_total : (float, float)
        (median, sigma_in_ln) of the log-normal per-star total weight for each
        sequence; the median controls where the sequence sits in [Fe/H]
    seed : int or None
        random seed

    Returns
    -------
    weights : 2-D array (n_star x n_age_bin)
        age-tracer mass weights
    labels : 1-D int array
        1 for high-alpha (thick-disk) stars, 0 for low-alpha (thin-disk) stars
    '''
    rng = np.random.default_rng(seed)
    n_high = int(round(high_alpha_frac * n_star))
    n_low = n_star - n_high

    x = np.linspace(0.0, 1.0, n_age_bin)  # 0 = youngest bin, 1 = oldest bin
    # high-alpha: strong young/CCSN (alpha) peak on top of a non-negligible old-bin
    # (delayed-Ia) floor, so the thick disk stays alpha-enhanced but remains
    # Ia-sensitive; low total weight (dilution) sets its lower [Fe/H]
    profile_high = np.exp(-((x - 0.15) ** 2) / (2 * 0.26 ** 2)) + 0.20
    # low-alpha: relatively more weight in the old (delayed-Ia) bins
    profile_low = 0.45 + 0.7 * np.exp(-((x - 0.6) ** 2) / (2 * 0.5 ** 2))

    w_high = _dirichlet_from_profile(profile_high, concentration, n_high, rng)
    w_high *= rng.lognormal(np.log(high_alpha_total[0]), high_alpha_total[1], (n_high, 1))

    w_low = _dirichlet_from_profile(profile_low, concentration, n_low, rng)
    w_low *= rng.lognormal(np.log(low_alpha_total[0]), low_alpha_total[1], (n_low, 1))

    weights = np.vstack([w_high, w_low])
    labels = np.concatenate([np.ones(n_high, dtype=int), np.zeros(n_low, dtype=int)])
    # shuffle so the two sequences are interleaved (order should not matter)
    order = rng.permutation(n_star)
    return weights[order], labels[order]


def massfractions_to_abundances(massfraction_dict, sun_massfraction, xfe='magnesium'):
    '''
    Convert a dictionary of linear elemental mass fractions into the abundance-ratio
    coordinates used for the fit: [Fe/H] and [X/Fe].

    Uses the same convention as gizmo_io: metallicity := log10(massfraction / solar),
    so [Fe/H] = metallicity.iron and [X/Fe] = metallicity.X - metallicity.iron.

    Parameters
    ----------
    massfraction_dict : dict
        keys are element names, values are 1-D arrays of linear mass fractions
    sun_massfraction : dict
        Solar mass fractions (e.g. gizmo_model.get_sun_massfraction())
    xfe : str
        numerator element for the [X/Fe] axis; either a single element name
        (e.g. 'magnesium') or 'alpha' for the mean over ALPHA_ELEMENTS

    Returns
    -------
    feh : 1-D array
        [Fe/H]
    xfe_ratio : 1-D array
        [X/Fe]
    '''
    with np.errstate(divide='ignore'):
        feh = np.log10(massfraction_dict['iron'] / sun_massfraction['iron'])
        if xfe == 'alpha':
            metallicities = [
                np.log10(massfraction_dict[e] / sun_massfraction[e]) for e in ALPHA_ELEMENTS
            ]
            x_over_h = np.mean(metallicities, axis=0)
        else:
            x_over_h = np.log10(massfraction_dict[xfe] / sun_massfraction[xfe])
    return feh, x_over_h - feh


# --------------------------------------------------------------------------------------------------
# Fast yield tabulation
#
# FIREYieldClass2.get_element_yields(continuous=True) calls scipy.integrate.quad once per element
# *and* per age bin, rebuilding feedback objects at every quadrature node -- the dominant cost of an
# MCMC step.  The per-bin yield factorizes, though: within a bin the integral is
#     yield[element][bin] = sum_channel  Y_channel[element] * INT_channel[bin],
# where INT_channel[bin] = integral over the bin of the channel's rate is *independent of element*.
# So we only need to integrate the (three) rate functions once each, on a shared grid, then combine
# with the per-element yields.  The integrator below works for *any* vectorized, non-negative rate
# function (not just Maoz/Mannucci), so new rate models -- e.g. the "kinked" Ia model below -- are
# just as fast.  It agrees with the quad path to ~1e-6 and is ~30x faster.
# --------------------------------------------------------------------------------------------------

# default FIRE-2.1 CCSN and wind rate-model breakpoints (Myr), matching gizmo_model
CC_TRANSITION_DEFAULT = (3.4, 10.37, 37.53)
CC_NORMALIZATION_DEFAULT = (0.0, 5.408e-4, 2.516e-4, 0.0)
WIND_TRANSITION_DEFAULT = (1.0, 3.5, 100.0)
IA_TRANSITION_DEFAULT = 37.53  # onset of the delayed Ia channel [Myr]


def _pow(t, exponent):
    '''t**exponent with t floored away from 0 (values below the onset are masked out anyway).'''
    return np.maximum(t, 1e-6) ** exponent


def ia_rate_maoz(t, n_ia, t_dd, t_ia=IA_TRANSITION_DEFAULT, ejecta=1.4):
    '''Maoz delayed Ia mass-loss rate: n_ia * (t/Gyr)^t_dd for t >= t_ia, times ejecta mass.'''
    t = np.asarray(t, dtype=float)
    r = ejecta * n_ia * _pow(t / 1e3, t_dd)
    return np.where(t >= t_ia, r, 0.0)


def ia_rate_mannucci(t, n_ia=1.0, t_dd=None, t_ia=IA_TRANSITION_DEFAULT, ejecta=1.4):
    '''Mannucci (FIRE-2) Ia rate: a prompt Gaussian bump; n_ia scales it (1 = original).'''
    t = np.asarray(t, dtype=float)
    r = ejecta * n_ia * (5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((t - 50.0) / 10.0) ** 2))
    return np.where(t >= t_ia, r, 0.0)


def ia_rate_kink(t, n_ia, t_dd, t_kink=200.0, t_dd2=-1.1, t_ia=IA_TRANSITION_DEFAULT, ejecta=1.4):
    '''
    "Kinked" Ia model: a broken power-law delay-time distribution -- like Maoz but with a
    modulatable kink at t_kink, where the slope changes from t_dd to t_dd2 (continuous in value).

    The kink *strength* is (t_dd2 - t_dd); t_kink modulates its *location*.  When t_dd2 == t_dd
    this reduces exactly to the plain Maoz power law (no kink).  It exists to exercise the fast
    integrator on an arbitrary, non-standard rate function.
    '''
    t = np.asarray(t, dtype=float)
    r_before = n_ia * _pow(t / 1e3, t_dd)
    r_after = n_ia * _pow(t_kink / 1e3, t_dd) * _pow(t / t_kink, t_dd2)
    r = ejecta * np.where(t < t_kink, r_before, r_after)
    return np.where(t >= t_ia, r, 0.0)


IA_RATE_MODELS = {'maoz': ia_rate_maoz, 'mannucci': ia_rate_mannucci, 'kink': ia_rate_kink}


def ccsn_rate(t, cc_normalization=CC_NORMALIZATION_DEFAULT, t_cc=CC_TRANSITION_DEFAULT, ejecta=10.5):
    '''FIRE-2.1 CCSN mass-loss rate: piecewise-constant across t_cc, times the CCSN ejecta mass.'''
    t = np.asarray(t, dtype=float)
    r = np.full(t.shape, cc_normalization[0], dtype=float)
    r = np.where((t > t_cc[0]) & (t <= t_cc[1]), cc_normalization[1], r)
    r = np.where((t > t_cc[1]) & (t <= t_cc[2]), cc_normalization[2], r)
    r = np.where(t > t_cc[2], cc_normalization[3], r)
    return ejecta * r


def wind_rate(t, t_w=WIND_TRANSITION_DEFAULT):
    '''FIRE-2.1 stellar-wind mass-loss rate (Solar metallicity), matching gizmo_model.'''
    t = np.asarray(t, dtype=float)
    tt = np.maximum(t, 1e-6)
    r = np.empty(t.shape, dtype=float)
    m1 = t <= t_w[0]
    m2 = (t > t_w[0]) & (t <= t_w[1])
    m3 = (t > t_w[1]) & (t <= t_w[2])
    m4 = t > t_w[2]
    r[m1] = 4.76317
    r[m2] = 4.76317 * tt[m2] ** (1.838 * 0.79)
    r[m3] = 29.4 * (tt[m3] / 3.5) ** -3.25 + 0.0041987
    r[m4] = 0.41987 * (tt[m4] / 1e3) ** -1.1 / (12.9 - np.log(tt[m4] / 1e3))
    return r / 1e3


def integrate_rate_over_bins(rate_fn, age_bins, breakpoints=(), n_grid=4000, age_min=0.0):
    '''
    Fast definite integral of an arbitrary vectorized rate function over each age bin.

    Builds one shared integration grid (dense, log-spaced) augmented with the exact bin edges and
    any model breakpoints (each tripled around a tiny neighborhood so steps/kinks are captured),
    evaluates rate_fn once on it, cumulatively trapezoid-integrates, and differences the cumulative
    integral at the bin edges.  Cost is O(n_grid), independent of the number of elements.

    Parameters
    ----------
    rate_fn : callable
        vectorized, non-negative rate as a function of stellar age [Myr]
    age_bins : array
        age bin edges [Myr] (len n_bin + 1)
    breakpoints : sequence
        ages [Myr] where the rate has a discontinuity or kink (e.g. transition times)
    n_grid : int
        number of log-spaced grid points
    age_min : float
        lower edge to use for the first bin (0 to match the age-tracer convention)

    Returns
    -------
    integrals : 1-D array (len n_bin)
        integral of rate_fn over each age bin
    '''
    edges = np.asarray(age_bins, dtype=float).copy()
    edges[0] = age_min
    a_max = edges[-1]
    grid = np.geomspace(1e-2, a_max, n_grid)
    extra = [age_min]
    for b in breakpoints:
        if 0 < b < a_max:
            extra += [b * (1 - 1e-9), b, b * (1 + 1e-9)]
    nodes = np.unique(np.concatenate([grid, edges, np.array(extra, dtype=float)]))
    r = rate_fn(nodes)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(nodes))])
    idx = np.searchsorted(nodes, edges)
    return np.diff(cumulative[idx])


def fast_element_yields(
    age_bins,
    element_names,
    ia_rate_fn=ia_rate_maoz,
    ia_kwargs=None,
    ia_breakpoints=(IA_TRANSITION_DEFAULT,),
    cc_normalization=CC_NORMALIZATION_DEFAULT,
    cc_transition=CC_TRANSITION_DEFAULT,
    wind_transition=WIND_TRANSITION_DEFAULT,
    ia_yield_source='ia',
    n_grid=4000,
):
    '''
    Tabulate per-age-bin nucleosynthetic yield mass fractions for the requested elements, using the
    fast factorized integrator.  Drop-in replacement for FIREYieldClass2.get_element_yields for the
    FIRE-2.1 rate model with an arbitrary Ia rate function.

    Returns a dict {element_name: 1-D array of per-bin yields}, matching the age-tracer convention.
    '''
    ia_kwargs = dict(ia_kwargs or {})
    y_ia = gizmo_model.nucleosyntheticYieldDict[ia_yield_source]
    y_cc = gizmo_model.nucleosyntheticYieldDict['cc']
    y_wind = gizmo_model.nucleosyntheticYieldDict['wind']

    int_ia = integrate_rate_over_bins(
        lambda t: ia_rate_fn(t, **ia_kwargs), age_bins, breakpoints=ia_breakpoints, n_grid=n_grid
    )
    int_cc = integrate_rate_over_bins(
        lambda t: ccsn_rate(t, cc_normalization, cc_transition), age_bins,
        breakpoints=cc_transition, n_grid=n_grid,
    )
    int_wind = integrate_rate_over_bins(
        lambda t: wind_rate(t, wind_transition), age_bins,
        breakpoints=wind_transition, n_grid=n_grid,
    )
    return {
        e: y_ia[e] * int_ia + y_cc[e] * int_cc + y_wind[e] * int_wind
        for e in element_names
    }


class MaozElementTracerModel:
    '''
    Forward model: (Maoz Ia parameters) -> ([Fe/H], [X/Fe]) for a fixed set of
    age-tracer mass weights, using the GizmoElementTracers element tracer module.

    For a trial parameter vector theta = (log10 n_ia, t_dd) it
        (1) builds a FIREYieldClass2 with those Maoz Ia parameters,
        (2) integrates the per-age-bin nucleosynthetic yields,
        (3) loads them into an ElementAgeTracerClass, and
        (4) applies the stored weights to get each star's elemental mass fractions,
    then converts to the [Fe/H]-[X/Fe] plane.
    '''

    def __init__(
        self,
        age_bins,
        weights,
        xfe='magnesium',
        model='fire2.1',
        ia_model='maoz',
        ia_transition_time=None,
        cc_normalization=None,
        cc_transition_time=None,
        wind_transition_time=None,
        initial_massfraction=None,
        fast=True,
        kink_params=None,
    ):
        '''
        Parameters
        ----------
        age_bins : array
            age-tracer stellar age bin edges [Myr] (len = n_age_bin + 1)
        weights : 2-D array (n_star x n_age_bin)
            age-tracer mass weights (e.g. from generate_agetracer_weights)
        xfe : str
            numerator for the [X/Fe] axis: an element name or 'alpha'
        model : str
            FIRE rate+yield model version (used by the slow FIREYieldClass2 path)
        ia_model : str
            Ia rate model: 'maoz' (default), 'mannucci', or 'kink' (broken power-law,
            see ia_rate_kink).  Only used by the fast path.
        ia_transition_time : list or None
            transition time(s) [Myr] for the Ia model (default [37.53])
        cc_normalization, cc_transition_time, wind_transition_time : sequence or None
            optionally override the CCSN / wind rate models (kept fixed during the fit)
        initial_massfraction : dict or None
            optional initial (pre-enrichment) linear mass fractions per element
        fast : bool
            if True (default), tabulate yields with the fast factorized integrator
            (fast_element_yields); if False, use FIREYieldClass2.get_element_yields (scipy.quad)
        kink_params : dict or None
            for ia_model='kink', the (fixed) shape of the kink, e.g.
            {'t_kink': 200.0, 't_dd2': -0.6}; n_ia and t_dd are still the sampled parameters
        '''
        self.age_bins = np.asarray(age_bins, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        assert self.weights.shape[1] == len(self.age_bins) - 1, (
            'weights have {} columns but there are {} age bins'.format(
                self.weights.shape[1], len(self.age_bins) - 1
            )
        )
        self.xfe = xfe
        self.model = model
        self.fast = fast
        self.ia_model = ia_model
        self.kink_params = dict(kink_params or {})
        self.sun_massfraction = gizmo_model.get_sun_massfraction()

        # elements we actually need to integrate (keep this minimal for speed)
        if xfe == 'alpha':
            needed = ['iron'] + list(ALPHA_ELEMENTS)
        else:
            needed = ['iron', xfe]
        # de-duplicate while preserving order
        self.element_names = list(dict.fromkeys(needed))

        # fixed rate-model settings (Ia transition, CCSN, winds)
        self.ia_transition = (
            ia_transition_time[0] if ia_transition_time is not None else IA_TRANSITION_DEFAULT
        )
        self.cc_normalization = (
            tuple(cc_normalization) if cc_normalization is not None else CC_NORMALIZATION_DEFAULT
        )
        self.cc_transition = (
            tuple(cc_transition_time) if cc_transition_time is not None else CC_TRANSITION_DEFAULT
        )
        self.wind_transition = (
            tuple(wind_transition_time) if wind_transition_time is not None
            else WIND_TRANSITION_DEFAULT
        )

        # knobs for the slow FIREYieldClass2 path (only used when fast=False)
        self._yield_kwargs = {'model': model, 'ia_type': ia_model}
        if ia_transition_time is not None:
            self._yield_kwargs['trans_time_ia'] = ia_transition_time
        if cc_normalization is not None:
            self._yield_kwargs['normalization_ccsn'] = cc_normalization
        if cc_transition_time is not None:
            self._yield_kwargs['trans_time_ccsn'] = cc_transition_time

        # set up the element tracer container once; only the yields change per eval
        self.tracer = gizmo_agetracer.ElementAgeTracerClass(element_index_start=0)
        self.tracer.assign_age_bins(age_bins=self.age_bins)
        self.initial_massfraction = initial_massfraction

    def yields(self, n_ia, t_dd):
        '''
        Integrate the per-age-bin nucleosynthetic yield mass fractions for the
        needed elements, for Ia parameters (n_ia, t_dd).

        Uses the fast factorized integrator (fast_element_yields) when self.fast, else
        FIREYieldClass2.get_element_yields.  Returns an element_yield_dict.
        '''
        if not self.fast:
            fyield = gizmo_agetracer.FIREYieldClass2(
                normalization_ia=n_ia, tdd_ia=t_dd, **self._yield_kwargs
            )
            return fyield.get_element_yields(
                self.age_bins, element_names=self.element_names, continuous=True
            )

        ia_rate_fn = IA_RATE_MODELS[self.ia_model]
        ia_kwargs = {'n_ia': n_ia, 't_dd': t_dd, 't_ia': self.ia_transition}
        ia_breakpoints = [self.ia_transition]
        if self.ia_model == 'kink':
            ia_kwargs.update(self.kink_params)
            ia_breakpoints.append(self.kink_params.get('t_kink', 200.0))
        ia_yield_source = 'mannucci' if self.ia_model == 'mannucci' else 'ia'

        return fast_element_yields(
            self.age_bins, self.element_names,
            ia_rate_fn=ia_rate_fn, ia_kwargs=ia_kwargs, ia_breakpoints=tuple(ia_breakpoints),
            cc_normalization=self.cc_normalization, cc_transition=self.cc_transition,
            wind_transition=self.wind_transition, ia_yield_source=ia_yield_source,
        )

    def massfractions(self, theta):
        '''
        Return a dict of linear elemental mass fractions (one array per element)
        for parameter vector theta = (log10 n_ia, t_dd), using the element tracer
        module to combine the per-bin yields with the stored weights.
        '''
        log10_n_ia, t_dd = theta
        n_ia = 10.0 ** log10_n_ia

        element_yield_dict = self.yields(n_ia, t_dd)
        # load the freshly integrated yields into the element tracer container
        self.tracer.assign_element_yield_massfractions(element_yield_dict, flush=True)
        if self.initial_massfraction is not None:
            self.tracer.assign_element_initial_massfraction(
                self.initial_massfraction, helium_massfraction=None
            )

        return {
            element_name: self.tracer.get_element_massfractions(element_name, self.weights)
            for element_name in self.element_names
        }

    def abundances(self, theta):
        '''
        Return ([Fe/H], [X/Fe]) arrays (one value per star) for parameter vector theta.
        '''
        return massfractions_to_abundances(
            self.massfractions(theta), self.sun_massfraction, self.xfe
        )

    def mean_abundances(self, theta):
        '''Return the population-mean (<[Fe/H]>, <[X/Fe]>) for parameter vector theta.'''
        feh, xfe = self.abundances(theta)
        return np.nanmean(feh), np.nanmean(xfe)


def generate_mock_data(model, true_theta, obs_std=(0.08, 0.06), seed=None, labels=None):
    '''
    Generate a "fake" abundance data set by running the forward model at the
    input "true" Maoz parameters and scattering each star with a Gaussian whose
    width is set by the observed standard deviations.

    Parameters
    ----------
    model : MaozElementTracerModel
        the forward model (carries the fixed age-tracer weights)
    true_theta : (float, float)
        true parameters (log10 n_ia, t_dd) used to generate the data
    obs_std : (float, float)
        observed standard deviations (sigma_[Fe/H], sigma_[X/Fe]) [dex] used both
        to scatter the mock data and as the measurement errors in the likelihood
    seed : int or None
        seed for the random number generator
    labels : array or None
        optional per-star population labels (e.g. from generate_bimodal_weights),
        carried through into data['label'] for plotting

    Returns
    -------
    data : dict
        keys: 'feh', 'xfe' (scattered mock observations), 'feh_err', 'xfe_err'
        (per-star errors), 'feh_true', 'xfe_true' (noise-free model values),
        'true_theta', 'obs_std', and 'label' if labels was given
    '''
    rng = np.random.default_rng(seed)
    feh_true, xfe_true = model.abundances(true_theta)
    sig_feh, sig_xfe = obs_std
    feh = feh_true + rng.normal(0.0, sig_feh, size=feh_true.shape)
    xfe = xfe_true + rng.normal(0.0, sig_xfe, size=xfe_true.shape)
    n_star = feh.size
    data = {
        'feh': feh,
        'xfe': xfe,
        'feh_err': np.full(n_star, sig_feh),
        'xfe_err': np.full(n_star, sig_xfe),
        'feh_true': feh_true,
        'xfe_true': xfe_true,
        'true_theta': np.asarray(true_theta, dtype=float),
        'obs_std': np.asarray(obs_std, dtype=float),
    }
    if labels is not None:
        data['label'] = np.asarray(labels)
    return data


def generate_gaussian_cloud(
    n_star, feh_mean, feh_std, xfe_mean, xfe_std, seed=None
):
    '''
    Generate a purely observational mock data set: a Gaussian cloud in the
    [X/Fe]-[Fe/H] plane about literature mean values with their standard
    deviations.  Useful for visualization / sanity checks (this does NOT come
    from the forward model, so a fit to it recovers whatever Maoz parameters best
    reproduce the cloud rather than a known input).

    Parameters
    ----------
    n_star : int
        number of mock stars
    feh_mean, feh_std : float
        mean and standard deviation of [Fe/H]
    xfe_mean, xfe_std : float
        mean and standard deviation of [X/Fe]
    seed : int or None
        random seed

    Returns
    -------
    data : dict
        keys 'feh', 'xfe', 'feh_err', 'xfe_err'
    '''
    rng = np.random.default_rng(seed)
    feh = rng.normal(feh_mean, feh_std, size=n_star)
    xfe = rng.normal(xfe_mean, xfe_std, size=n_star)
    return {
        'feh': feh,
        'xfe': xfe,
        'feh_err': np.full(n_star, feh_std),
        'xfe_err': np.full(n_star, xfe_std),
    }


# --------------------------------------------------------------------------------------------------
# Bayesian model: prior, likelihood, posterior
# --------------------------------------------------------------------------------------------------
def log_prior(theta, bounds=DEFAULT_BOUNDS):
    '''Flat (uniform) log prior on theta = (log10 n_ia, t_dd) within bounds.'''
    theta = np.asarray(theta)
    if np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]):
        return 0.0
    return -np.inf


def log_likelihood(theta, model, data):
    '''
    Gaussian log likelihood comparing the forward-model abundances to the mock
    data in both [Fe/H] and [X/Fe], summed over all stars.
    '''
    feh_model, xfe_model = model.abundances(theta)
    if not (np.all(np.isfinite(feh_model)) and np.all(np.isfinite(xfe_model))):
        return -np.inf
    resid_feh = (data['feh'] - feh_model) / data['feh_err']
    resid_xfe = (data['xfe'] - xfe_model) / data['xfe_err']
    chi2 = np.sum(resid_feh ** 2) + np.sum(resid_xfe ** 2)
    norm = np.sum(np.log(2 * np.pi * data['feh_err'] ** 2)) + np.sum(
        np.log(2 * np.pi * data['xfe_err'] ** 2)
    )
    return -0.5 * (chi2 + norm)


def log_probability(theta, model, data, bounds=DEFAULT_BOUNDS):
    '''Log posterior probability = log prior + log likelihood.'''
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, data)


# --------------------------------------------------------------------------------------------------
# Summary-statistic inference: match a dual-Gaussian (2-component) description of the
# [alpha/Fe]-[Fe/H] distribution between the simulation and (real, external) Milky Way data.
#
# This is the right posture when the *model is wrong*: rather than a star-by-star match to
# model-generated data, we reduce each distribution to an equivalent quantification -- two
# Gaussians (a high-alpha and a low-alpha sequence), each described by a mean and a standard
# deviation along *both* axes ([Fe/H] and [alpha/Fe]), plus the mixing fraction -- and compare
# those summary vectors.  The Maoz Ia parameters are then walked to best reproduce the Milky Way
# summary.  Supply MW summary values measured from a real catalog (e.g. APOGEE) as the target.
# --------------------------------------------------------------------------------------------------

# order of the flattened summary vector produced by summary_to_vector()
BIMODAL_SUMMARY_LABELS = [
    'high.alpha.fraction',
    'high.alpha.mean.feh', 'high.alpha.std.feh',
    'high.alpha.mean.xfe', 'high.alpha.std.xfe',
    'low.alpha.mean.feh', 'low.alpha.std.feh',
    'low.alpha.mean.xfe', 'low.alpha.std.xfe',
]


def fit_bimodal_gaussians(feh, xfe, n_iter=40, reg=1e-4):
    '''
    Fit a 2-component, diagonal-covariance 2-D Gaussian mixture to the
    [Fe/H]-[X/Fe] point cloud via a short, deterministic EM, and return the two
    Gaussians described by their mean and standard deviation along *both* axes.

    This is the "dual Gaussian along both axes" quantification: it captures the
    two sequences (high-alpha and low-alpha) each with (mean, std) in [Fe/H] and
    (mean, std) in [X/Fe], plus the mixing fraction.  Applied identically to the
    simulation and to the (real) Milky Way data, it provides equivalent summaries
    to compare.

    Parameters
    ----------
    feh, xfe : 1-D arrays
        [Fe/H] and [X/Fe] for each star
    n_iter : int
        number of EM iterations (fixed, for a deterministic, smooth summary)
    reg : float
        variance floor added for numerical stability

    Returns
    -------
    summary : dict of length-2 arrays (component 0 = high-alpha, 1 = low-alpha)
        'weight', 'mean_feh', 'std_feh', 'mean_xfe', 'std_xfe'
    '''
    X = np.column_stack([np.asarray(feh, float), np.asarray(xfe, float)])
    n = X.shape[0]
    # deterministic init: split on the median [X/Fe] into high- and low-alpha seeds
    hi = X[:, 1] >= np.median(X[:, 1])
    mu = np.array([
        [X[hi, 0].mean(), X[hi, 1].mean()],
        [X[~hi, 0].mean(), X[~hi, 1].mean()],
    ])
    var = np.tile(X.var(axis=0) + reg, (2, 1))
    w = np.array([hi.mean(), 1.0 - hi.mean()])

    for _ in range(n_iter):
        # E-step (diagonal Gaussians), in log space for stability
        logp = np.empty((n, 2))
        for k in range(2):
            d = X - mu[k]
            logp[:, k] = np.log(w[k] + 1e-300) - 0.5 * np.sum(
                d * d / var[k] + np.log(2 * np.pi * var[k]), axis=1
            )
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        # M-step
        Nk = r.sum(axis=0) + 1e-12
        w = Nk / n
        for k in range(2):
            mu[k] = (r[:, k:k + 1] * X).sum(axis=0) / Nk[k]
            d = X - mu[k]
            var[k] = (r[:, k:k + 1] * d * d).sum(axis=0) / Nk[k] + reg

    # order component 0 = high-alpha (larger mean [X/Fe])
    order = np.argsort(-mu[:, 1])
    std = np.sqrt(var)
    return {
        'weight': w[order],
        'mean_feh': mu[order, 0],
        'std_feh': std[order, 0],
        'mean_xfe': mu[order, 1],
        'std_xfe': std[order, 1],
    }


def summary_to_vector(summary):
    '''
    Flatten a dual-Gaussian summary (from fit_bimodal_gaussians or make_bimodal_summary)
    into the fixed-order vector described by BIMODAL_SUMMARY_LABELS.
    '''
    return np.array([
        summary['weight'][0],
        summary['mean_feh'][0], summary['std_feh'][0],
        summary['mean_xfe'][0], summary['std_xfe'][0],
        summary['mean_feh'][1], summary['std_feh'][1],
        summary['mean_xfe'][1], summary['std_xfe'][1],
    ])


def make_bimodal_summary(
    high_alpha_fraction,
    high_alpha_feh, high_alpha_feh_std, high_alpha_xfe, high_alpha_xfe_std,
    low_alpha_feh, low_alpha_feh_std, low_alpha_xfe, low_alpha_xfe_std,
):
    '''
    Build a dual-Gaussian summary dict (same structure as fit_bimodal_gaussians)
    from explicit values -- e.g. to encode a Milky Way target measured from a real
    survey.  Component 0 is the high-alpha sequence, component 1 the low-alpha.
    '''
    return {
        'weight': np.array([high_alpha_fraction, 1.0 - high_alpha_fraction]),
        'mean_feh': np.array([high_alpha_feh, low_alpha_feh]),
        'std_feh': np.array([high_alpha_feh_std, low_alpha_feh_std]),
        'mean_xfe': np.array([high_alpha_xfe, low_alpha_xfe]),
        'std_xfe': np.array([high_alpha_xfe_std, low_alpha_xfe_std]),
    }


# Illustrative Milky Way [alpha/Fe]-[Fe/H] target on this module's abundance scale
# (metallicity := log10(mass_fraction / mass_fraction_Solar)).  These stand in for a
# 2-component Gaussian fit to a real survey (e.g. APOGEE thin/thick disk); REPLACE with
# values measured from a real catalog for a science application.  Note the model's abundance
# zero-points need not coincide with the data's, so a perfect match may require large
# (non-perturbative) parameter excursions -- that mismatch is itself informative.
MW_TARGET_SUMMARY = make_bimodal_summary(
    high_alpha_fraction=0.40,
    high_alpha_feh=-1.00, high_alpha_feh_std=0.18, high_alpha_xfe=0.20, high_alpha_xfe_std=0.05,
    low_alpha_feh=-0.65, low_alpha_feh_std=0.16, low_alpha_xfe=0.05, low_alpha_xfe_std=0.07,
)

# default 1-sigma uncertainties on each summary statistic (same order as the vector)
DEFAULT_SUMMARY_SIGMA = np.array([0.05, 0.04, 0.03, 0.03, 0.02, 0.04, 0.03, 0.03, 0.02])


def summary_log_likelihood(theta, model, target_vector, sigma_vector, gmm_kwargs=None):
    '''
    Gaussian log likelihood comparing the *simulation's* dual-Gaussian summary at
    parameters theta to a fixed target summary vector (e.g. the Milky Way).

    Parameters
    ----------
    theta : (log10 n_ia, t_dd)
    model : MaozElementTracerModel
    target_vector : 1-D array
        target summary (from summary_to_vector), e.g. the MW quantification
    sigma_vector : 1-D array
        1-sigma uncertainty on each summary statistic
    gmm_kwargs : dict or None
        keyword arguments forwarded to fit_bimodal_gaussians
    '''
    feh, xfe = model.abundances(theta)
    if not (np.all(np.isfinite(feh)) and np.all(np.isfinite(xfe))):
        return -np.inf
    summary = fit_bimodal_gaussians(feh, xfe, **(gmm_kwargs or {}))
    resid = (summary_to_vector(summary) - target_vector) / sigma_vector
    return -0.5 * np.sum(resid ** 2 + np.log(2 * np.pi * sigma_vector ** 2))


def summary_log_probability(
    theta, model, target_vector, sigma_vector, bounds=DEFAULT_BOUNDS, gmm_kwargs=None
):
    '''Log posterior for the summary-statistic (dual-Gaussian) match.'''
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + summary_log_likelihood(theta, model, target_vector, sigma_vector, gmm_kwargs)


def run_mcmc(
    model,
    data=None,
    bounds=DEFAULT_BOUNDS,
    init=None,
    n_walker=24,
    n_step=700,
    n_burn=200,
    seed=None,
    progress=True,
    init_dist='ball',
    init_scale=0.05,
    log_prob_fn=None,
    log_prob_args=None,
):
    '''
    Run an affine-invariant MCMC (emcee) over theta = (log10 n_ia, t_dd).

    Parameters
    ----------
    model : MaozElementTracerModel
        forward model
    data : dict or None
        mock data set (from generate_mock_data), for the default star-by-star
        likelihood; ignored if a custom log_prob_fn is supplied
    bounds : array (2 x 2)
        flat-prior bounds; also used to initialize walkers if init is None
    init : (float, float) or None
        central guess for the walkers; defaults to the midpoint of bounds
    n_walker : int
        number of walkers
    n_step : int
        number of steps per walker (including burn-in)
    n_burn : int
        number of burn-in steps to discard
    seed : int or None
        random seed for walker initialization
    progress : bool
        show emcee's progress bar
    init_dist : str
        how to initialize the walkers:
        'ball'    -> tight Gaussian ball around `init` (default; fast burn-in)
        'uniform' -> spread uniformly in a box of half-width init_scale*(prior
                     range) around `init`, useful to *watch* the walkers converge
                     in the movie (see animate_mcmc_walkers)
    init_scale : float
        size of the initial walker spread as a fraction of the prior range
    log_prob_fn : callable or None
        custom log-posterior function log_prob_fn(theta, *log_prob_args).  If None,
        use the default star-by-star log_probability(theta, model, data, bounds).
        Pass summary_log_probability (with log_prob_args) to fit the dual-Gaussian
        summary to an external (Milky Way) target instead.
    log_prob_args : tuple or None
        extra positional arguments for log_prob_fn

    Returns
    -------
    sampler : emcee.EnsembleSampler
    flat_chain : 2-D array (n_sample x 2)
        post-burn-in flattened chain
    '''
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            'run_mcmc requires the emcee package (pip install emcee)'
        ) from exc

    rng = np.random.default_rng(seed)
    ndim = 2
    if init is None:
        init = np.mean(bounds, axis=1)
    init = np.asarray(init, dtype=float)

    span = bounds[:, 1] - bounds[:, 0]
    if init_dist == 'uniform':
        # spread walkers uniformly in a box around init (clipped to the prior)
        half = init_scale * span
        lo = np.maximum(init - half, bounds[:, 0])
        hi = np.minimum(init + half, bounds[:, 1])
        p0 = rng.uniform(lo, hi, size=(n_walker, ndim))
    else:
        # tight Gaussian ball around init
        p0 = init + init_scale * span * rng.standard_normal((n_walker, ndim))
        p0 = np.clip(p0, bounds[:, 0], bounds[:, 1])

    if log_prob_fn is None:
        log_prob_fn = log_probability
        log_prob_args = (model, data, bounds)
    elif log_prob_args is None:
        log_prob_args = ()

    sampler = emcee.EnsembleSampler(n_walker, ndim, log_prob_fn, args=log_prob_args)
    sampler.run_mcmc(p0, n_step, progress=progress)
    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    return sampler, flat_chain


def summarize_chain(flat_chain, truths=None, labels=('log10 n_ia', 't_dd')):
    '''
    Print and return median +/- 1-sigma (16th/84th percentile) estimates for each
    parameter, plus the implied linear n_ia.

    Returns
    -------
    summary : dict
        keys are labels; values are (median, minus, plus)
    '''
    summary = {}
    for i, label in enumerate(labels):
        lo, med, hi = np.percentile(flat_chain[:, i], [16, 50, 84])
        summary[label] = (med, med - lo, hi - med)
        truth_str = ''
        if truths is not None:
            truth_str = '   (true = {:.4g})'.format(truths[i])
        print('{:>12s} = {:.4g}  (+{:.3g} / -{:.3g}){}'.format(
            label, med, hi - med, med - lo, truth_str))
    # also report n_ia in linear units
    if labels[0].startswith('log10'):
        med_n = 10.0 ** summary[labels[0]][0]
        print('{:>12s} = {:.4g}'.format('n_ia', med_n))
    return summary


# --------------------------------------------------------------------------------------------------
# Plotting helpers (matplotlib optional)
# --------------------------------------------------------------------------------------------------
def plot_data_and_model(model, data, theta, path=None, xfe_label=None):
    '''
    Scatter the mock data in the [X/Fe]-[Fe/H] plane and overlay the forward-model
    prediction at parameter vector theta.  Saves to path if given, else shows.
    '''
    import matplotlib

    if path is not None:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    feh_model, xfe_model = model.abundances(theta)
    if xfe_label is None:
        xfe_label = 'alpha' if model.xfe == 'alpha' else model.xfe.capitalize()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(data['feh'], data['xfe'], s=10, alpha=0.4, color='0.5', label='mock data')
    ax.scatter(feh_model, xfe_model, s=10, alpha=0.6, color='crimson', label='best-fit model')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[{}/Fe]'.format(xfe_label))
    ax.legend(frameon=False)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_bimodal_data(data, model=None, theta=None, path=None, xfe_label=None):
    '''
    Scatter a bimodal mock data set in the [X/Fe]-[Fe/H] plane, coloring the two
    sequences (high-alpha vs low-alpha) if data['label'] is present, and
    optionally overlay the forward-model prediction at parameter vector theta.

    Parameters
    ----------
    data : dict
        mock data set; if it has a 'label' key (1 = high-alpha, 0 = low-alpha)
        the two sequences are colored separately
    model : MaozElementTracerModel or None
        if given (with theta), overlay the model prediction
    theta : sequence or None
        parameter vector for the model overlay
    path : str or None
        save to this path, else show
    xfe_label : str or None
        label for the [X/Fe] axis
    '''
    import matplotlib

    if path is not None:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if xfe_label is None:
        xfe_label = 'alpha' if (model is not None and model.xfe == 'alpha') else 'X'

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    if 'label' in data:
        hi = data['label'] == 1
        ax.scatter(data['feh'][hi], data['xfe'][hi], s=10, alpha=0.5,
                   color='firebrick', label='high-alpha (thick disk)')
        ax.scatter(data['feh'][~hi], data['xfe'][~hi], s=10, alpha=0.5,
                   color='steelblue', label='low-alpha (thin disk)')
    else:
        ax.scatter(data['feh'], data['xfe'], s=10, alpha=0.4, color='0.5',
                   label='mock data')
    if model is not None and theta is not None:
        feh_model, xfe_model = model.abundances(theta)
        ax.scatter(feh_model, xfe_model, s=6, alpha=0.5, color='k',
                   label='best-fit model')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[{}/Fe]'.format(xfe_label))
    ax.legend(frameon=False)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def _draw_dual_gaussian(ax, summary, n_sigma=2, colors=('firebrick', 'steelblue'),
                        ls='-', lw=2.0, label=None):
    '''
    Draw a dual-Gaussian summary as two axis-aligned n_sigma ellipses (component 0
    = high-alpha, 1 = low-alpha) on axis `ax`.  Returns the list of patches.
    '''
    from matplotlib.patches import Ellipse

    patches = []
    for k in range(2):
        e = Ellipse(
            (summary['mean_feh'][k], summary['mean_xfe'][k]),
            width=2 * n_sigma * summary['std_feh'][k],
            height=2 * n_sigma * summary['std_xfe'][k],
            fill=False, edgecolor=colors[k], ls=ls, lw=lw,
            label=(label if k == 0 else None),
        )
        ax.add_patch(e)
        patches.append(e)
    return patches


def animate_mcmc_walkers(
    chain,
    path,
    truths=None,
    labels=PARAM_LABELS,
    bounds=None,
    fps=30,
    burn=0,
    stride=1,
    dpi=110,
    model=None,
    data=None,
    target_summary=None,
    abundance_lims=None,
):
    '''
    Render an mp4 movie of the MCMC walkers evolving (post-processing).

    Layout
    ------
    Left column  : per-parameter walker traces (parameter value vs step), one thin
                   line per walker, revealed step by step.
    Middle block : a live 2-parameter "corner" -- the 1-D marginal histograms plus
                   the 2-D joint -- that fills in as samples accumulate, with a
                   moving dot marking the current ensemble-median position (the
                   "dot on the corner plot") and a cross marking the truth.
    Right panel  : (if `model` is given) the [X/Fe]-[Fe/H] plane showing the
                   forward-model prediction at the *current* ensemble-median
                   parameters, so you watch the abundance distribution shift as the
                   Ia delay-time-distribution parameters change.  The fixed target
                   it walks toward is either the observed data points (`data`) or,
                   if `target_summary` is given, the Milky Way dual-Gaussian target
                   drawn as 2-sigma ellipses (with the simulation's own fitted
                   dual-Gaussian ellipses overplotted as they evolve).

    Parameters
    ----------
    chain : 3-D array (n_step, n_walker, ndim=2)
        the full un-flattened chain, e.g. sampler.get_chain()
    path : str
        output .mp4 path
    truths : sequence or None
        true parameter values to mark
    labels : sequence of str
        parameter labels (length 2)
    bounds : array (2, 2) or None
        axis limits per parameter; defaults to the chain range (padded)
    fps : int
        frames per second (30 by default)
    burn : int
        steps excluded from the accumulating posterior histograms (still shown in
        the trace panels)
    stride : int
        render every `stride`-th step (1 = every step)
    dpi : int
        figure resolution
    model : MaozElementTracerModel or None
        if given, add the evolving [X/Fe]-[Fe/H] panel
    data : dict or None
        observed data points to show as the fixed target (from generate_mock_data);
        uses data['label'] to color the two sequences and, in summary mode, to
        color the evolving simulation points
    target_summary : dict or None
        a dual-Gaussian target (e.g. MW_TARGET_SUMMARY); if given, the panel draws
        this target as 2-sigma ellipses and overplots the simulation's own fitted
        dual-Gaussian ellipses at the current parameters ("summary mode")
    abundance_lims : ((xmin, xmax), (ymin, ymax)) or None
        fixed axis limits for the abundance panel; defaults to the data range
        (padded).  Fixed limits are important so the shifting distribution is
        visible against a stable frame.

    Returns
    -------
    path : str
        the output path written
    '''
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            'animate_mcmc_walkers requires imageio + a mp4 encoder '
            '(pip install imageio imageio-ffmpeg)'
        ) from exc
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    chain = np.asarray(chain)
    n_step, n_walker, ndim = chain.shape
    assert ndim == 2, 'animate_mcmc_walkers is written for a 2-parameter model'

    show_ab = model is not None and (data is not None or target_summary is not None)
    summary_mode = model is not None and target_summary is not None
    if show_ab:
        xfe_label = 'alpha' if model.xfe == 'alpha' else model.xfe.capitalize()
        has_label = data is not None and 'label' in data
        if has_label:
            hi_mask = data['label'] == 1
        # fixed limits for the abundance panel
        if abundance_lims is not None:
            ab_x, ab_y = abundance_lims
        elif data is not None:
            fpad = 0.20
            ab_x = (np.min(data['feh']) - fpad, np.max(data['feh']) + fpad)
            ab_y = (np.min(data['xfe']) - fpad, np.max(data['xfe']) + fpad)
        else:
            # derive from the target dual-Gaussian (means +/- ~4 sigma)
            fpad = 0.25
            mf, sf = target_summary['mean_feh'], target_summary['std_feh']
            mx, sx = target_summary['mean_xfe'], target_summary['std_xfe']
            ab_x = (np.min(mf - 3 * sf) - fpad, np.max(mf + 3 * sf) + fpad)
            ab_y = (np.min(mx - 3 * sx) - fpad, np.max(mx + 3 * sx) + fpad)

    # per-parameter axis limits
    if bounds is not None:
        lims = np.asarray(bounds, dtype=float)
    else:
        lims = np.empty((2, 2))
        for k in range(2):
            lo, hi = chain[..., k].min(), chain[..., k].max()
            pad = 0.05 * (hi - lo + 1e-12)
            lims[k] = [lo - pad, hi + pad]

    bins0 = np.linspace(lims[0, 0], lims[0, 1], 40)
    bins1 = np.linspace(lims[1, 0], lims[1, 1], 40)

    if show_ab:
        fig = plt.figure(figsize=(17, 6), dpi=dpi)
        gs = fig.add_gridspec(
            2, 5, width_ratios=[1.0, 1.0, 0.95, 0.36, 1.75], height_ratios=[1, 1],
            hspace=0.30, wspace=0.34,
        )
        ax_ab = fig.add_subplot(gs[0:2, 4])
    else:
        fig = plt.figure(figsize=(12, 6), dpi=dpi)
        gs = fig.add_gridspec(
            2, 4, width_ratios=[1.0, 1.0, 1.1, 0.4], height_ratios=[1, 1],
            hspace=0.28, wspace=0.30,
        )
    ax_tr0 = fig.add_subplot(gs[0, 0:2])
    ax_tr1 = fig.add_subplot(gs[1, 0:2])
    ax_h0 = fig.add_subplot(gs[0, 2])   # 1-D marginal of param 0 (top)
    ax_j = fig.add_subplot(gs[1, 2])    # 2-D joint (bottom-left of corner)
    ax_h1 = fig.add_subplot(gs[1, 3])   # 1-D marginal of param 1 (right, rotated)

    frames = list(range(1, n_step + 1, stride))
    if frames[-1] != n_step:
        frames.append(n_step)

    x_all = np.arange(n_step)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=None, codec='libx264')
    try:
        for f in frames:
            # ---- walker trace panels --------------------------------------------------------
            for k, (ax, lim) in enumerate(((ax_tr0, lims[0]), (ax_tr1, lims[1]))):
                ax.clear()
                ax.plot(x_all[:f], chain[:f, :, k], color='0.3', alpha=0.35, lw=0.6)
                if truths is not None:
                    ax.axhline(truths[k], color='crimson', ls='--', lw=1.2)
                ax.set_xlim(0, n_step)
                ax.set_ylim(lim)
                ax.set_ylabel(labels[k])
                ax.grid(ls=':', alpha=0.4)
            ax_tr1.set_xlabel('step')
            ax_tr0.set_title('walker traces', fontsize=11)

            # ---- accumulating corner --------------------------------------------------------
            if f > burn:
                samp = chain[burn:f].reshape(-1, 2)
            else:
                samp = chain[:f].reshape(-1, 2)
            current = chain[f - 1]                 # walker positions at this step
            median = np.median(current, axis=0)    # the moving "dot"

            ax_h0.clear()
            ax_h0.hist(samp[:, 0], bins=bins0, color='0.6')
            if truths is not None:
                ax_h0.axvline(truths[0], color='crimson', ls='--', lw=1.2)
            ax_h0.axvline(median[0], color='navy', lw=1.4)
            ax_h0.set_xlim(lims[0])
            ax_h0.set_xticklabels([])
            ax_h0.set_yticks([])
            ax_h0.set_title('posterior (building)', fontsize=11)

            ax_h1.clear()
            ax_h1.hist(samp[:, 1], bins=bins1, orientation='horizontal', color='0.6')
            if truths is not None:
                ax_h1.axhline(truths[1], color='crimson', ls='--', lw=1.2)
            ax_h1.axhline(median[1], color='navy', lw=1.4)
            ax_h1.set_ylim(lims[1])
            ax_h1.set_yticklabels([])
            ax_h1.set_xticks([])

            ax_j.clear()
            ax_j.scatter(samp[:, 0], samp[:, 1], s=4, alpha=0.10, color='0.5')
            ax_j.scatter(current[:, 0], current[:, 1], s=14, color='navy',
                         alpha=0.7, label='walkers')
            ax_j.scatter([median[0]], [median[1]], s=130, color='gold',
                         edgecolor='k', zorder=5, label='ensemble median')
            if truths is not None:
                ax_j.axvline(truths[0], color='crimson', ls='--', lw=1.0)
                ax_j.axhline(truths[1], color='crimson', ls='--', lw=1.0)
            ax_j.set_xlim(lims[0])
            ax_j.set_ylim(lims[1])
            ax_j.set_xlabel(labels[0])
            ax_j.set_ylabel(labels[1])
            ax_j.legend(loc='upper left', fontsize=8, frameon=False)

            # ---- evolving abundance plane ---------------------------------------------------
            if show_ab:
                ax_ab.clear()
                feh_m, xfe_m = model.abundances(median)  # simulation at current params
                if summary_mode:
                    # fixed Milky Way target as dual-Gaussian 2-sigma ellipses
                    _draw_dual_gaussian(ax_ab, target_summary, n_sigma=2,
                                        colors=('firebrick', 'steelblue'), ls='--', lw=2.2,
                                        label='MW target (2$\\sigma$)')
                    # evolving simulation points (colored by sequence if labels given)
                    if has_label:
                        ax_ab.scatter(feh_m[hi_mask], xfe_m[hi_mask], s=7,
                                      color='lightcoral', alpha=0.35)
                        ax_ab.scatter(feh_m[~hi_mask], xfe_m[~hi_mask], s=7,
                                      color='lightskyblue', alpha=0.35)
                    else:
                        ax_ab.scatter(feh_m, xfe_m, s=7, color='0.6', alpha=0.35)
                    ax_ab.scatter([], [], s=20, color='0.55', label='simulation')
                    # the simulation's own fitted dual-Gaussian, evolving toward the target
                    sim_summary = fit_bimodal_gaussians(feh_m, xfe_m)
                    _draw_dual_gaussian(ax_ab, sim_summary, n_sigma=2,
                                        colors=('darkred', 'navy'), ls='-', lw=1.8,
                                        label='simulation fit (2$\\sigma$)')
                else:
                    # fixed observed data points (the target)
                    if has_label:
                        ax_ab.scatter(data['feh'][hi_mask], data['xfe'][hi_mask], s=7,
                                      color='lightcoral', alpha=0.30)
                        ax_ab.scatter(data['feh'][~hi_mask], data['xfe'][~hi_mask], s=7,
                                      color='lightskyblue', alpha=0.30)
                        ax_ab.scatter([], [], s=20, color='0.55', label='observed data')
                    else:
                        ax_ab.scatter(data['feh'], data['xfe'], s=7, color='0.7',
                                      alpha=0.30, label='observed data')
                    # model prediction at the current ensemble-median parameters
                    if has_label:
                        ax_ab.scatter(feh_m[hi_mask], xfe_m[hi_mask], s=9,
                                      color='firebrick', alpha=0.6)
                        ax_ab.scatter(feh_m[~hi_mask], xfe_m[~hi_mask], s=9,
                                      color='steelblue', alpha=0.6)
                        ax_ab.scatter([], [], s=20, color='k', label='model (current params)')
                    else:
                        ax_ab.scatter(feh_m, xfe_m, s=9, color='k', alpha=0.6,
                                      label='model (current params)')
                ax_ab.set_xlim(ab_x)
                ax_ab.set_ylim(ab_y)
                ax_ab.set_xlabel('[Fe/H]')
                ax_ab.set_ylabel('[{}/Fe]'.format(xfe_label))
                ax_ab.set_title(
                    r'[{}/Fe] vs [Fe/H]   ($n_{{\rm Ia}}$={:.2e}, $t_{{\rm dd}}$={:.2f})'.format(
                        xfe_label, 10.0 ** median[0], median[1]),
                    fontsize=11)
                ax_ab.legend(loc='upper right', fontsize=8, frameon=False)
                ax_ab.grid(ls='-.', alpha=0.35)

            fig.suptitle('MCMC step {} / {}'.format(f, n_step), fontsize=13)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    return path


def plot_corner(flat_chain, truths=None, labels=PARAM_LABELS, path=None):
    '''
    Corner plot of the posterior.  Requires the `corner` package.  Saves to path
    if given, else shows.
    '''
    try:
        import corner
    except ImportError as exc:
        raise ImportError('plot_corner requires the corner package (pip install corner)') from exc
    import matplotlib

    if path is not None:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig = corner.corner(flat_chain, labels=labels, truths=truths, show_titles=True)
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig
