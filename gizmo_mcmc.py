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
        ia_transition_time=None,
        cc_normalization=None,
        cc_transition_time=None,
        initial_massfraction=None,
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
            FIRE rate+yield model version passed to FIREYieldClass2
        ia_transition_time : list or None
            transition time(s) [Myr] for the Ia model (default [37.53])
        cc_normalization, cc_transition_time : list or None
            optionally override the CCSN rate model (kept fixed during the fit)
        initial_massfraction : dict or None
            optional initial (pre-enrichment) linear mass fractions per element
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
        self.sun_massfraction = gizmo_model.get_sun_massfraction()

        # elements we actually need to integrate (keep this minimal for speed)
        if xfe == 'alpha':
            needed = ['iron'] + list(ALPHA_ELEMENTS)
        else:
            needed = ['iron', xfe]
        # de-duplicate while preserving order
        self.element_names = list(dict.fromkeys(needed))

        # keep the (fixed) rate-model knobs to pass through to FIREYieldClass2
        self._yield_kwargs = {'model': model, 'ia_type': 'maoz'}
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
        needed elements, for Maoz Ia parameters (n_ia, t_dd).

        Returns the element_yield_dict produced by FIREYieldClass2.get_element_yields.
        '''
        fyield = gizmo_agetracer.FIREYieldClass2(
            normalization_ia=n_ia, tdd_ia=t_dd, **self._yield_kwargs
        )
        return fyield.get_element_yields(
            self.age_bins, element_names=self.element_names, continuous=True
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


def generate_mock_data(model, true_theta, obs_std=(0.08, 0.06), seed=None):
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

    Returns
    -------
    data : dict
        keys: 'feh', 'xfe' (scattered mock observations), 'feh_err', 'xfe_err'
        (per-star errors), 'feh_true', 'xfe_true' (noise-free model values),
        'true_theta', 'obs_std'
    '''
    rng = np.random.default_rng(seed)
    feh_true, xfe_true = model.abundances(true_theta)
    sig_feh, sig_xfe = obs_std
    feh = feh_true + rng.normal(0.0, sig_feh, size=feh_true.shape)
    xfe = xfe_true + rng.normal(0.0, sig_xfe, size=xfe_true.shape)
    n_star = feh.size
    return {
        'feh': feh,
        'xfe': xfe,
        'feh_err': np.full(n_star, sig_feh),
        'xfe_err': np.full(n_star, sig_xfe),
        'feh_true': feh_true,
        'xfe_true': xfe_true,
        'true_theta': np.asarray(true_theta, dtype=float),
        'obs_std': np.asarray(obs_std, dtype=float),
    }


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


def run_mcmc(
    model,
    data,
    bounds=DEFAULT_BOUNDS,
    init=None,
    n_walker=24,
    n_step=700,
    n_burn=200,
    seed=None,
    progress=True,
):
    '''
    Run an affine-invariant MCMC (emcee) over theta = (log10 n_ia, t_dd).

    Parameters
    ----------
    model : MaozElementTracerModel
        forward model
    data : dict
        mock data set (from generate_mock_data)
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

    # start walkers in a small Gaussian ball around init, clipped to the prior
    scale = 0.05 * (bounds[:, 1] - bounds[:, 0])
    p0 = init + scale * rng.standard_normal((n_walker, ndim))
    p0 = np.clip(p0, bounds[:, 0], bounds[:, 1])

    sampler = emcee.EnsembleSampler(
        n_walker, ndim, log_probability, args=(model, data, bounds)
    )
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
