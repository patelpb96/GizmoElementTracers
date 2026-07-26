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
    init_dist='ball',
    init_scale=0.05,
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
    init_dist : str
        how to initialize the walkers:
        'ball'    -> tight Gaussian ball around `init` (default; fast burn-in)
        'uniform' -> spread uniformly in a box of half-width init_scale*(prior
                     range) around `init`, useful to *watch* the walkers converge
                     in the movie (see animate_mcmc_walkers)
    init_scale : float
        size of the initial walker spread as a fraction of the prior range

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
    Right panel  : (if `model` and `data` are given) the [X/Fe]-[Fe/H] plane, with
                   the fixed observed data and the forward-model prediction at the
                   *current* ensemble-median parameters overplotted, so you watch
                   the abundance distribution shift as the Ia delay-time-distribution
                   parameters change and settle onto the data.

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
        if given together with `data`, add the evolving [X/Fe]-[Fe/H] panel
    data : dict or None
        the observed mock data (from generate_mock_data); uses data['label'] to
        color the two sequences if present
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

    show_ab = model is not None and data is not None
    if show_ab:
        xfe_label = 'alpha' if model.xfe == 'alpha' else model.xfe.capitalize()
        has_label = 'label' in data
        if has_label:
            hi_mask = data['label'] == 1
        # fixed limits for the abundance panel
        if abundance_lims is not None:
            ab_x, ab_y = abundance_lims
        else:
            fpad = 0.20
            ab_x = (np.min(data['feh']) - fpad, np.max(data['feh']) + fpad)
            ab_y = (np.min(data['xfe']) - fpad, np.max(data['xfe']) + fpad)

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
                # fixed observed data (the target)
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
                feh_m, xfe_m = model.abundances(median)
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
