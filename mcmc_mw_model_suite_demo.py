'''
Milky-Way summary matches for a suite of physically motivated SNe Ia delay-time-distribution
(DTD) models.

For each DTD family in the suite we walk its parameters (with an affine-invariant MCMC) so that
the simulation's dual-Gaussian summary of the [alpha/Fe]-[Fe/H] plane -- two sequences, each a
(mean, std) along both axes, plus the mixing fraction -- best reproduces a fixed Milky Way target
(gizmo_mcmc.MW_TARGET_SUMMARY).  This is the "match a real, imperfect target" posture: rather than
a star-by-star fit to model-generated data, we compare equivalent quantifications and report how
close each (perturbative) DTD family can get, plus the residual mismatch per summary statistic.

The models (all delay-time distributions psi(t), evaluated through the same element-tracer forward
model and fast yield integrator), grounded in the DTD literature:

    M1 maoz            ~t^-1 power law (the fiducial Maoz DTD)                 [2 params]
    M2 maoz_onset      power law with a free white-dwarf-formation onset       [3 params]
    M3 kink            broken power law (slope change at t_kink)               [4 params]
    M4 prompt_delayed  power law + prompt Gaussian (prompt fraction)          [5 params]
    M5 long_delay      power law + long-delay (double-degenerate) Gaussian    [5 params]
    M6 skewnorm        Strolger (2020) skew-normal in log-age                 [4 params]
    M7 exponential     exponential DTD (a steep foil)                         [2 params]
    M8 kink_prompt     broken power law + prompt bump + free onset ("kitchen  [7 params]
                       sink"; the hardest, most degenerate corner)

Run (all models):
    python mcmc_mw_model_suite_demo.py

Run a subset, quickly, without movies:
    python mcmc_mw_model_suite_demo.py --models maoz kink skewnorm --quick --no-movie

Outputs (per model <key>, in --outdir):
    mw_suite_<key>_abundance.png  -- MW target ellipses + fiducial vs best-fit simulation
    mw_suite_<key>_corner.png     -- posterior of the best-fit parameters
    mw_suite_<key>_walkers.mp4    -- walkers converging + simulation walking onto the MW target
'''

import argparse
import os
import sys

import numpy as np

# ---- numpy>=2 compatibility shim -----------------------------------------------------------------
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_REPO_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
_PACKAGE_NAME = os.path.basename(_REPO_DIR)
gizmo_mcmc = __import__('{}.gizmo_mcmc'.format(_PACKAGE_NAME), fromlist=['gizmo_mcmc'])


# the suite, in order (registry key -> short human title)
SUITE = [
    ('maoz', 'M1 power law (Maoz)'),
    ('maoz_onset', 'M2 power law + free onset'),
    ('kink', 'M3 broken power law'),
    ('prompt_delayed', 'M4 prompt + delayed'),
    ('long_delay', 'M5 power law + long-delay bump'),
    ('skewnorm', 'M6 skew-normal (Strolger)'),
    ('exponential', 'M7 exponential'),
    ('kink_prompt', 'M8 kitchen sink'),
]


def _print_summary_match(target_vec, sim_vec, sigma_vec):
    '''Print target vs simulation for each dual-Gaussian summary statistic.'''
    print('{:>22s}  {:>8s} {:>8s} {:>8s}'.format('summary statistic', 'MW', 'sim', 'resid/sig'))
    for name, t, s, sg in zip(gizmo_mcmc.BIMODAL_SUMMARY_LABELS, target_vec, sim_vec, sigma_vec):
        print('{:>22s}  {:+8.3f} {:+8.3f} {:+8.2f}'.format(name, t, s, (s - t) / sg))
    chi2 = np.sum(((sim_vec - target_vec) / sigma_vec) ** 2)
    print('  chi^2 = {:.2f}  over {} summary statistics'.format(chi2, len(target_vec)))


def run_one_model(key, title, weights, labels, age_bins, target_vec, sigma_vec, target, args):
    '''Run the MW-summary match for a single DTD model and write its figures/movie.'''
    print('\n' + '=' * 92)
    print('{}   (registry key: {})'.format(title, key))
    print('=' * 92)

    sampled, bounds, param_labels, fiducial = gizmo_mcmc.model_prior(key)
    model = gizmo_mcmc.MaozElementTracerModel(
        age_bins, weights, xfe='alpha', ia_model=key, sampled_params=sampled
    )
    ndim = len(sampled)
    print('sampling {} parameter(s): {}'.format(ndim, ', '.join(sampled)))

    fiducial_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(fiducial))
    print('\nat the fiducial (unperturbed) model:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(fiducial_summary), sigma_vec)

    # start walkers spread (uniformly) around the fiducial so the movie shows them walk onto the MW
    print('\nrunning MCMC ({} walkers x {} steps)...'.format(args.n_walker, args.n_step))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, bounds=bounds, init=fiducial,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=0.15, progress=args.progress,
        log_prob_fn=gizmo_mcmc.summary_log_probability,
        log_prob_args=(model, target_vec, sigma_vec, bounds, None),
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    print('\nbest-fit parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, labels=sampled)

    theta_med = np.median(flat_chain, axis=0)
    best_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(theta_med))
    print('\nat the best-fit model:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(best_summary), sigma_vec)

    # ---- static abundance figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

    ab_lims = ((-1.6, 0.05), (-0.35, 0.45))
    fig, ax = plt.subplots(figsize=(7.6, 6))
    feh_f, xfe_f = model.abundances(fiducial)
    feh_b, xfe_b = model.abundances(theta_med)
    ax.scatter(feh_f, xfe_f, s=6, color='0.75', alpha=0.35, label='simulation (fiducial)')
    ax.scatter(feh_b, xfe_b, s=6, color='0.35', alpha=0.5, label='simulation (best-fit)')
    gizmo_mcmc._draw_dual_gaussian(ax, target, n_sigma=2, ls='--', lw=2.3,
                                   label='MW target (2$\\sigma$)')
    gizmo_mcmc._draw_dual_gaussian(ax, best_summary, n_sigma=2, colors=('darkred', 'navy'),
                                   ls='-', lw=1.8, label='best-fit sim (2$\\sigma$)')
    ax.set_xlim(ab_lims[0])
    ax.set_ylim(ab_lims[1])
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[alpha/Fe]')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    ab_path = os.path.join(args.outdir, 'mw_suite_{}_abundance.png'.format(key))
    fig.savefig(ab_path, dpi=150)
    plt.close(fig)
    print('\nwrote {}'.format(ab_path))

    try:
        corner_path = os.path.join(args.outdir, 'mw_suite_{}_corner.png'.format(key))
        gizmo_mcmc.plot_corner(flat_chain, truths=None, labels=param_labels, path=corner_path)
        print('wrote {}'.format(corner_path))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    # ---- movie -----------------------------------------------------------------------------------
    if not args.no_movie:
        chain = sampler.get_chain()
        movie_path = os.path.join(args.outdir, 'mw_suite_{}_walkers.mp4'.format(key))
        print('rendering movie ({} frames @ {} fps)...'.format(chain.shape[0], args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, movie_path, truths=None, labels=param_labels, bounds=bounds,
                fps=args.fps, burn=0, stride=args.stride, dpi=args.dpi, model=model,
                data={'label': labels}, target_summary=target, abundance_lims=ab_lims,
                rate_fiducial_theta=list(fiducial), rate_ylim=(1e-12, 1e-3),
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))

    chi2 = np.sum(((gizmo_mcmc.summary_to_vector(best_summary) - target_vec) / sigma_vec) ** 2)
    return chi2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=[k for k, _ in SUITE],
                        help='registry keys to run (default: all 8)')
    parser.add_argument('--n-star', type=int, default=1500)
    parser.add_argument('--high-alpha-frac', type=float, default=0.4)
    parser.add_argument('--n-age-bin', type=int, default=10)
    parser.add_argument('--n-walker', type=int, default=24)
    parser.add_argument('--n-step', type=int, default=150)
    parser.add_argument('--n-burn', type=int, default=60)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dpi', type=int, default=110)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--outdir', default=os.getcwd())
    parser.add_argument('--no-movie', action='store_true')
    parser.add_argument('--progress', action='store_true', help='show emcee progress bars')
    parser.add_argument('--quick', action='store_true',
                        help='fast settings for a smoke test (fewer stars/steps, coarse movie)')
    args = parser.parse_args()

    if args.quick:
        args.n_star = min(args.n_star, 500)
        args.n_step = min(args.n_step, 60)
        args.n_burn = min(args.n_burn, 25)
        args.stride = max(args.stride, 2)
        args.dpi = min(args.dpi, 90)

    os.makedirs(args.outdir, exist_ok=True)
    titles = dict(SUITE)
    unknown = [m for m in args.models if m not in gizmo_mcmc.IA_MODEL_SPECS]
    if unknown:
        parser.error('unknown model key(s): {}\navailable: {}'.format(
            ', '.join(unknown), ', '.join(k for k, _ in SUITE)))

    # one shared bimodal stellar population + MW target for all models
    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights, labels = gizmo_mcmc.generate_bimodal_weights(
        args.n_star, args.n_age_bin, high_alpha_frac=args.high_alpha_frac, seed=args.seed
    )
    target = gizmo_mcmc.MW_TARGET_SUMMARY
    target_vec = gizmo_mcmc.summary_to_vector(target)
    sigma_vec = gizmo_mcmc.DEFAULT_SUMMARY_SIGMA

    print('matching {} DTD model(s) to the Milky Way dual-Gaussian summary'.format(len(args.models)))
    print('(shared bimodal population: {} stars, {} high-alpha)'.format(
        args.n_star, int(round(args.high_alpha_frac * args.n_star))))

    results = {}
    for key in args.models:
        title = titles.get(key, key)
        results[key] = run_one_model(
            key, title, weights, labels, age_bins, target_vec, sigma_vec, target, args
        )

    # ---- leaderboard: how close each family gets to the MW summary --------------------------------
    print('\n' + '=' * 60)
    print('best-fit mismatch to the MW summary (lower is closer):')
    print('=' * 60)
    for key in sorted(results, key=results.get):
        print('  {:>16s}   chi^2 = {:7.2f}'.format(key, results[key]))


if __name__ == '__main__':
    main()
