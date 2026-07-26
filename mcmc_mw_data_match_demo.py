'''
Infer the Maoz SNe Ia parameters that best reproduce *real* Milky Way data, under
the (honest) assumption that the model is imperfect -- by matching a dual-Gaussian
summary of the [alpha/Fe]-[Fe/H] distribution rather than doing a star-by-star fit.

Idea
----
The Milky Way is bimodal in [alpha/Fe] vs [Fe/H].  We quantify each distribution --
the (external, fixed) MW data and our simulation -- the *same* way: a two-component
Gaussian mixture (a high-alpha and a low-alpha sequence), each described by a mean
and a standard deviation along *both* axes, plus the mixing fraction.  We then walk
the Maoz Ia parameters (log10 n_ia, t_dd) to minimize the mismatch between the two
summary vectors.  Because the model is not expected to match the data perfectly, the
best-fit is the closest the (perturbative) model can get, and the residual mismatch
per summary statistic is itself informative.

The MW target here (gizmo_mcmc.MW_TARGET_SUMMARY) is an illustrative dual-Gaussian on
this module's abundance scale; for a real analysis, drop in a 2-component Gaussian fit
to an actual catalog (e.g. APOGEE) via gizmo_mcmc.make_bimodal_summary(...) or
gizmo_mcmc.fit_bimodal_gaussians(feh, xfe).

Run:
    python mcmc_mw_data_match_demo.py

Outputs (current directory):
    mcmc_mw_match_abundance.png  -- MW target ellipses + fiducial vs best-fit simulation
    mcmc_mw_match_corner.png     -- posterior of the best-fit Maoz parameters
    mcmc_mw_match_walkers.mp4    -- walkers converging + simulation walking onto the MW target
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


def _print_summary_match(target_vec, sim_vec, sigma_vec):
    '''Print target vs simulation for each dual-Gaussian summary statistic.'''
    print('{:>22s}  {:>8s} {:>8s} {:>8s}'.format('summary statistic', 'MW', 'sim', 'resid/sig'))
    for name, t, s, sg in zip(gizmo_mcmc.BIMODAL_SUMMARY_LABELS, target_vec, sim_vec, sigma_vec):
        print('{:>22s}  {:+8.3f} {:+8.3f} {:+8.2f}'.format(name, t, s, (s - t) / sg))
    chi2 = np.sum(((sim_vec - target_vec) / sigma_vec) ** 2)
    print('  chi^2 = {:.2f}  over {} summary statistics'.format(chi2, len(target_vec)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-star', type=int, default=1500, help='number of simulated stars')
    parser.add_argument('--high-alpha-frac', type=float, default=0.4)
    parser.add_argument('--n-age-bin', type=int, default=10)
    parser.add_argument('--n-walker', type=int, default=24)
    parser.add_argument('--n-step', type=int, default=150)
    parser.add_argument('--n-burn', type=int, default=60)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-movie', action='store_true')
    args = parser.parse_args()

    # ---- simulated bimodal population ------------------------------------------------------------
    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights, labels = gizmo_mcmc.generate_bimodal_weights(
        args.n_star, args.n_age_bin, high_alpha_frac=args.high_alpha_frac, seed=args.seed
    )
    model = gizmo_mcmc.MaozElementTracerModel(age_bins, weights, xfe='alpha')

    # ---- the (external, fixed) Milky Way dual-Gaussian target ------------------------------------
    target = gizmo_mcmc.MW_TARGET_SUMMARY
    target_vec = gizmo_mcmc.summary_to_vector(target)
    sigma_vec = gizmo_mcmc.DEFAULT_SUMMARY_SIGMA

    fiducial_theta = (np.log10(gizmo_mcmc.NIA_DEFAULT), gizmo_mcmc.TDD_DEFAULT)
    fiducial_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(fiducial_theta))
    print('quantifying both distributions as a dual Gaussian (mean & std along both axes)\n')
    print('at the fiducial (unperturbed) model:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(fiducial_summary), sigma_vec)

    # ---- MCMC: walk the Maoz parameters to match the MW summary ----------------------------------
    # start the walkers at an off guess so the movie shows the simulation walk onto the MW target
    init_guess = (np.log10(gizmo_mcmc.NIA_DEFAULT * 1.25), -1.15)
    print('\nrunning MCMC ({} walkers x {} steps), matching the dual-Gaussian summary...'.format(
        args.n_walker, args.n_step))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, bounds=gizmo_mcmc.DEFAULT_BOUNDS, init=init_guess,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=0.15,
        log_prob_fn=gizmo_mcmc.summary_log_probability,
        log_prob_args=(model, target_vec, sigma_vec, gizmo_mcmc.DEFAULT_BOUNDS, None),
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    print('\nbest-fit Maoz parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain)

    theta_med = np.median(flat_chain, axis=0)
    best_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(theta_med))
    print('\nat the best-fit model:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(best_summary), sigma_vec)

    # ---- static abundance figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    ab_lims = ((-1.6, 0.05), (-0.35, 0.45))
    fig, ax = plt.subplots(figsize=(7.6, 6))
    feh_f, xfe_f = model.abundances(fiducial_theta)
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
    ax.legend(frameon=False, fontsize=9)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    ab_path = os.path.join(os.getcwd(), 'mcmc_mw_match_abundance.png')
    fig.savefig(ab_path, dpi=150)
    plt.close(fig)
    print('\nwrote {}'.format(ab_path))

    try:
        corner_path = os.path.join(os.getcwd(), 'mcmc_mw_match_corner.png')
        gizmo_mcmc.plot_corner(flat_chain, truths=None, path=corner_path)
        print('wrote {}'.format(corner_path))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    # ---- movie: walkers + simulation walking onto the MW target ----------------------------------
    if not args.no_movie:
        chain = sampler.get_chain()
        movie_path = os.path.join(os.getcwd(), 'mcmc_mw_match_walkers.mp4')
        print('rendering movie ({} frames @ {} fps)...'.format(chain.shape[0], args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, movie_path, truths=None, bounds=gizmo_mcmc.DEFAULT_BOUNDS,
                fps=args.fps, burn=0, model=model,
                data={'label': labels}, target_summary=target, abundance_lims=ab_lims,
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))


if __name__ == '__main__':
    main()
