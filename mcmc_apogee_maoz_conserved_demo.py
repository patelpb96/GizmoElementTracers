'''
Number-conserving Maoz SNe Ia parameter variation, matched to the real APOGEE DR17 Milky Way
[Mg/Fe]-[Fe/H] target.

This ties three things together:

(a) DATA  -- the real APOGEE DR17 giant-disk sample (data/apogee_dr17_stellar_labels.csv; see
    apogee_analysis.py and data/APOGEE_PROVENANCE.md) is reduced to a dual-Gaussian summary of its
    two sequences (high-alpha thick disk, low-alpha thin disk).  That summary is the MW target.

(b) INTERFACE -- the simulation abundances and the APOGEE abundances are compared on the SAME
    element ratio ([Mg/Fe] on both) and the SAME summary statistic (the dual Gaussian).  Because
    the simulation's ground-truth total number of Ia events is FIXED (see (c)), its overall [Fe/H]
    normalization is not a free knob, so the two data sets are put on a common zero-point with a
    single constant calibration offset (matched at the fiducial model).  The comparison is then a
    comparison of the bimodal *shape*, which is the physically meaningful part.

(c) MODEL -- we vary the Maoz delay-time-distribution *shape* (the power-law slope t_dd and the
    white-dwarf-formation onset t_ia) while CONSERVING the total number of Ia explosions: the
    normalization n_ia is derived at every step so the DTD integrated over [t_ia, t_hubble] stays
    equal to the fiducial event count.  This respects the simulation's ground truth that a definite
    number of Ia events actually occur -- a change of DTD shape only *redistributes* those fixed
    explosions in time, it does not create or destroy them.

Run:
    python mcmc_apogee_maoz_conserved_demo.py
    python mcmc_apogee_maoz_conserved_demo.py --no-movie --n-step 80   # quick

Outputs (--outdir, default cwd):
    apogee_maoz_conserved_abundance.png  -- APOGEE target + fiducial vs best-fit simulation
    apogee_maoz_conserved_corner.png     -- posterior of (t_dd, t_ia) at fixed event count
    apogee_maoz_conserved_walkers.mp4    -- walkers + simulation walking onto the APOGEE target
'''

import argparse
import os
import sys

import numpy as np

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

APOGEE_CSV = os.path.join(_REPO_DIR, 'data', 'apogee_dr17_stellar_labels.csv')


def _print_summary_match(target_vec, sim_vec, sigma_vec):
    print('{:>22s}  {:>8s} {:>8s} {:>8s}'.format('summary statistic', 'APOGEE', 'sim', 'resid/sig'))
    for name, t, s, sg in zip(gizmo_mcmc.BIMODAL_SUMMARY_LABELS, target_vec, sim_vec, sigma_vec):
        print('{:>22s}  {:+8.3f} {:+8.3f} {:+8.2f}'.format(name, t, s, (s - t) / sg))
    chi2 = np.sum(((sim_vec - target_vec) / sigma_vec) ** 2)
    print('  chi^2 = {:.2f}  over {} summary statistics'.format(chi2, len(target_vec)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-star', type=int, default=1200)
    parser.add_argument('--high-alpha-frac', type=float, default=0.4)
    parser.add_argument('--n-age-bin', type=int, default=10)
    parser.add_argument('--n-walker', type=int, default=48)
    parser.add_argument('--gmm-iter', type=int, default=20,
                        help='EM iterations for the per-step dual-Gaussian summary (converged by ~20)')
    parser.add_argument('--n-step', type=int, default=3000)
    parser.add_argument('--n-burn', type=int, default=1000)
    parser.add_argument('--fps', type=int, default=30)
    # animate a subset of steps so the movie is a sensible length: n_step/stride frames / fps
    # seconds (default 3000/10 = 300 frames = 10 s at 30 fps)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=110)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--outdir', default=os.getcwd())
    parser.add_argument('--no-movie', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- (a) real APOGEE DR17 dual-Gaussian target -----------------------------------------------
    target, feh_apo, mgfe_apo, info = gizmo_mcmc.apogee_target_summary(APOGEE_CSV)
    target_vec = gizmo_mcmc.summary_to_vector(target)
    sigma_vec = gizmo_mcmc.DEFAULT_SUMMARY_SIGMA
    print('APOGEE DR17 target from {} giant-disk stars (of {}):'.format(
        info['n_kept'], info['n_total']))
    print('  high-alpha: frac {:.3f}  [Fe/H] {:+.3f}  [Mg/Fe] {:+.3f}'.format(
        target['weight'][0], target['mean_feh'][0], target['mean_xfe'][0]))
    print('  low-alpha : frac {:.3f}  [Fe/H] {:+.3f}  [Mg/Fe] {:+.3f}'.format(
        target['weight'][1], target['mean_feh'][1], target['mean_xfe'][1]))

    # ---- simulated bimodal population (compared in [Mg/Fe], matching APOGEE) ----------------------
    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights, labels = gizmo_mcmc.generate_bimodal_weights(
        args.n_star, args.n_age_bin, high_alpha_frac=args.high_alpha_frac, seed=args.seed
    )

    # fiducial Maoz parameters and the ground-truth total number of Ia events to conserve
    fid_shape = np.array([gizmo_mcmc.TDD_DEFAULT, gizmo_mcmc.IA_TRANSITION_DEFAULT])  # (t_dd, t_ia)
    base = gizmo_mcmc.MaozElementTracerModel(age_bins, weights, xfe='magnesium', ia_model='maoz')
    n_events = base.dtd_event_count((np.log10(gizmo_mcmc.NIA_DEFAULT), gizmo_mcmc.TDD_DEFAULT))
    print('\n(c) conserving the total Ia event count N = {:.4e} (fiducial n_ia = {:.3e})'.format(
        n_events, gizmo_mcmc.NIA_DEFAULT))

    # ---- (b) interface: put the simulation on APOGEE's abundance zero-point -----------------------
    # match global medians at the fiducial model with a single constant (d[Fe/H], d[Mg/Fe]) offset
    sim0 = gizmo_mcmc.MaozElementTracerModel(
        age_bins, weights, xfe='magnesium', ia_model='maoz_onset',
        sampled_params=['t_dd', 't_ia'], conserve_events=n_events,
    )
    feh_sim0, mgfe_sim0 = sim0.abundances(fid_shape)
    offset = (np.median(feh_apo) - np.median(feh_sim0), np.median(mgfe_apo) - np.median(mgfe_sim0))
    print('(b) calibration offset (sim -> APOGEE zero-point): '
          'd[Fe/H] = {:+.3f}, d[Mg/Fe] = {:+.3f} dex'.format(*offset))

    # the model used for inference: number-conserving Maoz shape variation, on APOGEE's zero-point
    model = gizmo_mcmc.MaozElementTracerModel(
        age_bins, weights, xfe='magnesium', ia_model='maoz_onset',
        sampled_params=['t_dd', 't_ia'], conserve_events=n_events, abundance_offset=offset,
    )
    sampled, bounds, param_labels, _ = gizmo_mcmc.model_prior('maoz_onset', ['t_dd', 't_ia'])

    fid_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(fid_shape))
    print('\nat the fiducial DTD shape (t_dd={:+.2f}, t_ia={:.0f}):'.format(*fid_shape))
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(fid_summary), sigma_vec)

    # ---- (c) MCMC over the DTD shape at fixed event count -----------------------------------------
    print('\nrunning MCMC ({} walkers x {} steps) over (t_dd, t_ia), n_ia slaved to conserve N...'
          .format(args.n_walker, args.n_step))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, bounds=bounds, init=fid_shape,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=0.15, progress=args.progress,
        log_prob_fn=gizmo_mcmc.summary_log_probability,
        log_prob_args=(model, target_vec, sigma_vec, bounds, {'n_iter': args.gmm_iter}),
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    print('\nbest-fit DTD shape (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, labels=sampled)

    theta_med = np.median(flat_chain, axis=0)
    # verify event conservation held across the walk (ground truth preserved)
    counts = np.array([model.dtd_event_count(th) for th in flat_chain[::max(1, len(flat_chain)//200)]])
    print('\nevent-count conservation across the posterior: '
          'N = {:.4e} +/- {:.2e}  (target {:.4e}); derived n_ia at best fit = {:.3e}'.format(
              counts.mean(), counts.std(), n_events,
              model._ia_kwargs_from_params(model.params_from_theta(theta_med))[1]['n_ia']))

    best_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(theta_med))
    print('\nat the best-fit DTD shape:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(best_summary), sigma_vec)

    # ---- static abundance figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

    ab_lims = ((-1.25, 0.5), (-0.15, 0.45))
    fig, ax = plt.subplots(figsize=(7.8, 6))
    ax.scatter(feh_apo, mgfe_apo, s=5, color='0.8', alpha=0.35, label='APOGEE DR17 (data)')
    feh_f, xfe_f = model.abundances(fid_shape)
    feh_b, xfe_b = model.abundances(theta_med)
    ax.scatter(feh_f, xfe_f, s=6, color='0.55', alpha=0.4, label='simulation (fiducial)')
    ax.scatter(feh_b, xfe_b, s=6, color='0.2', alpha=0.5, label='simulation (best-fit)')
    gizmo_mcmc._draw_dual_gaussian(ax, target, n_sigma=2, ls='--', lw=2.3,
                                   label='APOGEE target (2$\\sigma$)')
    gizmo_mcmc._draw_dual_gaussian(ax, best_summary, n_sigma=2, colors=('darkred', 'navy'),
                                   ls='-', lw=1.8, label='best-fit sim (2$\\sigma$)')
    ax.set_xlim(ab_lims[0])
    ax.set_ylim(ab_lims[1])
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[Mg/Fe]')
    ax.set_title('Number-conserving Maoz variation vs APOGEE DR17')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    ab_path = os.path.join(args.outdir, 'apogee_maoz_conserved_abundance.png')
    fig.savefig(ab_path, dpi=150)
    plt.close(fig)
    print('\nwrote {}'.format(ab_path))

    try:
        corner_path = os.path.join(args.outdir, 'apogee_maoz_conserved_corner.png')
        gizmo_mcmc.plot_corner(flat_chain, truths=list(fid_shape), labels=param_labels,
                               path=corner_path)
        print('wrote {}'.format(corner_path))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    # ---- movie -----------------------------------------------------------------------------------
    if not args.no_movie:
        chain = sampler.get_chain()
        movie_path = os.path.join(args.outdir, 'apogee_maoz_conserved_walkers.mp4')
        n_frames = len(range(1, chain.shape[0] + 1, args.stride))
        print('rendering movie ({} of {} steps @ stride {} -> {:.0f} s at {} fps)...'.format(
            n_frames, chain.shape[0], args.stride, n_frames / args.fps, args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, movie_path, truths=list(fid_shape), labels=param_labels, bounds=bounds,
                fps=args.fps, burn=args.n_burn, stride=args.stride, dpi=args.dpi, model=model,
                data={'feh': feh_apo, 'xfe': mgfe_apo}, target_summary=target,
                abundance_lims=ab_lims, rate_fiducial_theta=list(fid_shape),
                rate_ylim=(1e-12, 1e-3),
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))


if __name__ == '__main__':
    main()
