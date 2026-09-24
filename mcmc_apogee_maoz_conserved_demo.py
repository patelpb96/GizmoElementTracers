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
    normalization is not a free knob, so the two data sets are related by a constant zero-point
    offset (d[Fe/H], d[Mg/Fe]).  That offset is a genuinely unknown calibration, so by default it
    is SAMPLED as a nuisance parameter (it shifts abundances only and never touches the event
    count); the comparison of the bimodal *shape* is what constrains the DTD.  To make the
    convergence visible, the walkers START with the simulation mean displaced --start-sigma MW
    standard deviations (default 1) from the APOGEE mean -- toward lower [Fe/H] and higher [Mg/Fe]
    -- and walk it back onto the data.  (The number-conserving DTD shape alone can move the
    simulation by only ~0.3 sigma across its whole prior, so it could not produce this on its own.)
    --fixed-offset recovers the 2-D DTD-only fit with the offset pinned by median matching.

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
    apogee_maoz_conserved_corner.png     -- posterior of (t_dd, t_ia, d[Fe/H], d[Mg/Fe]) at fixed event count
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
    # animate a subset of steps so the movie is a sensible length (~n_frames / fps seconds).  The
    # default 'log' schedule lingers on the early steps, where the walkers converge from the
    # displaced start, then skims the long equilibrated tail; 'linear' uses a uniform stride.
    parser.add_argument('--n-frames', type=int, default=300)
    parser.add_argument('--frame-schedule', choices=('log', 'linear'), default='log')
    parser.add_argument('--start-sigma', type=float, default=1.0,
                        help='start the simulation this many MW standard deviations from the MW '
                             'mean (toward lower [Fe/H], higher [Mg/Fe]) so convergence is visible')
    parser.add_argument('--fixed-offset', action='store_true',
                        help='hold the calibration offset fixed (2-D DTD-only fit; no displaced start)')
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
    # reference calibration: match global medians at the fiducial model with a constant
    # (d[Fe/H], d[Mg/Fe]) offset.  By default the offset is then SAMPLED as a nuisance parameter (the
    # sim-data zero-point is genuinely unknown); it shifts abundances only and leaves the Ia event
    # count untouched.  The number-conserving DTD shape alone moves the simulation by < ~0.3 sigma of
    # the MW spread across its whole prior, so the offset is what lets a displaced start converge.
    sim0 = gizmo_mcmc.MaozElementTracerModel(
        age_bins, weights, xfe='magnesium', ia_model='maoz_onset',
        sampled_params=['t_dd', 't_ia'], conserve_events=n_events,
    )
    feh_sim0, mgfe_sim0 = sim0.abundances(fid_shape)
    offset = (np.median(feh_apo) - np.median(feh_sim0), np.median(mgfe_apo) - np.median(mgfe_sim0))
    print('(b) reference calibration offset (sim -> APOGEE zero-point): '
          'd[Fe/H] = {:+.3f}, d[Mg/Fe] = {:+.3f} dex'.format(*offset))

    sampled, bounds, param_labels, _ = gizmo_mcmc.model_prior('maoz_onset', ['t_dd', 't_ia'])
    fid_theta = np.array(fid_shape)
    init = np.array(fid_shape)
    init_scale = 0.15
    sig_feh, sig_mgfe = np.std(feh_apo), np.std(mgfe_apo)
    if not args.fixed_offset:
        # 4-D: (t_dd, t_ia, d_feh, d_xfe); flat offset priors +/- 2.5 MW sigma about the reference
        sampled = sampled + ['d_feh', 'd_xfe']
        param_labels = param_labels + [r'$\Delta$[Fe/H]', r'$\Delta$[Mg/Fe]']
        bounds = np.vstack([bounds, [[offset[0] - 2.5 * sig_feh, offset[0] + 2.5 * sig_feh],
                                     [offset[1] - 2.5 * sig_mgfe, offset[1] + 2.5 * sig_mgfe]]])
        fid_theta = np.concatenate([fid_shape, offset])
        # displaced start: the simulation MEAN begins start_sigma MW-sigma below the MW mean in
        # [Fe/H] and above it in [Mg/Fe] (anchored to the mean-matched offset, so the displacement is
        # exact); the DTD walkers start spread around the fiducial shape, the offsets in a tight ball
        mean_offset = (feh_apo.mean() - feh_sim0.mean(), mgfe_apo.mean() - mgfe_sim0.mean())
        init = np.concatenate([fid_shape, [mean_offset[0] - args.start_sigma * sig_feh,
                                           mean_offset[1] + args.start_sigma * sig_mgfe]])
        init_scale = np.array([0.15, 0.15, 0.02, 0.02])

    # the model used for inference: number-conserving Maoz shape variation (+ sampled zero-point)
    model = gizmo_mcmc.MaozElementTracerModel(
        age_bins, weights, xfe='magnesium', ia_model='maoz_onset',
        sampled_params=sampled, conserve_events=n_events, abundance_offset=offset,
    )

    fid_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(fid_theta))
    print('\nat the fiducial DTD shape (t_dd={:+.2f}, t_ia={:.0f}) and reference offset:'.format(
        *fid_shape))
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(fid_summary), sigma_vec)
    if not args.fixed_offset:
        feh_s, mgfe_s = model.abundances(init)
        print('\nSTART: simulation mean displaced from the MW mean by '
              '{:+.2f} sigma in [Fe/H] and {:+.2f} sigma in [Mg/Fe]'.format(
                  (feh_s.mean() - feh_apo.mean()) / sig_feh,
                  (mgfe_s.mean() - mgfe_apo.mean()) / sig_mgfe))
        _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(
            gizmo_mcmc.fit_bimodal_gaussians(feh_s, mgfe_s)), sigma_vec)

    # ---- (c) MCMC over the DTD shape at fixed event count -----------------------------------------
    print('\nrunning MCMC ({} walkers x {} steps) over ({}), n_ia slaved to conserve N...'
          .format(args.n_walker, args.n_step, ', '.join(sampled)))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, bounds=bounds, init=init,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=init_scale, progress=args.progress,
        log_prob_fn=gizmo_mcmc.summary_log_probability,
        log_prob_args=(model, target_vec, sigma_vec, bounds, {'n_iter': args.gmm_iter}),
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    print('\nbest-fit parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, labels=sampled)

    theta_med = np.median(flat_chain, axis=0)
    # verify event conservation held across the walk (ground truth preserved)
    counts = np.array([model.dtd_event_count(th) for th in flat_chain[::max(1, len(flat_chain)//200)]])
    print('\nevent-count conservation across the posterior: '
          'N = {:.4e} +/- {:.2e}  (target {:.4e}); derived n_ia at best fit = {:.3e}'.format(
              counts.mean(), counts.std(), n_events,
              model._ia_kwargs_from_params(model.params_from_theta(theta_med))[1]['n_ia']))

    # how fast the simulation walked onto the MW: distance of the ensemble-median sim mean from the
    # MW mean (in MW sigma), versus step
    chain = sampler.get_chain()
    if not args.fixed_offset:
        print('\nconvergence of the simulation mean onto the MW mean (ensemble-median walker), in '
              'MW sigma:')
        for step in [1, 10, 30, 100, 300, 1000, args.n_step]:
            if step > args.n_step:
                continue
            f_c, x_c = model.abundances(np.median(chain[step - 1], axis=0))
            print('  step {:>5d}:  d[Fe/H] {:+.2f}   d[Mg/Fe] {:+.2f}'.format(
                step, (f_c.mean() - feh_apo.mean()) / sig_feh,
                (x_c.mean() - mgfe_apo.mean()) / sig_mgfe))

    best_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(theta_med))
    print('\nat the best fit:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(best_summary), sigma_vec)

    # ---- static abundance figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

    # wide enough to hold the displaced starting cloud as well as the MW target
    ab_lims = ((-1.6, 0.55), (-0.2, 0.65))
    fig, ax = plt.subplots(figsize=(7.8, 6))
    ax.scatter(feh_apo, mgfe_apo, s=5, color='0.8', alpha=0.35, label='APOGEE DR17 (data)')
    start_theta = init if not args.fixed_offset else fid_theta
    feh_f, xfe_f = model.abundances(start_theta)
    feh_b, xfe_b = model.abundances(theta_med)
    ax.scatter(feh_f, xfe_f, s=6, color='sandybrown', alpha=0.4,
               label='simulation (start)' if not args.fixed_offset else 'simulation (fiducial)')
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
        gizmo_mcmc.plot_corner(flat_chain, truths=list(fid_theta), labels=param_labels,
                               path=corner_path)
        print('wrote {}'.format(corner_path))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    # ---- movie -----------------------------------------------------------------------------------
    if not args.no_movie:
        movie_path = os.path.join(args.outdir, 'apogee_maoz_conserved_walkers.mp4')
        frame_steps = frame_schedule(chain.shape[0], args.n_frames, args.frame_schedule)
        print('rendering movie ({} of {} steps, {} schedule -> {:.1f} s at {} fps)...'.format(
            len(frame_steps), chain.shape[0], args.frame_schedule,
            len(frame_steps) / args.fps, args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, movie_path, truths=list(fid_theta), labels=param_labels, bounds=bounds,
                fps=args.fps, burn=args.n_burn, frame_steps=frame_steps, dpi=args.dpi,
                model=model, data={'feh': feh_apo, 'xfe': mgfe_apo}, target_summary=target,
                abundance_lims=ab_lims, rate_ylim=(1e-12, 1e-3),
                truth_label='reference (fiducial DTD)',
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))


def frame_schedule(n_step, n_frames, kind='log'):
    '''
    Chain steps (1..n_step) to render as ~n_frames movie frames.  'linear' is a uniform stride;
    'log' is geometric spacing -- every step early on, where the walkers converge, thinning out
    through the long equilibrated tail -- so the convergence is not squeezed into the first second.
    '''
    if kind == 'linear' or n_frames >= n_step:
        return np.unique(np.linspace(1, n_step, min(n_frames, n_step)).round().astype(int))
    k = n_frames
    steps = np.unique(np.geomspace(1, n_step, k).round().astype(int))
    while steps.size < n_frames:  # rounding merges early steps; oversample until we have enough
        k = int(k * 1.1) + 1
        steps = np.unique(np.geomspace(1, n_step, k).round().astype(int))
    return steps


if __name__ == '__main__':
    main()
