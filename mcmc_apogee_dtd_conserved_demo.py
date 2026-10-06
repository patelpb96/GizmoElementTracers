'''
Number-conserving SNe Ia delay-time-distribution (DTD) fits to the real APOGEE DR17 Milky Way
[Mg/Fe]-[Fe/H] target, for a suite of well-known DTD families plus a new peak + growth model.

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
    -- and walk it back onto the data.  (A number-conserving DTD shape alone moves the simulation
    by only ~0.3 sigma across its whole prior, so it could not produce this on its own.)
    --fixed-offset recovers the DTD-only fit with the offset pinned by median matching.

(c) MODEL -- for each DTD family we vary its *shape* while CONSERVING the total number of Ia
    explosions: every amplitude (n_ia and any bump height) is rescaled by one common factor at
    every step so the DTD integrated over [t_ia, t_hubble] equals the SAME event count for every
    family (the fiducial Maoz count).  A shape change only *redistributes* those fixed explosions
    in time; a sampled bump amplitude sets the bump's share of them.

The families (registry keys in gizmo_mcmc.IA_MODEL_SPECS; shape parameters sampled):

    maoz_onset      Maoz power law with a free onset            (t_dd, t_ia)
    maoz            Maoz ~t^-1 power law                        (t_dd)
    kink            broken power law                            (t_dd, t_kink, t_dd2)
    prompt_delayed  power law + prompt Gaussian                 (t_dd, A_p, t_p, sigma_p)
    long_delay      power law + long-delay (DD) Gaussian        (t_dd, A_L, t_L, sigma_L)
    skewnorm        Strolger (2020) skew-normal in log-age      (xi, omega, a)
    exponential     exponential DTD                             (tau)
    peak_growth     NEW: skewed Gaussian peak, then a slowly    (A_pk, sigma_pk, alpha_pk, k_grow)
                    exponentially growing tail once the peak
                    is over (peak location held at 100 Myr)

plus the two zero-point offsets for each.

Run:
    python mcmc_apogee_dtd_conserved_demo.py                       # every family, then summarize
    python mcmc_apogee_dtd_conserved_demo.py --models peak_growth  # one family
    python mcmc_apogee_dtd_conserved_demo.py --summarize           # combine existing results
    python mcmc_apogee_dtd_conserved_demo.py --models kink --no-movie --n-step 200   # quick

Outputs (--outdir, default cwd), per family <key>:
    apogee_<key>_conserved_abundance.png  -- APOGEE target + starting vs best-fit simulation
    apogee_<key>_conserved_corner.png     -- posterior of the shape parameters + offsets
    apogee_<key>_conserved_walkers.mp4    -- walkers + simulation walking onto the APOGEE target
    apogee_<key>_conserved_results.json   -- best fit, chi^2, convergence, bump share
and, from --summarize (or a multi-family run):
    apogee_dtd_suite_summary.png / .md    -- every best-fit DTD at the same event count + leaderboard
'''

import argparse
import json
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

# the suite, in order (registry key -> short title)
SUITE = [
    ('maoz_onset', 'Maoz power law + free onset'),
    ('maoz', 'Maoz power law'),
    ('kink', 'Broken power law'),
    ('prompt_delayed', 'Prompt + delayed'),
    ('long_delay', 'Power law + long-delay bump'),
    ('skewnorm', 'Skew-normal (Strolger 2020)'),
    ('exponential', 'Exponential'),
    ('peak_growth', 'Skewed peak + exponential growth'),
]
TITLES = dict(SUITE)

OFFSET_LABELS = [r'$\Delta$[Fe/H]', r'$\Delta$[Mg/Fe]']


def shape_params(key):
    '''A family's sampled shape parameters: its defaults, minus n_ia (fixed by event conservation).'''
    return [p for p in gizmo_mcmc.IA_MODEL_SPECS[key].default_sampled if p != 'log10_n_ia']


def has_bump(key):
    '''True if the family has an amplitude besides n_ia (so a share of events lives in a bump).'''
    return sum(p.amplitude for p in gizmo_mcmc.IA_MODEL_SPECS[key].params) > 1


def chi2_of(model, theta, target_vec, sigma_vec):
    summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(theta))
    return float(np.sum(((gizmo_mcmc.summary_to_vector(summary) - target_vec) / sigma_vec) ** 2))


def _print_summary_match(target_vec, sim_vec, sigma_vec):
    print('{:>22s}  {:>8s} {:>8s} {:>8s}'.format('summary statistic', 'APOGEE', 'sim', 'resid/sig'))
    for name, t, s, sg in zip(gizmo_mcmc.BIMODAL_SUMMARY_LABELS, target_vec, sim_vec, sigma_vec):
        print('{:>22s}  {:+8.3f} {:+8.3f} {:+8.2f}'.format(name, t, s, (s - t) / sg))
    chi2 = np.sum(((sim_vec - target_vec) / sigma_vec) ** 2)
    print('  chi^2 = {:.2f}  over {} summary statistics'.format(chi2, len(target_vec)))


def setup(args):
    '''Shared inputs for every family: the APOGEE target, the simulated population, and N.'''
    target, feh_apo, mgfe_apo, info = gizmo_mcmc.apogee_target_summary(APOGEE_CSV)
    print('APOGEE DR17 target from {} giant-disk stars (of {}):'.format(
        info['n_kept'], info['n_total']))
    for k, tag in [(0, 'high-alpha'), (1, 'low-alpha ')]:
        print('  {}: frac {:.3f}  [Fe/H] {:+.3f}  [Mg/Fe] {:+.3f}'.format(
            tag, target['weight'][k], target['mean_feh'][k], target['mean_xfe'][k]))

    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights, _ = gizmo_mcmc.generate_bimodal_weights(
        args.n_star, args.n_age_bin, high_alpha_frac=args.high_alpha_frac, seed=args.seed
    )
    # the ground-truth total number of Ia events, conserved identically for EVERY family: the
    # fiducial Maoz DTD integrated over [t_ia, t_hubble]
    base = gizmo_mcmc.MaozElementTracerModel(age_bins, weights, xfe='magnesium', ia_model='maoz')
    n_events = base.dtd_event_count((np.log10(gizmo_mcmc.NIA_DEFAULT), gizmo_mcmc.TDD_DEFAULT))
    print('conserving the total Ia event count N = {:.4e} for every DTD family'.format(n_events))
    return {
        'target': target, 'target_vec': gizmo_mcmc.summary_to_vector(target),
        'sigma_vec': gizmo_mcmc.DEFAULT_SUMMARY_SIGMA,
        'feh_apo': feh_apo, 'mgfe_apo': mgfe_apo,
        'age_bins': age_bins, 'weights': weights, 'n_events': n_events,
    }


def build_model(key, sampled, ctx, offset=None):
    return gizmo_mcmc.MaozElementTracerModel(
        ctx['age_bins'], ctx['weights'], xfe='magnesium', ia_model=key, sampled_params=sampled,
        conserve_events=ctx['n_events'], abundance_offset=offset,
    )


def run_model(key, ctx, args):
    '''Fit one DTD family to the APOGEE target from a displaced start; write figures/movie/JSON.'''
    title = TITLES.get(key, key)
    print('\n' + '=' * 96)
    print('{}   (registry key: {})'.format(title, key))
    print('=' * 96)
    target, target_vec, sigma_vec = ctx['target'], ctx['target_vec'], ctx['sigma_vec']
    feh_apo, mgfe_apo = ctx['feh_apo'], ctx['mgfe_apo']
    sig_feh, sig_mgfe = np.std(feh_apo), np.std(mgfe_apo)

    shapes, shape_bounds, shape_labels, fid_shape = gizmo_mcmc.model_prior(key, shape_params(key))

    # (b) reference zero-point at this family's fiducial shape (median matching)
    sim0 = build_model(key, shapes, ctx)
    feh_sim0, mgfe_sim0 = sim0.abundances(fid_shape)
    offset = (np.median(feh_apo) - np.median(feh_sim0), np.median(mgfe_apo) - np.median(mgfe_sim0))
    print('reference calibration offset: d[Fe/H] = {:+.3f}, d[Mg/Fe] = {:+.3f} dex'.format(*offset))

    sampled, bounds, param_labels = list(shapes), shape_bounds, list(shape_labels)
    fid_theta, init = np.array(fid_shape), np.array(fid_shape)
    init_scale = 0.15
    if not args.fixed_offset:
        # sample the zero-point (flat priors +/- 2.5 MW sigma about the reference) and start the
        # simulation MEAN exactly start_sigma MW-sigma below the MW mean in [Fe/H] and above it in
        # [Mg/Fe]; the shape walkers start spread around the fiducial, the offsets in a tight ball
        sampled += ['d_feh', 'd_xfe']
        param_labels += OFFSET_LABELS
        bounds = np.vstack([bounds, [[offset[0] - 2.5 * sig_feh, offset[0] + 2.5 * sig_feh],
                                     [offset[1] - 2.5 * sig_mgfe, offset[1] + 2.5 * sig_mgfe]]])
        fid_theta = np.concatenate([fid_shape, offset])
        mean_offset = (feh_apo.mean() - feh_sim0.mean(), mgfe_apo.mean() - mgfe_sim0.mean())
        init = np.concatenate([fid_shape, [mean_offset[0] - args.start_sigma * sig_feh,
                                           mean_offset[1] + args.start_sigma * sig_mgfe]])
        init_scale = np.concatenate([np.full(len(shapes), 0.15), [0.02, 0.02]])

    model = build_model(key, sampled, ctx, offset=offset)
    chi2_fid = chi2_of(model, fid_theta, target_vec, sigma_vec)
    chi2_start = chi2_of(model, init, target_vec, sigma_vec)
    feh_s, mgfe_s = model.abundances(init)
    start_sig = ((feh_s.mean() - feh_apo.mean()) / sig_feh, (mgfe_s.mean() - mgfe_apo.mean()) / sig_mgfe)
    print('chi^2 at the fiducial shape + reference offset: {:.2f}'.format(chi2_fid))
    print('START: simulation mean {:+.2f} sigma in [Fe/H], {:+.2f} sigma in [Mg/Fe] from the MW '
          'mean (chi^2 = {:.2f})'.format(start_sig[0], start_sig[1], chi2_start))

    # (c) MCMC over the shape (+ offsets) at fixed event count
    print('running MCMC ({} walkers x {} steps) over ({}) at fixed N...'.format(
        args.n_walker, args.n_step, ', '.join(sampled)))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, bounds=bounds, init=init,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=init_scale, progress=args.progress,
        log_prob_fn=gizmo_mcmc.summary_log_probability,
        log_prob_args=(model, target_vec, sigma_vec, bounds, {'n_iter': args.gmm_iter}),
    )
    acceptance = float(np.mean(sampler.acceptance_fraction))
    print('mean acceptance fraction: {:.2f}'.format(acceptance))
    print('best-fit parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, labels=sampled)
    lo, med, hi = np.percentile(flat_chain, [16, 50, 84], axis=0)

    thin = flat_chain[::max(1, len(flat_chain) // 200)]
    counts = np.array([model.dtd_event_count(th) for th in thin])
    print('event-count conservation across the posterior: N = {:.4e} +/- {:.2e} (target {:.4e})'
          .format(counts.mean(), counts.std(), ctx['n_events']))
    bump = None
    if has_bump(key):
        shares = np.array([model.dtd_component_fraction(th) for th in thin])
        bump = [float(v) for v in np.percentile(shares, [16, 50, 84])]
        print('share of the events in the bump/peak: {:.1%} (16-84%: {:.1%} - {:.1%})'.format(
            bump[1], bump[0], bump[2]))

    chain = sampler.get_chain()
    convergence = []
    if not args.fixed_offset:
        print('convergence of the simulation mean onto the MW mean (ensemble median), in MW sigma:')
        for step in [1, 10, 30, 100, 300, 1000, args.n_step]:
            if step > args.n_step:
                continue
            f_c, x_c = model.abundances(np.median(chain[step - 1], axis=0))
            d = ((f_c.mean() - feh_apo.mean()) / sig_feh, (x_c.mean() - mgfe_apo.mean()) / sig_mgfe)
            convergence.append([step, float(d[0]), float(d[1])])
            print('  step {:>5d}:  d[Fe/H] {:+.2f}   d[Mg/Fe] {:+.2f}'.format(step, *d))

    best_summary = gizmo_mcmc.fit_bimodal_gaussians(*model.abundances(med))
    chi2_best = float(np.sum(((gizmo_mcmc.summary_to_vector(best_summary) - target_vec)
                              / sigma_vec) ** 2))
    print('at the best fit:')
    _print_summary_match(target_vec, gizmo_mcmc.summary_to_vector(best_summary), sigma_vec)

    stem = os.path.join(args.outdir, 'apogee_{}_conserved'.format(key))
    result = {
        'key': key, 'title': title, 'sampled': sampled, 'labels': param_labels,
        'median': med.tolist(), 'p16': lo.tolist(), 'p84': hi.tolist(),
        'reference_theta': fid_theta.tolist(), 'start_theta': init.tolist(),
        'start_sigma': [float(v) for v in start_sig],
        'chi2_fiducial': chi2_fid, 'chi2_start': chi2_start, 'chi2_best': chi2_best,
        'acceptance': acceptance, 'n_walker': args.n_walker, 'n_step': args.n_step,
        'n_burn': args.n_burn, 'n_events': ctx['n_events'],
        'n_events_posterior': [float(counts.mean()), float(counts.std())],
        'bump_share': bump, 'convergence': convergence,
    }
    with open(stem + '_results.json', 'w') as fh:
        json.dump(result, fh, indent=1)

    # ---- static abundance figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

    ab_lims = ((-1.6, 0.55), (-0.2, 0.65))  # holds the displaced start and the MW target
    fig, ax = plt.subplots(figsize=(7.8, 6))
    ax.scatter(feh_apo, mgfe_apo, s=5, color='0.8', alpha=0.35, label='APOGEE DR17 (data)')
    feh_b, xfe_b = model.abundances(med)
    ax.scatter(feh_s, mgfe_s, s=6, color='sandybrown', alpha=0.4,
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
    ax.set_title('{} vs APOGEE DR17 (fixed Ia event count)'.format(title))
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    fig.savefig(stem + '_abundance.png', dpi=150)
    plt.close(fig)
    print('wrote {}_abundance.png'.format(stem))

    try:
        gizmo_mcmc.plot_corner(flat_chain, truths=list(fid_theta), labels=param_labels,
                               path=stem + '_corner.png')
        print('wrote {}_corner.png'.format(stem))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    # ---- movie -----------------------------------------------------------------------------------
    if not args.no_movie:
        frame_steps = frame_schedule(chain.shape[0], args.n_frames, args.frame_schedule)
        print('rendering movie ({} of {} steps, {} schedule -> {:.1f} s at {} fps)...'.format(
            len(frame_steps), chain.shape[0], args.frame_schedule,
            len(frame_steps) / args.fps, args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, stem + '_walkers.mp4', truths=list(fid_theta), labels=param_labels,
                bounds=bounds, fps=args.fps, burn=args.n_burn, frame_steps=frame_steps,
                dpi=args.dpi, model=model, data={'feh': feh_apo, 'xfe': mgfe_apo},
                target_summary=target, abundance_lims=ab_lims, rate_ylim=(1e-12, 1e-3),
                truth_label='reference (fiducial shape)', title=title,
            )
            print('wrote {}_walkers.mp4'.format(stem))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))
    return result


def summarize(keys, ctx, outdir):
    '''Leaderboard + every best-fit DTD (same event count) from the per-family JSON results.'''
    results = []
    for key in keys:
        path = os.path.join(outdir, 'apogee_{}_conserved_results.json'.format(key))
        if os.path.exists(path):
            with open(path) as fh:
                results.append(json.load(fh))
        else:
            print('(no results yet for {}: {})'.format(key, path))
    if not results:
        return
    results.sort(key=lambda r: r['chi2_best'])

    def fmt(r):
        out = []
        for name, m, lo, hi in zip(r['sampled'], r['median'], r['p16'], r['p84']):
            if name in ('d_feh', 'd_xfe'):
                continue
            out.append('{} = {:.3g} (+{:.2g}/-{:.2g})'.format(name, m, hi - m, m - lo))
        return '; '.join(out)

    lines = ['# APOGEE DR17 fits at fixed Ia event count', '',
             'Every family is fit to the same dual-Gaussian APOGEE target with the same conserved '
             'total Ia event count N = {:.4e}, starting 1 sigma off the MW mean. chi^2 is over the 9 '
             'summary statistics, evaluated at the posterior median (not the maximum-likelihood '
             'point), so differences of a few are not significant.'.format(ctx['n_events']), '',
             '| rank | DTD family | chi^2 (posterior median) | chi^2 start | bump share | acc. | '
             'posterior median (16-84%) |',
             '|---|---|---|---|---|---|---|']
    print('\n' + '=' * 72)
    print('mismatch to the APOGEE summary at the posterior median, fixed event count '
          '(lower is closer):')
    for i, r in enumerate(results, 1):
        share = '{:.1%}'.format(r['bump_share'][1]) if r.get('bump_share') else '-'
        print('  {:>2d}. {:<34s} chi^2 = {:6.2f}   (start {:6.1f})   bump share {}'.format(
            i, r['title'], r['chi2_best'], r['chi2_start'], share))
        lines.append('| {} | {} (`{}`) | {:.2f} | {:.1f} | {} | {:.2f} | {} |'.format(
            i, r['title'], r['key'], r['chi2_best'], r['chi2_start'], share, r['acceptance'],
            fmt(r)))
    with open(os.path.join(outdir, 'apogee_dtd_suite_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
    ages = np.logspace(0.0, np.log10(13700.0), 1500)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={'width_ratios': [1.6, 1]})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in enumerate(results):
        m = build_model(r['key'], r['sampled'], ctx)
        ax0.loglog(ages, np.maximum(m.ia_rate(ages, r['median']), 1e-30), color=colors[i % 10],
                   lw=2.4 if r['key'] == 'peak_growth' else 1.6, label=r['title'])
    ax0.set_ylim(1e-10, 1e-4)
    ax0.set_xlim(ages[0], ages[-1])
    ax0.set_xlabel('stellar age [Myr]')
    ax0.set_ylabel(r'Ia rate [$M_\odot$/$M_\odot$/Myr]')
    ax0.set_title('posterior-median DTDs, each with the same total Ia events')
    ax0.legend(fontsize=8, frameon=False, loc='lower left')
    ax0.grid(ls=':', alpha=0.4, which='both')
    y = np.arange(len(results))
    ax1.barh(y + 0.2, [r['chi2_start'] for r in results], height=0.4, color='sandybrown',
             label='start (1$\\sigma$ off)')
    ax1.barh(y - 0.2, [r['chi2_best'] for r in results], height=0.4, color='0.3',
             label='posterior median')
    ax1.set_yticks(y)
    ax1.set_yticklabels([r['title'] for r in results], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$\chi^2$ vs APOGEE summary (9 statistics)')
    ax1.set_title('mismatch: start vs posterior median')
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(ls=':', alpha=0.4, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'apogee_dtd_suite_summary.png'), dpi=150)
    plt.close(fig)
    print('wrote apogee_dtd_suite_summary.png / .md in {}'.format(outdir))
    if 'peak_growth' in keys:
        plot_peak_growth_shapes(ctx, outdir)


def plot_peak_growth_shapes(ctx, outdir):
    '''What each peak_growth parameter does to the DTD, every curve holding the same total events.'''
    from matplotlib import pyplot as plt
    shapes = shape_params('peak_growth')
    m = build_model('peak_growth', shapes, ctx)
    ref = dict(zip(shapes, gizmo_mcmc.model_prior('peak_growth', shapes)[3]))
    maoz = build_model('maoz', ['t_dd'], ctx)
    ages = np.logspace(np.log10(30.0), np.log10(13700.0), 4000)
    panels = [('log10_a_peak', [-6.0, -5.0, -4.5], 'peak amplitude', 'log A = {:g}'),
              ('sigma_peak', [15.0, 30.0, 80.0], 'peak width', r'$\sigma$ = {:g} Myr'),
              ('skew_peak', [-4.0, 0.0, 4.0], 'peak skew', r'$\alpha$ = {:g}'),
              ('k_grow', [0.0, 0.1, 0.3], 'tail growth exponent', 'k = {:g} / Gyr')]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    for ax, (name, values, label, fmt) in zip(axs, panels):
        for v in values:
            pars = dict(ref, **{name: v})
            theta = [pars[p] for p in shapes]
            ax.loglog(ages, np.maximum(m.ia_rate(ages, theta), 1e-30), lw=1.8,
                      label=(fmt + '  (peak share {:.0%})').format(v, m.dtd_component_fraction(theta)))
        ax.loglog(ages, maoz.ia_rate(ages, [gizmo_mcmc.TDD_DEFAULT]), 'k:', lw=1.2,
                  label='Maoz fiducial')
        ax.set_ylim(1e-9, 1e-4)
        ax.set_title(label)
        ax.set_xlabel('stellar age [Myr]')
        ax.legend(fontsize=7.5, frameon=False, loc='lower left')
        ax.grid(ls=':', alpha=0.4, which='both')
    axs[0].set_ylabel(r'Ia rate [$M_\odot$/$M_\odot$/Myr] (same total events)')
    fig.suptitle('Skewed peak + exponential growth DTD: one parameter varied per panel from the '
                 'reference shape', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'apogee_peak_growth_dtd_shapes.png'), dpi=140)
    plt.close(fig)
    print('wrote apogee_peak_growth_dtd_shapes.png in {}'.format(outdir))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=[k for k, _ in SUITE],
                        help='DTD families (registry keys) to fit; default: the whole suite')
    parser.add_argument('--summarize', action='store_true',
                        help='only combine existing per-family results into the suite summary')
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
                        help='hold the calibration offset fixed (DTD-only fit; no displaced start)')
    parser.add_argument('--dpi', type=int, default=110)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--outdir', default=os.getcwd())
    parser.add_argument('--no-movie', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    unknown = [m for m in args.models if m not in gizmo_mcmc.IA_MODEL_SPECS]
    if unknown:
        parser.error('unknown DTD family: {} (choose from {})'.format(
            ', '.join(unknown), ', '.join(k for k, _ in SUITE)))

    ctx = setup(args)
    if not args.summarize:
        for key in args.models:
            run_model(key, ctx, args)
    if args.summarize or len(args.models) > 1:
        summarize(args.models, ctx, args.outdir)


if __name__ == '__main__':
    main()
