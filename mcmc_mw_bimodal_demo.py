'''
Infer the Maoz SNe Ia model parameters that best describe a Milky-Way-like,
*bimodal* [alpha/Fe]-[Fe/H] abundance distribution, using the GizmoElementTracers
element-tracer forward model, and render a movie of the MCMC walkers converging.

Story
-----
1. Build a bimodal population of age-tracer mass weights (two star-formation
   histories) that reproduce the Milky Way's two sequences in the
   [alpha/Fe]-[Fe/H] plane: a high-alpha "thick disk" and a low-alpha "thin
   disk".  The bimodality is set entirely by the (known) star-formation
   histories; the Maoz Ia parameters are global and shared by all stars.
2. Define the "Milky Way" reference distribution as the forward model at the
   fiducial Maoz parameters, then generate the actual data set at slightly
   offset "true" parameters so that it sits ~1-2 sigma away from the MW
   reference.  This is the distribution we fit.
3. Run MCMC over (log10 n_ia, t_dd) to recover the offset "true" parameters --
   i.e. infer the model parameters that best describe the data.
4. Post-process the full chain into a 30 fps mp4 showing the walkers evolving
   (traces) together with the moving dot on the corner plot.

Run:
    python mcmc_mw_bimodal_demo.py

Outputs (written to the current directory):
    mcmc_mw_bimodal_data.png    -- the bimodal data + best-fit model
    mcmc_mw_bimodal_corner.png  -- posterior corner plot
    mcmc_mw_walkers.mp4         -- walker-evolution movie (30 fps)
'''

import argparse
import os
import sys

import numpy as np

# ---- numpy>=2 compatibility shim (see mcmc_maoz_demo.py) ------------------------------------------
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


def _population_means(feh, xfe, labels):
    '''Return {population: (mean_feh, mean_xfe)} for label values 1 (high) and 0 (low).'''
    out = {}
    for name, val in (('high-alpha', 1), ('low-alpha', 0)):
        sel = labels == val
        out[name] = (np.mean(feh[sel]), np.mean(xfe[sel]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-star', type=int, default=1500, help='number of mock stars')
    parser.add_argument('--high-alpha-frac', type=float, default=0.4,
                        help='fraction of stars on the high-alpha (thick-disk) sequence')
    parser.add_argument('--n-age-bin', type=int, default=10, help='number of age-tracer bins')
    parser.add_argument('--n-walker', type=int, default=24, help='number of MCMC walkers')
    parser.add_argument('--n-step', type=int, default=150,
                        help='MCMC steps per walker (stops a bit after convergence)')
    parser.add_argument('--n-burn', type=int, default=60, help='burn-in steps to discard')
    parser.add_argument('--true-n-ia-factor', type=float, default=1.35,
                        help='true n_ia as a multiple of the fiducial (MW) value')
    parser.add_argument('--true-t-dd', type=float, default=-1.0,
                        help='true Maoz delay-time exponent for the data')
    parser.add_argument('--sigma-feh', type=float, default=0.05,
                        help='observed standard deviation of [Fe/H] [dex]')
    parser.add_argument('--sigma-afe', type=float, default=0.03,
                        help='observed standard deviation of [alpha/Fe] [dex]')
    parser.add_argument('--fps', type=int, default=30, help='movie frame rate')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--no-movie', action='store_true', help='skip the movie')
    args = parser.parse_args()

    obs_std = (args.sigma_feh, args.sigma_afe)

    # ---- 1. bimodal age-tracer weights (two star-formation histories) ----------------------------
    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights, labels = gizmo_mcmc.generate_bimodal_weights(
        args.n_star, args.n_age_bin, high_alpha_frac=args.high_alpha_frac, seed=args.seed
    )
    model = gizmo_mcmc.MaozElementTracerModel(age_bins, weights, xfe='alpha')
    print('built {} stars ({} high-alpha, {} low-alpha) x {} age bins'.format(
        args.n_star, int(np.sum(labels == 1)), int(np.sum(labels == 0)), args.n_age_bin))

    # ---- 2. MW reference (fiducial params) and offset "true" data --------------------------------
    fiducial_theta = (np.log10(gizmo_mcmc.NIA_DEFAULT), gizmo_mcmc.TDD_DEFAULT)
    true_theta = (np.log10(gizmo_mcmc.NIA_DEFAULT * args.true_n_ia_factor), args.true_t_dd)

    feh_mw, xfe_mw = model.abundances(fiducial_theta)      # MW reference (noise-free)
    data = gizmo_mcmc.generate_mock_data(model, true_theta, obs_std=obs_std,
                                         seed=args.seed, labels=labels)

    # report how far the data sits from the MW reference, per sequence, in sigma units
    mw_means = _population_means(feh_mw, xfe_mw, labels)
    data_means = _population_means(data['feh_true'], data['xfe_true'], labels)
    print('\nfiducial (MW) parameters: n_ia = {:.3g}, t_dd = {:.3f}'.format(
        gizmo_mcmc.NIA_DEFAULT, gizmo_mcmc.TDD_DEFAULT))
    print('true (data)  parameters: n_ia = {:.3g}, t_dd = {:.3f}'.format(
        gizmo_mcmc.NIA_DEFAULT * args.true_n_ia_factor, args.true_t_dd))
    print('\noffset of the data from the MW reference (per sequence):')
    for name in ('high-alpha', 'low-alpha'):
        d_feh = data_means[name][0] - mw_means[name][0]
        d_afe = data_means[name][1] - mw_means[name][1]
        print('  {:>11s}: d[Fe/H] = {:+.3f} ({:+.1f}s),  d[alpha/Fe] = {:+.3f} ({:+.1f}s)'.format(
            name, d_feh, d_feh / obs_std[0], d_afe, d_afe / obs_std[1]))

    # ---- 3. MCMC ---------------------------------------------------------------------------------
    # deliberately start the walkers at an *off* DTD guess (not the truth) so the
    # movie shows the model abundance distribution sweep from that guess onto the
    # data as the parameters change, and the corner dot travel to the truth.
    init_guess = (np.log10(gizmo_mcmc.NIA_DEFAULT * 0.62), -1.28)
    print('\nrunning MCMC ({} walkers x {} steps)...'.format(args.n_walker, args.n_step))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, data, init=init_guess, n_walker=args.n_walker, n_step=args.n_step,
        n_burn=args.n_burn, seed=args.seed, init_dist='uniform', init_scale=0.12,
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    try:
        print('autocorrelation time (steps): {}'.format(
            np.round(sampler.get_autocorr_time(tol=0), 1)))
    except Exception:
        pass
    print('\nrecovered parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, truths=true_theta)

    # ---- 4. figures + movie ----------------------------------------------------------------------
    theta_median = np.median(flat_chain, axis=0)
    data_path = os.path.join(os.getcwd(), 'mcmc_mw_bimodal_data.png')
    gizmo_mcmc.plot_bimodal_data(data, model=model, theta=theta_median, path=data_path)
    print('\nwrote {}'.format(data_path))

    try:
        corner_path = os.path.join(os.getcwd(), 'mcmc_mw_bimodal_corner.png')
        gizmo_mcmc.plot_corner(flat_chain, truths=list(true_theta), path=corner_path)
        print('wrote {}'.format(corner_path))
    except ImportError:
        print('(install the `corner` package to get the posterior corner plot)')

    if not args.no_movie:
        chain = sampler.get_chain()  # (n_step, n_walker, ndim)
        movie_path = os.path.join(os.getcwd(), 'mcmc_mw_walkers.mp4')
        print('rendering walker movie ({} frames @ {} fps)...'.format(chain.shape[0], args.fps))
        try:
            gizmo_mcmc.animate_mcmc_walkers(
                chain, movie_path, truths=list(true_theta),
                bounds=gizmo_mcmc.DEFAULT_BOUNDS, fps=args.fps, burn=0,
                model=model, data=data,
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))


if __name__ == '__main__':
    main()
