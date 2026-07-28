'''
Sample the parameters of the "kinked" Ia delay-time distribution -- a 4-parameter
MCMC over (log10 n_ia, t_dd, t_kink, t_dd2) -- and render a movie in which the Ia
rate model itself evolves alongside the walkers.

The kinked model (gizmo_mcmc.ia_rate_kink) is a broken power law: slope t_dd until the
kink at t_kink, then slope t_dd2.  This demo generates a mock bimodal [alpha/Fe]-[Fe/H]
data set at a known "true" kink, then recovers the four parameters.  The movie's right
column shows both the evolving abundance distribution and the evolving Ia rate model
(a fixed no-kink fiducial in gray, the truth dashed, and the current-median kink in red).

Run:
    python mcmc_kink_demo.py

Outputs (current directory):
    mcmc_kink_corner.png    -- posterior over the four kink parameters
    mcmc_kink_walkers.mp4   -- walkers + evolving abundance panel + evolving Ia rate model
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
_PACKAGE = os.path.basename(_REPO_DIR)
gm = __import__('{}.gizmo_mcmc'.format(_PACKAGE), fromlist=['gizmo_mcmc'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-star', type=int, default=1500)
    parser.add_argument('--n-age-bin', type=int, default=10)
    parser.add_argument('--n-walker', type=int, default=48)
    parser.add_argument('--n-step', type=int, default=300)
    parser.add_argument('--n-burn', type=int, default=120)
    parser.add_argument('--true-t-kink', type=float, default=300.0)
    parser.add_argument('--true-t-dd2', type=float, default=-0.6)
    parser.add_argument('--sigma-feh', type=float, default=0.04)
    parser.add_argument('--sigma-afe', type=float, default=0.025)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-movie', action='store_true')
    args = parser.parse_args()

    age_bins = gm.default_age_bins(age_bin_number=args.n_age_bin)
    weights, labels = gm.generate_bimodal_weights(args.n_star, args.n_age_bin, seed=args.seed)
    model = gm.MaozElementTracerModel(
        age_bins, weights, xfe='alpha', ia_model='kink',
        sampled_params=gm.KINK_SAMPLED_PARAMS,
    )

    # true kink parameters used to generate the data
    true_theta = [np.log10(gm.NIA_DEFAULT), gm.TDD_DEFAULT, args.true_t_kink, args.true_t_dd2]
    data = gm.generate_mock_data(model, true_theta, obs_std=(args.sigma_feh, args.sigma_afe),
                                 seed=args.seed, labels=labels)
    print('true kink parameters: log10 n_ia={:.3f}, t_dd={:.2f}, t_kink={:.0f}, t_dd2={:.2f}'.format(
        *true_theta))

    # start the walkers off the truth (no-kink-ish guess) so the movie shows the kink emerge
    init = [np.log10(gm.NIA_DEFAULT * 1.3), -1.05, 800.0, -1.1]
    print('\nrunning 4-parameter MCMC ({} walkers x {} steps)...'.format(
        args.n_walker, args.n_step))
    sampler, flat_chain = gm.run_mcmc(
        model, data, bounds=gm.KINK_BOUNDS, init=init,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
        init_dist='uniform', init_scale=0.12,
    )
    print('mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))
    print('\nrecovered parameters (median, 16-84th percentile):')
    gm.summarize_chain(flat_chain, truths=true_theta,
                       labels=['log10 n_ia', 't_dd', 't_kink', 't_dd2'])

    try:
        corner_path = os.path.join(os.getcwd(), 'mcmc_kink_corner.png')
        gm.plot_corner(flat_chain, truths=list(true_theta), labels=gm.KINK_PARAM_LABELS,
                       path=corner_path)
        print('\nwrote {}'.format(corner_path))
    except ImportError:
        print('(install `corner` for the posterior corner plot)')

    if not args.no_movie:
        chain = sampler.get_chain()
        movie_path = os.path.join(os.getcwd(), 'mcmc_kink_walkers.mp4')
        # fiducial reference for the rate panel: the same n_ia/t_dd but NO kink (t_dd2 = t_dd)
        fiducial_theta = [np.log10(gm.NIA_DEFAULT), gm.TDD_DEFAULT, args.true_t_kink, gm.TDD_DEFAULT]
        print('rendering movie ({} frames @ {} fps)...'.format(chain.shape[0], args.fps))
        try:
            gm.animate_mcmc_walkers(
                chain, movie_path, truths=list(true_theta), labels=gm.KINK_PARAM_LABELS,
                bounds=gm.KINK_BOUNDS, fps=args.fps, burn=args.n_burn,
                model=model, data=data, rate_fiducial_theta=fiducial_theta,
                rate_ylim=(1e-12, 1e-3),
            )
            print('wrote {}'.format(movie_path))
        except ImportError as exc:
            print('(skipping movie: {})'.format(exc))


if __name__ == '__main__':
    main()
