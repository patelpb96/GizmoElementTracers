'''
End-to-end demo: recover the Maoz SNe Ia parameters (normalization n_ia and
delay-time exponent t_dd) with MCMC, using the GizmoElementTracers element-tracer
forward model.

Steps
-----
1. Build age-tracer age bins and a set of reasonable, random age-tracer mass
   weights (one row per mock star).
2. Generate a "fake" abundance data set in the [Mg/Fe]-[Fe/H] plane: run the
   element-tracer forward model at chosen "true" Maoz parameters, then scatter
   each star by a Gaussian whose width is set by the observed standard deviations.
3. Run MCMC over (log10 n_ia, t_dd) and compare the recovered posterior to the
   input truth.

Run:
    python mcmc_maoz_demo.py            # [Mg/Fe] vs [Fe/H] (fast, 2 elements)
    python mcmc_maoz_demo.py --xfe alpha   # [alpha/Fe] vs [Fe/H] (slower)

Figures are written to the current directory:
    mcmc_maoz_data_model.png   -- data + best-fit model in the abundance plane
    mcmc_maoz_corner.png       -- posterior corner plot
'''

import argparse
import os
import sys

import numpy as np

# ---- numpy>=2 compatibility shim -----------------------------------------------------------------
# Some legacy modules imported by the GizmoElementTracers package __init__ (e.g. gizmo_plot) still
# use numpy.Inf / numpy.NaN, which were removed in numpy 2.0.  Re-add them before importing the
# package so the import succeeds on newer numpy.  (No-op on numpy < 2.0.)
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# make the package importable whether this script is run from inside the repo or elsewhere
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_REPO_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

_PACKAGE_NAME = os.path.basename(_REPO_DIR)
gizmo_mcmc = __import__('{}.gizmo_mcmc'.format(_PACKAGE_NAME), fromlist=['gizmo_mcmc'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xfe', default='magnesium',
                        help="numerator for the [X/Fe] axis: 'magnesium' (default) or 'alpha'")
    parser.add_argument('--n-star', type=int, default=400, help='number of mock stars')
    parser.add_argument('--n-age-bin', type=int, default=10, help='number of age-tracer age bins')
    parser.add_argument('--n-walker', type=int, default=24, help='number of MCMC walkers')
    parser.add_argument('--n-step', type=int, default=700, help='MCMC steps per walker')
    parser.add_argument('--n-burn', type=int, default=200, help='MCMC burn-in steps to discard')
    parser.add_argument('--true-n-ia', type=float, default=gizmo_mcmc.NIA_DEFAULT,
                        help='true Maoz Ia normalization used to generate the data')
    parser.add_argument('--true-t-dd', type=float, default=gizmo_mcmc.TDD_DEFAULT,
                        help='true Maoz delay-time exponent used to generate the data')
    parser.add_argument('--sigma-feh', type=float, default=0.08,
                        help='observed standard deviation of [Fe/H] [dex]')
    parser.add_argument('--sigma-xfe', type=float, default=0.06,
                        help='observed standard deviation of [X/Fe] [dex]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no-plots', action='store_true', help='skip figure generation')
    args = parser.parse_args()

    # ---- 1. age bins and random age-tracer mass weights ------------------------------------------
    age_bins = gizmo_mcmc.default_age_bins(age_bin_number=args.n_age_bin)
    weights = gizmo_mcmc.generate_agetracer_weights(
        args.n_star, args.n_age_bin, older_bias=2.0, concentration=1.0, seed=args.seed
    )
    print('generated {} mock stars x {} age bins of age-tracer weights'.format(
        *weights.shape))

    # ---- 2. forward model + fake data set --------------------------------------------------------
    model = gizmo_mcmc.MaozElementTracerModel(age_bins, weights, xfe=args.xfe)
    true_theta = (np.log10(args.true_n_ia), args.true_t_dd)
    print('true parameters: n_ia = {:.4g}  (log10 = {:.4f}),  t_dd = {:.4f}'.format(
        args.true_n_ia, true_theta[0], true_theta[1]))

    data = gizmo_mcmc.generate_mock_data(
        model, true_theta, obs_std=(args.sigma_feh, args.sigma_xfe), seed=args.seed
    )
    print('mock data: <[Fe/H]> = {:+.3f} +/- {:.3f},  <[{}/Fe]> = {:+.3f} +/- {:.3f}'.format(
        np.mean(data['feh']), np.std(data['feh']),
        'alpha' if args.xfe == 'alpha' else args.xfe.capitalize(),
        np.mean(data['xfe']), np.std(data['xfe'])))

    # ---- 3. MCMC ---------------------------------------------------------------------------------
    print('\nrunning MCMC ({} walkers x {} steps)...'.format(args.n_walker, args.n_step))
    sampler, flat_chain = gizmo_mcmc.run_mcmc(
        model, data, init=true_theta,
        n_walker=args.n_walker, n_step=args.n_step, n_burn=args.n_burn, seed=args.seed,
    )
    try:
        tau = sampler.get_autocorr_time(tol=0)
        print('integrated autocorrelation time (steps): {}'.format(np.round(tau, 1)))
    except Exception:
        pass
    mean_accept = np.mean(sampler.acceptance_fraction)
    print('mean acceptance fraction: {:.2f}'.format(mean_accept))

    print('\nrecovered parameters (median, 16-84th percentile):')
    gizmo_mcmc.summarize_chain(flat_chain, truths=true_theta)

    # ---- 4. figures ------------------------------------------------------------------------------
    if not args.no_plots:
        theta_median = np.median(flat_chain, axis=0)
        data_path = os.path.join(os.getcwd(), 'mcmc_maoz_data_model.png')
        gizmo_mcmc.plot_data_and_model(model, data, theta_median, path=data_path)
        print('\nwrote {}'.format(data_path))
        try:
            corner_path = os.path.join(os.getcwd(), 'mcmc_maoz_corner.png')
            gizmo_mcmc.plot_corner(flat_chain, truths=list(true_theta), path=corner_path)
            print('wrote {}'.format(corner_path))
        except ImportError:
            print('(install the `corner` package to get the posterior corner plot)')


if __name__ == '__main__':
    main()
