'''
Demonstrate and validate the fast yield integrator in gizmo_mcmc, and the new
tunable "kinked" Ia model.

The fast integrator tabulates per-age-bin nucleosynthetic yields by integrating each
feedback channel's rate function *once* on a shared grid (element-independent), instead
of calling scipy.quad per element and per bin.  It works for any vectorized rate
function, so a non-standard model such as the broken-power-law "kink" Ia model is just
as fast as Maoz.

Run:
    python fast_yields_demo.py

Outputs:
    fast_yields_rate_models.png  -- the Ia rate models (Maoz, Mannucci, kinked)
    fast_yields_abundance.png    -- [alpha/Fe]-[Fe/H] for Maoz vs a few kink settings
plus a printed fast-vs-slow accuracy check and timing speedup.
'''

import os
import sys
import time

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
    age_bins = gm.default_age_bins(age_bin_number=12)
    elements = ['iron', 'magnesium', 'oxygen', 'silicon', 'calcium']
    weights, labels = gm.generate_bimodal_weights(1500, len(age_bins) - 1, seed=7)

    # ---- accuracy: fast vs slow (scipy.quad) yield tabulation ------------------------------------
    model_fast = gm.MaozElementTracerModel(age_bins, weights, xfe='alpha', fast=True)
    model_slow = gm.MaozElementTracerModel(age_bins, weights, xfe='alpha', fast=False)
    yf = model_fast.yields(gm.NIA_DEFAULT, gm.TDD_DEFAULT)
    ys = model_slow.yields(gm.NIA_DEFAULT, gm.TDD_DEFAULT)
    print('fast-vs-slow yield agreement (max relative error per element):')
    for e in elements:
        rel = np.abs(yf[e] - ys[e]) / np.maximum(np.abs(ys[e]), 1e-30)
        print('  {:>10s}: {:.2e}'.format(e, rel.max()))

    # ---- timing speedup --------------------------------------------------------------------------
    theta = (np.log10(gm.NIA_DEFAULT), gm.TDD_DEFAULT)
    n_rep = 30
    t = time.time()
    for _ in range(n_rep):
        model_slow.abundances(theta)
    t_slow = (time.time() - t) / n_rep
    t = time.time()
    for _ in range(n_rep):
        model_fast.abundances(theta)
    t_fast = (time.time() - t) / n_rep
    print('\nper forward-model evaluation:  slow {:.4f} s   fast {:.5f} s   speedup {:.0f}x'.format(
        t_slow, t_fast, t_slow / t_fast))

    # ---- plot the Ia rate models -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    ages = np.logspace(np.log10(1.0), np.log10(13700.0), 1000)
    n_ia, t_dd = gm.NIA_DEFAULT, gm.TDD_DEFAULT

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(ages, gm.ia_rate_maoz(ages, n_ia, t_dd), 'k-', lw=2, label='Maoz ($t_{dd}=-1.1$)')
    ax.loglog(ages, gm.ia_rate_mannucci(ages), color='0.5', lw=2, label='Mannucci')
    for t_kink, t_dd2, c in [(200.0, -0.5, 'crimson'), (200.0, -1.8, 'seagreen'),
                             (1000.0, -0.4, 'darkorange')]:
        ax.loglog(ages, gm.ia_rate_kink(ages, n_ia, t_dd, t_kink=t_kink, t_dd2=t_dd2),
                  lw=1.8, ls='--', color=c,
                  label='kink ($t_{{kink}}={:.0f}$, $t_{{dd2}}={:+.1f}$)'.format(t_kink, t_dd2))
    ax.axvline(gm.IA_TRANSITION_DEFAULT, color='0.8', lw=1, zorder=0)
    ax.set_xlabel('stellar age [Myr]')
    ax.set_ylabel(r'Ia mass-loss rate [$M_\odot$ / $M_\odot$ / Myr]')
    ax.set_title('Ia rate models (the kink model is a modulatable broken power law)')
    ax.set_ylim(1e-11, 1e-6)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(ls=':', alpha=0.4)
    fig.tight_layout()
    rate_path = os.path.join(os.getcwd(), 'fast_yields_rate_models.png')
    fig.savefig(rate_path, dpi=150)
    plt.close(fig)
    print('\nwrote {}'.format(rate_path))

    # ---- how the kink reshapes the abundance plane -----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    hi = labels == 1
    base = gm.MaozElementTracerModel(age_bins, weights, xfe='alpha', fast=True)
    fb, xb = base.abundances(theta)
    ax.scatter(fb[hi], xb[hi], s=7, color='lightcoral', alpha=0.4)
    ax.scatter(fb[~hi], xb[~hi], s=7, color='lightskyblue', alpha=0.4)
    ax.scatter([], [], color='0.5', label='Maoz (fiducial)')
    for t_dd2, c in [(-0.5, 'crimson'), (-1.8, 'seagreen')]:
        mk = gm.MaozElementTracerModel(age_bins, weights, xfe='alpha', fast=True,
                                       ia_model='kink', kink_params={'t_kink': 200.0, 't_dd2': t_dd2})
        fk, xk = mk.abundances(theta)
        ax.scatter(fk, xk, s=6, color=c, alpha=0.5,
                   label='kink $t_{{dd2}}={:+.1f}$'.format(t_dd2))
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[alpha/Fe]')
    ax.set_title('same weights, same n_ia & t_dd -- only the post-kink slope changes')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(ls='-.', alpha=0.4)
    fig.tight_layout()
    ab_path = os.path.join(os.getcwd(), 'fast_yields_abundance.png')
    fig.savefig(ab_path, dpi=150)
    plt.close(fig)
    print('wrote {}'.format(ab_path))


if __name__ == '__main__':
    main()
