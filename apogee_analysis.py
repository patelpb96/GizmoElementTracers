'''
Cursory analysis of the real APOGEE DR17 [Mg/Fe]-[Fe/H] sample used as the Milky Way target.

This reads the committed APOGEE catalog (data/apogee_dr17_stellar_labels.csv), applies standard
giant-disk quality cuts, characterizes the sample, and fits the two-sequence (high-alpha /
low-alpha) dual-Gaussian summary that serves as the MW target for the delay-time-distribution
inference.  It writes a diagnostic figure and prints the target summary vector.

Data provenance: see data/APOGEE_PROVENANCE.md.  Columns are ASPCAP [Fe/H] and [Mg/Fe] (dex,
relative to solar), plus Teff and log g for quality selection.

Run:
    python apogee_analysis.py
'''

import os
import sys

import numpy as np

if not hasattr(np, 'Inf'):
    np.Inf = np.inf

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_REPO_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
_PACKAGE_NAME = os.path.basename(_REPO_DIR)
gizmo_mcmc = __import__('{}.gizmo_mcmc'.format(_PACKAGE_NAME), fromlist=['gizmo_mcmc'])

APOGEE_CSV = os.path.join(_REPO_DIR, 'data', 'apogee_dr17_stellar_labels.csv')


def main():
    print('reading APOGEE DR17 sample: {}'.format(APOGEE_CSV))
    feh, mgfe, info = gizmo_mcmc.load_apogee_disk(APOGEE_CSV)
    print('quality cuts (giant disk): {}'.format(info['cuts']))
    print('kept {} of {} stars\n'.format(info['n_kept'], info['n_total']))

    # ---- (a) cursory statistics ------------------------------------------------------------------
    print('sample statistics (clean giant-disk sample):')
    for name, a in [('[Fe/H] ', feh), ('[Mg/Fe]', mgfe)]:
        print('  {}: min {:+.2f}  p16 {:+.2f}  median {:+.2f}  p84 {:+.2f}  max {:+.2f}'.format(
            name, a.min(), np.percentile(a, 16), np.median(a), np.percentile(a, 84), a.max()))

    # bimodality check: [Mg/Fe] distribution in a mid-metallicity slice
    sl = (feh > -0.6) & (feh < -0.3)
    print('\n[Mg/Fe] in the -0.6 < [Fe/H] < -0.3 slice ({} stars) -- note the two peaks:'.format(
        int(sl.sum())))
    hist, edges = np.histogram(mgfe[sl], bins=np.linspace(-0.1, 0.45, 12))
    for i in range(len(hist)):
        bar = '#' * int(round(40 * hist[i] / hist.max()))
        print('  [{:+.2f},{:+.2f})  {:>3d}  {}'.format(edges[i], edges[i + 1], hist[i], bar))

    # ---- dual-Gaussian (high-alpha / low-alpha) target -------------------------------------------
    summary = gizmo_mcmc.fit_bimodal_gaussians(feh, mgfe)
    print('\nAPOGEE dual-Gaussian target (the MW target for the DTD inference):')
    for k, tag in [(0, 'high-alpha (thick disk)'), (1, 'low-alpha (thin disk) ')]:
        print('  {}: fraction {:.3f}  [Fe/H] {:+.3f} +/- {:.3f}   [Mg/Fe] {:+.3f} +/- {:.3f}'.format(
            tag, summary['weight'][k], summary['mean_feh'][k], summary['std_feh'][k],
            summary['mean_xfe'][k], summary['std_xfe'][k]))
    print('\n  summary vector ({}):'.format('see BIMODAL_SUMMARY_LABELS'))
    vec = gizmo_mcmc.summary_to_vector(summary)
    for name, v in zip(gizmo_mcmc.BIMODAL_SUMMARY_LABELS, vec):
        print('    {:>22s} = {:+.4f}'.format(name, v))

    # ---- diagnostic figure -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # left: the [Mg/Fe]-[Fe/H] plane with the dual-Gaussian fit
    ax0.hexbin(feh, mgfe, gridsize=45, cmap='Greys', bins='log', mincnt=1)
    gizmo_mcmc._draw_dual_gaussian(ax0, summary, n_sigma=2, colors=('firebrick', 'steelblue'),
                                   ls='-', lw=2.4, label='dual-Gaussian fit (2$\\sigma$)')
    ax0.set_xlabel('[Fe/H]')
    ax0.set_ylabel('[Mg/Fe]')
    ax0.set_title('APOGEE DR17 giant disk ({} stars)'.format(info['n_kept']))
    ax0.legend(frameon=False, fontsize=9, loc='upper right')
    ax0.grid(ls='-.', alpha=0.3)

    # right: [Mg/Fe] histogram in the mid-metallicity slice (the bimodality)
    ax1.hist(mgfe[sl], bins=np.linspace(-0.1, 0.45, 26), color='0.6', edgecolor='0.3')
    for k, c in [(0, 'firebrick'), (1, 'steelblue')]:
        ax1.axvline(summary['mean_xfe'][k], color=c, lw=2.0,
                    label='{}: {:+.2f}'.format(['high-alpha', 'low-alpha'][k],
                                               summary['mean_xfe'][k]))
    ax1.set_xlabel('[Mg/Fe]')
    ax1.set_ylabel('stars')
    ax1.set_title(r'[Mg/Fe] for $-0.6<$[Fe/H]$<-0.3$ (bimodal)')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(ls='-.', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_REPO_DIR, 'apogee_dr17_analysis.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('\nwrote {}'.format(out))


if __name__ == '__main__':
    main()
