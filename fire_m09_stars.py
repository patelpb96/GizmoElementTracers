'''
Load the star particles of the public FIRE-2 ultra-faint dwarf m09_res30 (Wheeler et al. 2019) at
z = 0 and plot their abundances: [Mg/Fe] and [O/Fe] vs [Fe/H], the metallicity distribution and
the formation history of the host galaxy.

The full z = 0 snapshot (snapdir_600) is 8 HDF5 files, ~16.6 GB, almost all gas and dark matter.
data/fire2_m09_res30/ holds a stars-only copy (all 1349 star particles, original Header with the
particle counts adjusted), extracted from the FIRE-2 public release with HTTP range reads, so it
reads with gizmo_io like any 1-file snapshot.  Provenance: data/fire2_m09_res30/PROVENANCE.md.

This run predates the age-tracers: Metallicity has 15 columns (11 elements + 4 r-process
scalars, all zero here) and gizmo_parameters.txt has no AgeTracer settings.  So these are the
in-code abundances, from the FIRE-2 rates and yields (Mannucci Ia DTD) used at run time.

Selection:
    host     -- stars within N_RHALF stellar half-mass radii of a shrinking-sphere centre (most
                star particles in the zoom region sit in other tiny haloes, up to ~2 Mpc away)
    floor    -- stars at the initial metallicity floor ([Fe/H] = -3.9) are counted but kept out
                of the abundance panels, where they would collapse onto one point

Run:
    python fire_m09_stars.py

Outputs:
    results/fire2_m09_res30/m09_res30_star_abundances.png
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
_package = __import__(_PACKAGE)
# the PyPI `utilities` package imports this repository under its original name
sys.modules.setdefault('gizmo_analysis', _package)
gizmo_io = __import__('{}.gizmo_io'.format(_PACKAGE), fromlist=['gizmo_io'])
gizmo_model = __import__('{}.gizmo_model'.format(_PACKAGE), fromlist=['gizmo_model'])
import utilities as ut  # noqa: E402

SIMULATION_DIRECTORY = os.path.join(_REPO_DIR, 'data', 'fire2_m09_res30')
OUTPUT_DIRECTORY = os.path.join(_REPO_DIR, 'results', 'fire2_m09_res30')
N_RHALF = 4  # host radius, in stellar half-mass radii
FLOOR_TOLERANCE = 0.01  # [Fe/H] within this of the minimum counts as the metallicity floor


def shrinking_sphere_center(position, mass, r_min=0.05, shrink=0.8, n_min=20):
    '''Mass-weighted shrinking-sphere centre [same units as position].'''
    center = np.average(position, axis=0, weights=mass)
    radius = np.sqrt(((position - center) ** 2).sum(1)).max()
    while radius > r_min:
        sel = np.sqrt(((position - center) ** 2).sum(1)) < radius
        if sel.sum() < n_min:
            break
        center = np.average(position[sel], axis=0, weights=mass[sel])
        radius *= shrink
    return center


def half_mass_radius(distance, mass):
    order = np.argsort(distance)
    cumulative = np.cumsum(mass[order])
    return distance[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]


def load_m09_stars(simulation_directory=SIMULATION_DIRECTORY, n_rhalf=N_RHALF):
    '''
    Read the m09_res30 z = 0 star particles with gizmo_io and return a dict of per-star arrays
    (distance [kpc physical], mass [Msun], age and formation time [Gyr], [Fe/H], [Mg/Fe], [O/Fe])
    plus the host/floor masks and the host centre and half-mass radius.
    '''
    part = gizmo_io.Read.read_snapshots(
        'star', 'index', 600, simulation_directory=simulation_directory, assign_hosts=False)
    star = part['star']
    position = star['position'] * part.snapshot['scalefactor']  # [kpc physical]
    mass = star['mass']
    center = shrinking_sphere_center(position, mass)
    distance = np.sqrt(((position - center) ** 2).sum(1))
    r_half = half_mass_radius(distance[distance < 5], mass[distance < 5])

    feh = star.prop('metallicity.fe')
    stars = {
        'distance': distance,
        'mass': mass,
        'age': star.prop('age'),
        'form.time': star.prop('form.time'),
        'feh': feh,
        'mgfe': star.prop('metallicity.mg - metallicity.fe'),
        'ofe': star.prop('metallicity.o - metallicity.fe'),
        'center': center,
        'r_half': r_half,
        'host': distance < n_rhalf * r_half,
        'floor': feh <= feh.min() + FLOOR_TOLERANCE,
        'feh_floor': feh.min(),
    }
    return stars


def cc_plateau(element, model='fire2.1'):
    '''[X/Fe] of pure core-collapse ejecta for the FIRE-2.1 IMF-averaged CCSN yields.'''
    cc = gizmo_model.EventYieldDictionary().event_yield_dictionaries(model)[1]
    return np.log10(cc[element] / cc['iron']) - np.log10(
        ut.constant.sun_massfraction[element] / ut.constant.sun_massfraction['iron'])


def plot_m09_stars(stars, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
    ink, muted, grid = '#1f2328', '#6e7781', '#d8dee4'
    cmap = LinearSegmentedColormap.from_list(
        'age', plt.get_cmap('Blues')(np.linspace(0.35, 1.0, 256)))

    host, floor = stars['host'], stars['floor']
    enriched = host & ~floor
    n_floor_host = int((host & floor).sum())
    age = stars['age'][enriched]
    age_lim = (np.floor(age.min() * 4) / 4, np.ceil(age.max() * 4) / 4)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax in axes.flat:
        ax.grid(color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # abundance planes, coloured by age, with the core-collapse plateau
    for ax, key, element, label in (
        (axes[0, 0], 'mgfe', 'magnesium', '[Mg/Fe]'),
        (axes[0, 1], 'ofe', 'oxygen', '[O/Fe]'),
    ):
        order = np.argsort(age)[::-1]  # oldest first, so younger (Ia-enriched) stars sit on top
        sc = ax.scatter(stars['feh'][enriched][order], stars[key][enriched][order],
                        c=age[order], cmap=cmap, vmin=age_lim[0], vmax=age_lim[1],
                        s=30, edgecolors='white', linewidths=0.6)
        plateau = cc_plateau(element)
        ax.axhline(plateau, color=muted, linestyle='--', linewidth=1.2)
        ax.text(0.02, plateau + 0.02, 'core-collapse yield ratio ({:+.2f})'.format(plateau),
                transform=ax.get_yaxis_transform(), ha='left', va='bottom', color=muted,
                fontsize=9)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel(label)
    cbar = fig.colorbar(sc, ax=axes[0, :], pad=0.02, fraction=0.03)
    cbar.set_label('stellar age [Gyr]')

    # metallicity distribution: enriched host stars, and the floor as its own bar
    ax = axes[1, 0]
    bins = np.arange(-4.0, -2.3, 0.1)
    ax.hist(stars['feh'][enriched], bins=bins, color=cmap(0.75), edgecolor='white',
            linewidth=1.0, label='enriched')
    ax.bar(stars['feh_floor'], n_floor_host, width=0.08, color='#afb8c1', edgecolor='white',
           linewidth=1.0, label='at floor')
    ax.annotate('{} stars at the\nmetallicity floor'.format(n_floor_host),
                (stars['feh_floor'], n_floor_host), xytext=(12, -4), textcoords='offset points',
                va='top', fontsize=9, color=muted)
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('number of host star particles')
    ax.legend(frameon=False, loc='upper right')

    # formation history: cumulative stellar mass formed in the host vs cosmic time
    ax = axes[1, 1]
    for sel, color, label in (
        (host & floor, '#8c959f', 'at floor'),
        (enriched, cmap(0.85), 'enriched'),
    ):
        t = np.sort(stars['form.time'][sel])
        m = stars['mass'][sel][np.argsort(stars['form.time'][sel])]
        ax.step(t, np.cumsum(m) / stars['mass'][host].sum(), where='post', color=color,
                linewidth=2, label=label)
        ax.text(t[-1] + 0.03, (np.cumsum(m) / stars['mass'][host].sum())[-1], label,
                color=ink, fontsize=9, va='center')
    ax.set_xlim(0, 3.2)
    ax.set_xlabel('cosmic time at formation [Gyr]')
    ax.set_ylabel('cumulative fraction of host stellar mass')
    ax.legend(frameon=False, loc='upper left')

    fig.suptitle(
        'FIRE-2 m09_res30, z = 0: {} host star particles within {} '
        r'$r_{{1/2}}$ = {:.2f} kpc ({:.2g} M$_\odot$)'.format(
            int(host.sum()), N_RHALF, N_RHALF * stars['r_half'], stars['mass'][host].sum()),
        fontsize=12, color=ink)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default=os.path.join(
        OUTPUT_DIRECTORY, 'm09_res30_star_abundances.png'))
    args = parser.parse_args()

    stars = load_m09_stars()
    host, floor = stars['host'], stars['floor']
    enriched = host & ~floor
    print('\nall star particles: {} ({:.3g} Msun); median distance from host {:.0f} kpc'.format(
        len(host), stars['mass'].sum(), np.median(stars['distance'])))
    print('host: r_half = {:.3f} kpc; {} stars within {} r_half ({:.3g} Msun)'.format(
        stars['r_half'], host.sum(), N_RHALF, stars['mass'][host].sum()))
    print('  at the metallicity floor ([Fe/H] = {:.2f}): {}; enriched: {}'.format(
        stars['feh_floor'], (host & floor).sum(), enriched.sum()))
    for key, label in (('feh', '[Fe/H]'), ('mgfe', '[Mg/Fe]'), ('ofe', '[O/Fe]'), ('age', 'age')):
        lo, med, hi = np.percentile(stars[key][enriched], [5, 50, 95])
        print('  enriched {:8s} median {:+.2f} (5-95%: {:+.2f} to {:+.2f})'.format(
            label, med, lo, hi))
    print('  core-collapse plateau: [Mg/Fe] = {:+.2f}, [O/Fe] = {:+.2f}'.format(
        cc_plateau('magnesium'), cc_plateau('oxygen')))

    plot_m09_stars(stars, args.output)
    print('\nwrote {}'.format(args.output))


if __name__ == '__main__':
    main()
