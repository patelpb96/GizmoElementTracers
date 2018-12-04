'''
Analyze nucleosynthetic yields in Gizmo.

@author: Andrew Wetzel

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import collections
import numpy as np
# local ----
import utilities as ut


#===================================================================================================
# nucleosynthetic yields
#===================================================================================================
def get_nucleosynthetic_yields(
    event_kind='supernova.ii', star_metallicity=1.0, normalize=True):
    '''
    Get nucleosynthetic element yields, according to input event_kind.
    Note: this only returns the *additional* nucleosynthetic yields that Gizmo adds to the
    star's existing metallicity, so these are not the actual yields that get deposited to gas.

    Parameters
    ----------
    event_kind : str : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float :
        total metallicity of star prior to event, relative to solar (sun_metal_mass_fraction)
    normalize : bool : whether to normalize yields to be mass fractions (instead of masses)

    Returns
    -------
    yields : ordered dictionary : yield mass [M_sun] or mass fraction for each element
        can covert to regular dictionary via dict(yields) or list of values via yields.values()
    '''
    sun_metal_mass_fraction = 0.02  # total metal mass fraction of sun that Gizmo assumes

    yield_dict = collections.OrderedDict()
    yield_dict['metals'] = 0.0
    yield_dict['helium'] = 0.0
    yield_dict['carbon'] = 0.0
    yield_dict['nitrogen'] = 0.0
    yield_dict['oxygen'] = 0.0
    yield_dict['neon'] = 0.0
    yield_dict['magnesium'] = 0.0
    yield_dict['silicon'] = 0.0
    yield_dict['sulphur'] = 0.0
    yield_dict['calcium'] = 0.0
    yield_dict['iron'] = 0.0

    assert event_kind in ['wind', 'supernova.ii', 'supernova.ia']

    star_metal_mass_fraction = star_metallicity * sun_metal_mass_fraction

    if event_kind == 'wind':
        # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
        # treat AGB and O-star yields in more detail for light elements
        ejecta_mass = 1.0  # these yields already are mass fractions

        yield_dict['helium'] = 0.36
        yield_dict['carbon'] = 0.016
        yield_dict['nitrogen'] = 0.0041
        yield_dict['oxygen'] = 0.0118

        # oxygen yield strongly depends on initial metallicity of star
        if star_metal_mass_fraction < 0.033:
            yield_dict['oxygen'] *= star_metal_mass_fraction / sun_metal_mass_fraction
        else:
            yield_dict['oxygen'] *= 1.65

        for k in yield_dict:
            if k is not 'helium':
                yield_dict['metals'] += yield_dict[k]

    elif event_kind == 'supernova.ii':
        # yields from Nomoto et al 2006, IMF averaged
        # rates from Starburst99
        # in Gizmo core-collapse occur 3.4 to 37.53 Myr after formation
        # from 3.4 to 10.37 Myr, rate / M_sun = 5.408e-10 yr ^ -1
        # from 10.37 to 37.53 Myr, rate / M_sun = 2.516e-10 yr ^ -1
        ejecta_mass = 10.5  # [M_sun]

        yield_dict['metals'] = 2.0
        yield_dict['helium'] = 3.87
        yield_dict['carbon'] = 0.133
        yield_dict['nitrogen'] = 0.0479
        yield_dict['oxygen'] = 1.17
        yield_dict['neon'] = 0.30
        yield_dict['magnesium'] = 0.0987
        yield_dict['silicon'] = 0.0933
        yield_dict['sulphur'] = 0.0397
        yield_dict['calcium'] = 0.00458
        yield_dict['iron'] = 0.0741

        yield_nitrogen_orig = np.float(yield_dict['nitrogen'])

        # nitrogen yield strongly depends on initial metallicity of star
        if star_metal_mass_fraction < 0.033:
            yield_dict['nitrogen'] *= star_metal_mass_fraction / sun_metal_mass_fraction
        else:
            yield_dict['nitrogen'] *= 1.65

        # correct total metal mass for nitrogen correction
        yield_dict['metals'] += yield_dict['nitrogen'] - yield_nitrogen_orig

    elif event_kind == 'supernova.ia':
        # yields from Iwamoto et al 1999, W7 model, IMF averaged
        # rates from Mannucci, Della Valle & Panagia 2006
        # in Gizmo, these occur starting 37.53 Myr after formation, with rate / M_sun =
        # 5.3e-14 + 1.6e-11 * exp(-0.5 * ((age - 0.05) / 0.01) * ((age - 0.05) / 0.01)) yr^-1
        ejecta_mass = 1.4  # [M_sun]

        yield_dict['metals'] = 1.4
        yield_dict['helium'] = 0.0
        yield_dict['carbon'] = 0.049
        yield_dict['nitrogen'] = 1.2e-6
        yield_dict['oxygen'] = 0.143
        yield_dict['neon'] = 0.0045
        yield_dict['magnesium'] = 0.0086
        yield_dict['silicon'] = 0.156
        yield_dict['sulphur'] = 0.087
        yield_dict['calcium'] = 0.012
        yield_dict['iron'] = 0.743

    if normalize:
        for k in yield_dict:
            yield_dict[k] /= ejecta_mass

    return yield_dict


def plot_nucleosynthetic_yields(
    event_kind='wind', star_metallicity=0.1, normalize=False,
    axis_y_scaling='linear', axis_y_limits=[1e-3, None],
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot nucleosynthetic element yields, according to input event_kind.

    Parameters
    ----------
    event_kind : str : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float : total metallicity of star prior to event, relative to solar
    normalize : bool : whether to normalize yields to be mass fractions (instead of masses)
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    axis_y_limits : list : min and max limits of y-axis
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    title_dict = {
        'wind': 'Stellar Wind',
        'supernova.ii': 'Supernova: Core Collapse',
        'supernova.ia': 'Supernova: Ia',
    }

    yield_dict = get_nucleosynthetic_yields(event_kind, star_metallicity, normalize)

    yield_indices = np.arange(1, len(yield_dict))
    yield_values = np.array(yield_dict.values())[yield_indices]
    yield_names = np.array(yield_dict.keys())[yield_indices]
    yield_labels = [str.capitalize(ut.constant.element_symbol_from_name[k]) for k in yield_names]
    yield_indices = np.arange(yield_indices.size)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)
    subplots = [subplot]

    colors = ut.plot.get_colors(yield_indices.size, use_black=False)

    for si in range(1):
        subplots[si].set_xlim([yield_indices.min() - 0.5, yield_indices.max() + 0.5])
        subplots[si].set_ylim(ut.plot.get_axis_limits(yield_values, axis_y_scaling, axis_y_limits))

        subplots[si].set_xticks(yield_indices)
        subplots[si].set_xticklabels(yield_labels)

        if normalize:
            y_label = 'yield (mass fraction)'
        else:
            y_label = 'yield $\\left[ {\\rm M}_\odot \\right]$'
        subplots[si].set_ylabel(y_label)
        subplots[si].set_xlabel('element')

        for yi in yield_indices:
            if yield_values[yi] > 0:
                subplot.plot(
                    yield_indices[yi], yield_values[yi], 'o', markersize=14, color=colors[yi])
                subplots[si].text(
                    yield_indices[yi] * 0.98, yield_values[yi] * 0.6, yield_labels[yi])

        subplots[si].set_title(title_dict[event_kind])

        ut.plot.make_label_legend(
            subplots[si], '$\\left[ Z / {\\rm Z}_\odot={:.3f} \\right]$'.format(star_metallicity))

    plot_name = 'element.yields_{}_Z.{:.2f}'.format(event_kind, star_metallicity)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)
