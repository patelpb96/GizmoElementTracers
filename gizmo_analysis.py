'''
Analyze Gizmo simulations.

Masses in {M_sun}, positions in {kpc comoving}, distances in {kpc physical}.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import collections
import numpy as np
from numpy import log10, Inf  # @UnusedImport
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import colors
# local ----
import utilities as ut


#===================================================================================================
# utility
#===================================================================================================
def get_nucleosynthetic_yields(
    event_kind='supernova.ii', star_metallicity=1.0, normalize=True):
    '''
    Get nucleosynthetic element yields, according to input event_kind.
    Note: this only returns the *additional* nucleosynthetic yields that Gizmo adds to the
    star's existing metallicity, so these are not the actual yields that get deposited to gas.

    Parameters
    ----------
    event_kind : string : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float :
        total metallicity of star prior to event, relative to solar (sun_metal_mass_fraction)
    normalize : boolean : whether to normalize yields to be mass fractions (instead of masses)

    Returns
    -------
    yields : ordered dictionary : yield mass {M_sun} or mass fraction for each element
        can covert to regular dictionary via dict(yields) or list of values via yields.values()
    '''
    sun_metal_mass_fraction = 0.02  # total metal mass fraction that Gizmo assumes

    element_dict = collections.OrderedDict()
    element_dict['metal'] = 0
    element_dict['helium'] = 1
    element_dict['carbon'] = 2
    element_dict['nitrogen'] = 3
    element_dict['oxygen'] = 4
    element_dict['neon'] = 5
    element_dict['magnesium'] = 6
    element_dict['silicon'] = 7
    element_dict['sulphur'] = 8
    element_dict['calcium'] = 9
    element_dict['iron'] = 10

    yield_dict = collections.OrderedDict()
    for k in element_dict:
        yield_dict[k] = 0.0

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
                yield_dict['metal'] += yield_dict[k]

    elif event_kind == 'supernova.ii':
        # yields from Nomoto et al 2006, IMF averaged
        # rates from Starburst99
        # in Gizmo, these occur from 3.4 to 37.53 Myr after formation
        # from 3.4 to 10.37 Myr, rate / M_sun = 5.408e-10 yr ^ -1
        # from 10.37 to 37.53 Myr, rate / M_sun = 2.516e-10 yr ^ -1
        ejecta_mass = 10.5  # {M_sun}

        yield_dict['metal'] = 2.0
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
        yield_dict['metal'] += yield_dict['nitrogen'] - yield_nitrogen_orig

    elif event_kind == 'supernova.ia':
        # yields from Iwamoto et al 1999, W7 model, IMF averaged
        # rates from Mannucci, Della Valle & Panagia 2006
        # in Gizmo, these occur starting 37.53 Myr after formation, with rate / M_sun =
        # 5.3e-14 + 1.6e-11 * exp(-0.5 * ((star_age - 0.05) / 0.01) *
        #                         ((star_age - 0.05) / 0.01)) yr ^ -1
        ejecta_mass = 1.4  # {M_sun}

        yield_dict['metal'] = 1.4
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
    axis_y_scaling='lin', axis_y_limits=[1e-3, None],
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot nucleosynthetic element yields, according to input event_kind.

    Parameters
    ----------
    event_kind : string : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float : total metallicity of star prior to event, relative to solar
    normalize : boolean : whether to normalize yields to be mass fractions (instead of masses)
    axis_y_scaling : string : scaling along y-axis: 'log', 'lin'
    axis_y_limits : list : min and max limits of y-axis
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
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
    yield_labels = [ut.plot.element_name_dict[k] for k in yield_names]
    yield_indices = np.arange(yield_indices.size)

    # plot ----------
    colors = ut.plot.get_colors(yield_indices.size, use_black=False)

    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    subplots = [subplot]
    #fig, subplots = plt.subplots(3, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.94, bottom=0.14, hspace=0.03, wspace=0.03)

    for si in range(1):
        subplots[si].set_xlim([yield_indices.min() - 0.5, yield_indices.max() + 0.5])
        subplots[si].set_ylim(ut.plot.get_axis_limits(yield_values, axis_y_scaling, axis_y_limits))

        subplots[si].set_xticks(yield_indices)
        subplots[si].set_xticklabels(yield_labels)

        if normalize:
            y_label = 'yield (mass fraction)'
        else:
            y_label = 'yield $[M_\odot]$'
        subplots[si].set_ylabel(y_label, fontsize=28)
        subplots[si].set_xlabel('element', fontsize=28)
        #fig.set_ylabel(y_label, fontsize=26)
        #fig.set_xlabel('element', fontsize=26)

        plot_func = ut.plot.get_plot_function(subplots[si], 'lin', axis_y_scaling)

        for yi in yield_indices:
            if yield_values[yi] > 0:
                plot_func(yield_indices[yi], yield_values[yi], 'o', markersize=14, color=colors[yi])
                subplots[si].text(yield_indices[yi] * 0.98, yield_values[yi] * 0.6,
                                  yield_labels[yi])

        subplots[si].set_title(title_dict[event_kind])

        # metallicity legend
        legend_z = subplots[si].legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$Z/Z_\odot={:.3f}$'.format(star_metallicity)],
            loc='best', prop=FontProperties(size=16)
        )
        legend_z.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    plot_name = 'element.yields_{}_z.{:.1f}'.format(event_kind, star_metallicity)
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


class SpeciesProfileClass(ut.io.SayClass):
    '''
    Get profiles of either summation or stastitics (such as average, median) of given property for
    given particle species.
    '''
    def get_profiles(
        self, part, species=['all'], prop_name='', prop_statistic='sum', weight_by_mass=False,
        DistanceBin=None, center_position=None, center_velocity=None, rotation_vectors=None,
        axis_distance_max=Inf, other_axis_distance_max=None, other_prop_limits={},
        part_indicess=None):
        '''
        Parameters
        ----------
        part : dict : catalog of particles
        species : string or list : species to compute total mass of
        prop_name : string : name of property to get statistics of
        prop_statistic : string : statistic to get profile of
        weight_by_mass : boolean : whether to weight property by species mass
        DistanceBin : class : distance bin class
        center_position : array : position of center
        center_velocity : array : velocity of center
        axis_distance_max : float : maximum distance to use to define principal axes {kpc physical}
        rotation_vectors : array : eigen-vectors to define rotation
        other_axis_distance_max : float :
            maximum distance along other axis[s] to keep particles {kpc physical}
        other_prop_limits : dict : dictionary with properties as keys and limits as values
        part_indicess : array or list : indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if ('sum' in prop_statistic or 'vel.circ' in prop_statistic or 'density' in prop_statistic):
            return self.get_sum_profiles(
                part, species, prop_name, DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_max, other_prop_limits, part_indicess)
        else:
            return self.get_statistics_profiles(
                part, species, prop_name, weight_by_mass, DistanceBin, center_position,
                center_velocity, rotation_vectors, axis_distance_max, other_axis_distance_max,
                other_prop_limits, part_indicess)

    def get_sum_profiles(
        self, part, species=['all'], prop_name='mass', DistanceBin=None, center_position=None,
        rotation_vectors=None, axis_distance_max=Inf, other_axis_distance_max=None,
        other_prop_limits={}, part_indicess=None):
        '''
        Get profiles of summed quantity (such as mass or density) for given property for each
        particle species.

        Parameters
        ----------
        part : dict : catalog of particles
        species : string or list : species to compute total mass of
        prop_name : string : property to get sum of
        DistanceBin : class : distance bin class
        center_position : list : center position
        rotation_vectors : array : eigen-vectors to define rotation
        axis_distance_max : float : maximum distance to use to define principal axes {kpc physical}
        other_axis_distance_max : float :
            maximum distance along other axis[s] to keep particles {kpc physical}
        other_prop_limits : dict : dictionary with properties as keys and limits as values
        part_indicess : array : indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if 'gas' in species and 'consume.time' in prop_name:
            pros_mass = self.get_sum_profiles(
                part, species, 'mass', DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_max, other_prop_limits, part_indicess)

            pros_sfr = self.get_sum_profiles(
                part, species, 'sfr', DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_max, other_prop_limits, part_indicess)

            pros = pros_sfr
            for k in pros_sfr['gas']:
                if 'distance' not in k:
                    pros['gas'][k] = pros_mass['gas'][k] / pros_sfr['gas'][k] / 1e9

            return pros

        pros = {}

        Fraction = ut.math.FractionClass()

        if np.isscalar(species):
            species = [species]
        if species == ['baryon']:
            # treat this case specially for baryon fraction
            species = ['gas', 'star', 'dark', 'dark.2']
        species = ut.particle.parse_species(part, species)

        center_position = ut.particle.parse_property(part, 'position', center_position)
        part_indicess = ut.particle.parse_property(species, 'indices', part_indicess)

        assert 0 < DistanceBin.dimension_number <= 3

        for spec_i, spec_name in enumerate(species):
            part_indices = part_indicess[spec_i]
            if part_indices is None or not len(part_indices):
                part_indices = ut.array.arange_length(part[spec_name].prop(prop_name))

            if other_prop_limits:
                part_indices = ut.catalog.get_indices_catalog(
                    part[spec_name], other_prop_limits, part_indices)

            prop_values = part[spec_name].prop(prop_name, part_indices)

            if DistanceBin.dimension_number == 3:
                distances = ut.coordinate.get_distances(
                    'scalar', part[spec_name]['position'][part_indices], center_position,
                    part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}
            elif DistanceBin.dimension_number in [1, 2]:
                distancess = ut.particle.get_distances_along_principal_axes(
                    part, spec_name, '2d', center_position, rotation_vectors, axis_distance_max,
                    part_indices, scalarize=True)

                if DistanceBin.dimension_number == 1:
                    distances = distancess[0]
                    other_distances = distancess[1]
                elif DistanceBin.dimension_number == 2:
                    distances = distancess[1]
                    other_distances = distancess[0]

                if 0 < other_axis_distance_max < Inf:
                    masks = (other_distances < other_axis_distance_max)
                    distances = distances[masks]
                    prop_values = prop_values[masks]

            pros[spec_name] = DistanceBin.get_sum_profile(distances, prop_values)

        props = [pro_prop for pro_prop in pros[species[0]] if 'distance' not in pro_prop]
        props_dist = [pro_prop for pro_prop in pros[species[0]] if 'distance' in pro_prop]

        if prop_name == 'mass':
            # create dictionary for baryonic mass
            if 'star' in species or 'gas' in species:
                spec_name_new = 'baryon'
                pros[spec_name_new] = {}
                for spec_name in np.intersect1d(species, ['star', 'gas']):
                    for pro_prop in props:
                        if pro_prop not in pros[spec_name_new]:
                            pros[spec_name_new][pro_prop] = np.array(pros[spec_name][pro_prop])
                        elif 'log' in pro_prop:
                            pros[spec_name_new][pro_prop] = ut.math.get_log(
                                10 ** pros[spec_name_new][pro_prop] +
                                10 ** pros[spec_name][pro_prop])
                        else:
                            pros[spec_name_new][pro_prop] += pros[spec_name][pro_prop]

                for pro_prop in props_dist:
                    pros[spec_name_new][pro_prop] = pros[species[0]][pro_prop]
                species.append(spec_name_new)

            if len(species) > 1:
                # create dictionary for total mass
                spec_name_new = 'total'
                pros[spec_name_new] = {}
                for spec_name in np.setdiff1d(species, ['baryon', 'total']):
                    for pro_prop in props:
                        if pro_prop not in pros[spec_name_new]:
                            pros[spec_name_new][pro_prop] = np.array(pros[spec_name][pro_prop])
                        elif 'log' in pro_prop:
                            pros[spec_name_new][pro_prop] = ut.math.get_log(
                                10 ** pros[spec_name_new][pro_prop] +
                                10 ** pros[spec_name][pro_prop])
                        else:
                            pros[spec_name_new][pro_prop] += pros[spec_name][pro_prop]

                for pro_prop in props_dist:
                    pros[spec_name_new][pro_prop] = pros[species[0]][pro_prop]
                species.append(spec_name_new)

                # create mass fraction wrt total mass
                for spec_name in np.setdiff1d(species, ['total']):
                    for pro_prop in ['sum', 'sum.cum']:
                        pros[spec_name][pro_prop + '.fraction'] = Fraction.get_fraction(
                            pros[spec_name][pro_prop], pros['total'][pro_prop])

                        if spec_name == 'baryon':
                            # units of cosmic baryon fraction
                            pros[spec_name][pro_prop + '.fraction'] /= (
                                part.Cosmology['omega_baryon'] / part.Cosmology['omega_matter'])

            # create circular velocity = sqrt (G m(< r) / r)
            for spec_name in species:
                pros[spec_name]['vel.circ'] = ut.halo_property.get_circular_velocity(
                    pros[spec_name]['sum.cum'], pros[spec_name]['distance.cum'])
                #pros[spec_name]['vel.circ'] = (
                #    pros[spec_name]['sum.cum'] / pros[spec_name]['distance.cum'] *
                #    ut.const.grav_kpc_msun_yr)
                #pros[spec_name]['vel.circ'] = np.sqrt(pros[spec_name]['vel.circ'])
                #pros[spec_name]['vel.circ'] *= ut.const.km_per_kpc * ut.const.yr_per_sec

        return pros

    def get_statistics_profiles(
        self, part, species=['all'], prop_name='', weight_by_mass=True, DistanceBin=None,
        center_position=None, center_velocity=None, rotation_vectors=None, axis_distance_max=Inf,
        other_axis_distance_max=None, other_prop_limits={}, part_indicess=None):
        '''
        Get profiles of statistics (such as median, average) for given property for each
        particle species.

        Parameters
        ----------
        part : dict : catalog of particles
        species : string or list : species to compute total mass of
        prop_name : string : name of property to get statistics of
        weight_by_mass : boolean : whether to weight property by species mass
        DistanceBin : class : distance bin class
        center_position : array : position of center
        center_velocity : array : velocity of center
        axis_distance_max : float : maximum distance to use to define principal axes {kpc physical}
        rotation_vectors : array : eigen-vectors to define rotation
        other_axis_distance_max : float :
            maximum distance along other axis[s] to keep particles {kpc physical}
        other_prop_limits : dict : dictionary with properties as keys and limits as values
        part_indicess : array or list : indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        pros = {}

        species = ut.particle.parse_species(part, species)

        center_position = ut.particle.parse_property(part, 'position', center_position)
        part_indicess = ut.particle.parse_property(species, 'indices', part_indicess)
        if 'velocity' in prop_name:
            center_velocity = ut.particle.parse_property(part, 'velocity', center_velocity)

        assert 0 < DistanceBin.dimension_number <= 3

        for spec_i, spec_name in enumerate(species):
            part_indices = part_indicess[spec_i]
            if part_indices is None or not len(part_indices):
                part_indices = ut.array.arange_length(part[spec_name].prop(prop_name))

            if other_prop_limits:
                part_indices = ut.catalog.get_indices_catalog(
                    part[spec_name], other_prop_limits, part_indices)

            masses = None
            if weight_by_mass:
                masses = part[spec_name].prop('mass', part_indices)

            if 'velocity' in prop_name:
                distances = ut.coordinate.get_distances(
                    'vector', part[spec_name]['position'][part_indices], center_position,
                    part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}

                velocity_vectors = ut.coordinate.get_velocity_differences(
                    'vector', part[spec_name]['velocity'][part_indices], center_velocity, True,
                    part[spec_name]['position'][part_indices], center_position,
                    part.snapshot['scalefactor'], part.snapshot['time.hubble'],
                    part.info['box.length'])

                pro = DistanceBin.get_velocity_profile(distances, velocity_vectors, masses)

                pros[spec_name] = pro[prop_name.replace('host.', '')]
                for prop in pro:
                    if 'velocity' not in prop:
                        pros[spec_name][prop] = pro[prop]
            else:
                prop_values = part[spec_name].prop(prop_name, part_indices)

                if DistanceBin.dimension_number == 3:
                    distances = ut.coordinate.get_distances(
                        'scalar', part[spec_name]['position'][part_indices], center_position,
                        part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}

                elif DistanceBin.dimension_number in [1, 2]:
                    distancess = ut.particle.get_distances_along_principal_axes(
                        part, spec_name, '2d', center_position, rotation_vectors, axis_distance_max,
                        part_indices, scalarize=True)

                    if DistanceBin.dimension_number == 1:
                        distances = distancess[0]
                        other_distances = distancess[1]
                    elif DistanceBin.dimension_number == 2:
                        distances = distancess[1]
                        other_distances = distancess[0]

                    if 0 < other_axis_distance_max < Inf:
                        masks = (other_distances < other_axis_distance_max)
                        distances = distances[masks]
                        masses = masses[masks]
                        prop_values = prop_values[masks]

                pros[spec_name] = DistanceBin.get_statistics_profile(distances, prop_values, masses)

        return pros


#===================================================================================================
# diagnostic
#===================================================================================================
def plot_mass_contamination(
    part, distance_limits=[1, 2000], distance_bin_width=0.02, distance_bin_number=None,
    distance_scaling='log', halo_radius=None, scale_to_halo_radius=False, center_position=None,
    axis_y_scaling='log', write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot contamination from lower-resolution particles v distance from center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    distance_limits : list : min and max limits for distance from galaxy
    distance_bin_width : float : width of each distance bin (in units of distance_scaling)
    distance_bin_number : int : number of distance bins
    distance_scaling : string : 'log', 'lin'
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_position : array : position of galaxy/halo center
    axis_y_scaling : string : scaling of y-axis: 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    species_test = ['dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6', 'gas', 'star']
    species_reference = 'dark'

    virial_kind = '200m'

    Say = ut.io.SayClass(plot_mass_contamination)

    species_test = ut.particle.parse_species(part, species_test)
    center_position = ut.particle.parse_property(part, 'position', center_position)

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, distance_bin_width, distance_bin_number)

    pros = collections.OrderedDict()
    pros[species_reference] = {}
    for spec_name in species_test:
        pros[spec_name] = {}

    ratios = {}

    for spec_name in pros:
        distances = ut.coordinate.get_distances(
            'scalar', part[spec_name]['position'], center_position, part.info['box.length'])
        distances *= part.snapshot['scalefactor']  # convert to {kpc physical}
        if scale_to_halo_radius:
            distances /= halo_radius
        pros[spec_name] = DistanceBin.get_sum_profile(distances, part[spec_name]['mass'])

    for spec_name in species_test:
        mass_ratio_bin = pros[spec_name]['sum'] / pros[species_reference]['sum']
        mass_ratio_cum = pros[spec_name]['sum.cum'] / pros[species_reference]['sum.cum']
        ratios[spec_name] = {'bin': mass_ratio_bin, 'cum': mass_ratio_cum}

    # print diagnostics
    for spec_name in ['dark.2', 'dark.3']:
        Say.say(spec_name + ' cumulative mass / mass_fraction / number:')
        if pros[spec_name]['sum.cum'][-1] == 0:
            Say.say('  none. yay.')
            continue
        distances = pros[spec_name]['distance.cum']

        if scale_to_halo_radius:
            print_string = '  d/R_halo < {:.3f}: '
        else:
            print_string = '  d < {:.3f} kpc: '
        print_string += 'mass = {:.2e}, mass_frac = {:.3f}, number = {:.0f}'

        for dist_i in range(pros[spec_name]['sum.cum'].size):
            if pros[spec_name]['sum.cum'][dist_i] > 0:
                Say.say(print_string.format(
                        distances[dist_i], pros[spec_name]['sum.cum'][dist_i],
                        ratios[spec_name]['cum'][dist_i],
                        pros[spec_name]['sum.cum'][dist_i] / part[spec_name]['mass'][0]))

    if write_plot is None:
        return

    # plot ----------
    colors = ut.plot.get_colors(len(species_test), use_black=False)

    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03, wspace=0.03)

    subplot.set_xlim(distance_limits)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim([0.0001, 3])

    subplot.set_ylabel(
        '$M_{{\\rm species}} / M_{{\\rm {}}}$'.format(species_reference), fontsize=30)
    if scale_to_halo_radius:
        axis_x_label = '$d \, / \, R_{{\\rm {}}}$'.format(virial_kind)
    else:
        axis_x_label = 'distance $[\\rm kpc]$'
    subplot.set_xlabel(axis_x_label, fontsize=30)

    plot_func = ut.plot.get_plot_function(subplot, distance_scaling, axis_y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    for spec_i, spec_name in enumerate(species_test):
        plot_func(
            DistanceBin.mids, ratios[spec_name]['bin'], color=colors[spec_i], alpha=0.7,
            label=spec_name)

    legend = subplot.legend(loc='best', prop=FontProperties(size=16))
    legend.get_frame().set_alpha(0.7)

    # plt.tight_layout(pad=0.02)

    distance_name = 'dist'
    if halo_radius and scale_to_halo_radius:
        distance_name += '.' + virial_kind
    plot_name = 'mass.ratio_v_{}_z.{:.1f}'.format(distance_name, part.snapshot['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_metal_v_distance(
    part, spec_name='gas',
    distance_limits=[10, 3000], distance_bin_width=0.1, distance_bin_number=None,
    distance_scaling='log',
    halo_radius=None, scale_to_halo_radius=False, center_position=None,
    plot_kind='metallicity', axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot metallicity (in bin or cumulative) of gas or stars v distance from galaxy.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    spec_name : string : particle species
    distance_limits : list : min and max limits for distance from galaxy
    distance_bin_width : float : width of each distance bin (in units of distance_scaling)
    distance_bin_number : int : number of distance bins
    distance_scaling : string : scaling of distance: 'log', 'lin'
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_position : array : position of galaxy center {kpc comoving}
    plot_kind : string : metallicity or metal.mass.cum
    axis_y_scaling : string : scaling of y-axis: 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    virial_kind = '200m'

    center_position = ut.particle.parse_property(part, 'position', center_position)

    distance_limits_use = np.array(distance_limits)
    if halo_radius and scale_to_halo_radius:
        distance_limits_use *= halo_radius

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits_use, distance_bin_width, distance_bin_number)

    distances = ut.coordinate.get_distances(
        'scalar', part[spec_name]['position'], center_position, part.info['box.length'])
    distances *= part.snapshot['scalefactor']  # convert to {kpc physical}

    metal_masses = part[spec_name].prop('metallicity.total.solar') * part[spec_name]['mass']

    pro_metal = DistanceBin.get_sum_profile(distances, metal_masses, get_fraction=True)
    if 'metallicity' in plot_kind:
        pro_mass = DistanceBin.get_sum_profile(distances, part[spec_name]['mass'])
        ys = pro_metal['sum'] / pro_mass['sum']
        axis_y_limits = np.clip(ut.plot.get_axis_limits(ys), 0.0001, 10)
    elif 'metal.mass.cum' in plot_kind:
        ys = pro_metal['fraction.cum']
        axis_y_limits = [0.001, 1]

    # plot ----------
    # colors = ut.plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03, wspace=0.03)

    subplot.set_xlim(distance_limits)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim(axis_y_limits)

    if 'metallicity' in plot_kind:
        subplot.set_ylabel('$Z \, / \, Z_\odot$', fontsize=30)
    elif 'metal.mass.cum' in plot_kind:
        subplot.set_ylabel('$M_{\\rm Z}(< r) \, / \, M_{\\rm Z,tot}$', fontsize=30)

    if scale_to_halo_radius:
        axis_x_label = '$d \, / \, R_{{\\rm {}}}$'.format(virial_kind)
    else:
        #axis_x_label = 'distance $[\\rm kpc\,physical]$'
        axis_x_label = 'distance $[\\mathrm{kpc}]$'

    subplot.set_xlabel(axis_x_label, fontsize=26)

    plot_func = ut.plot.get_plot_function(subplot, distance_scaling, axis_y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    plot_func(xs, ys, color='blue', alpha=0.6)

    # legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    # legend.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    distance_name = 'dist'
    if halo_radius and scale_to_halo_radius:
        distance_name += '.' + virial_kind
    plot_name = plot_kind + '_v_' + distance_name + '_z.{:.1f}'.format(part.info['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def test_image(position_limits=[[0., 1.], [0., 1.1]]):

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(1)
    fig.clf()

    subplot = fig.add_subplot(111)
    subplot.axis('equal')
    #fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03, wspace=0.03)

    subplot.set_xlim(position_limits[0])
    subplot.set_ylim(position_limits[1])

    circle1 = plt.Circle((0.5, 0.5), 0.4, color='b', fill=False)
    fig.gca().add_artist(circle1)


#===================================================================================================
# visualize
#===================================================================================================
def plot_image(
    part, spec_name='dark', dimen_indices_plot=[0, 1], dimen_indices_select=[0, 1, 2],
    distance_max=1000, distance_bin_width=1, distance_bin_number=None, center_position=None,
    weight_prop_name='mass', other_prop_limits={}, part_indices=None, subsample_factor=None,
    align_principal_axes=False, image_limits=[None, None], background_color='white',
    hal=None, hal_indices=None, hal_position_kind='position', hal_radius_kind='radius',
    write_plot=False, plot_directory='.', add_image_limits=True, add_simulation_name=False,
    figure_index=1):
    '''
    Visualize the positions of given partcle species, using either a single panel for 2 axes or
    3 panels for all axes.

    Parameters
    ----------
    part : dict : catalog of particles
    spec_name : string : particle species to plot
    dimen_indices_plot : list : which dimensions to plot
        if length 2, plot one v other, if length 3, plot all via 3 panels
    dimen_indices_select : list : which dimensions to use to select particles
        note : use this to set selection 'depth' of an image
    distance_max : float : distance from center to plot
    distance_bin_width : float : length pixel
    distance_bin_number : number of pixels from distance = 0 to max (2x this across image)
    center_position : array-like : position of center
    weight_prop_name : string : property to weight positions by
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indices : array : input selection indices for particles
    subsample_factor : int : factor by which periodically to sub-sample particles
    align_principal_axes : boolean : whether to align positions with principal axes
    image_limits : list : min and max limits to impose on image dynamic range (exposure)
    background_color : string : name of color for background: 'white', 'black'
    hal : dict : catalog of halos at snapshot
    hal_indices : array : indices of halos to plot
    hal_position_kind : string : name of position to use for center of halo
    hal_radius_kind : string : name of radius to use for size of halo
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    add_image_limits : boolean : add range of image to file name
    add_simulation_name : boolean : whether to add name of simulation to figure name
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_image)

    dimen_label = {0: 'x', 1: 'y', 2: 'z'}

    if dimen_indices_select is None or not len(dimen_indices_select):
        dimen_indices_select = dimen_indices_plot

    distance_max *= 1.0

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.arange_length(part[spec_name]['position'].shape[0])

    if other_prop_limits:
        part_indices = ut.catalog.get_indices_catalog(
            part[spec_name], other_prop_limits, part_indices)

    if subsample_factor is not None and subsample_factor > 1:
        part_indices = part_indices[::subsample_factor]

    positions = np.array(part[spec_name]['position'][part_indices])
    if weight_prop_name:
        weights = part[spec_name].prop(weight_prop_name, part_indices)
    else:
        weights = None

    center_position = ut.particle.parse_property(part, 'position', center_position)

    if center_position is not None and len(center_position):
        # re-orient to input center
        positions -= center_position
        positions *= part.snapshot['scalefactor']

        masks = (positions[:, dimen_indices_select[0]] <= distance_max)  # initialize masks
        for dimen_i in dimen_indices_select:
            masks *= ((positions[:, dimen_i] <= distance_max) *
                      (positions[:, dimen_i] >= -distance_max))

        positions = positions[masks]
        if weights is not None:
            weights = weights[masks]

        if align_principal_axes:
            eigen_vectors = ut.coordinate.get_principal_axes(positions, weights)[0]
            positions = ut.coordinate.get_coordinates_rotated(positions, eigen_vectors)
    else:
        distance_max = 0.5 * np.max(np.max(positions, 0) - np.min(positions, 0))

    if distance_bin_width is not None and distance_bin_width > 0:
        position_bin_number = int(np.round(2 * distance_max / distance_bin_width))
    elif distance_bin_number is not None and distance_bin_number > 0:
        position_bin_number = 2 * distance_bin_number
    else:
        raise ValueError('need to input either distance bin width or bin number')

    position_limits = np.array([[-distance_max, distance_max], [-distance_max, distance_max]])

    if hal is not None:
        # compile halos
        if hal_indices is None or not len(hal_indices):
            hal_indices = ut.array.arange_length(hal['total.mass'])

        if 0 not in hal_indices:
            hal_indices = np.concatenate([[0], hal_indices])

        hal_positions = np.array(hal[hal_position_kind][hal_indices])
        if center_position is not None and len(center_position):
            hal_positions -= center_position
        hal_positions *= hal.snapshot['scalefactor']
        hal_radiuss = hal[hal_radius_kind][hal_indices]

        masks = (hal_positions[:, dimen_indices_select[0]] <= distance_max)  # initialize masks
        for dimen_i in dimen_indices_select:
            masks *= ((hal_positions[:, dimen_i] <= distance_max) *
                      (hal_positions[:, dimen_i] >= -distance_max))

        hal_radiuss = hal_radiuss[masks]
        hal_positions = hal_positions[masks]
        hal_positions = hal_positions

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()

    BYW = colors.LinearSegmentedColormap('byw', ut.plot.cmap_dict['BlackYellowWhite'])
    plt.register_cmap(cmap=BYW)
    BBW = colors.LinearSegmentedColormap('bbw', ut.plot.cmap_dict['BlackBlueWhite'])
    plt.register_cmap(cmap=BBW)

    if background_color == 'black':
        if spec_name == 'dark':
            color_map = plt.get_cmap('bbw')
        elif spec_name == 'star':
            color_map = plt.get_cmap('byw')
    elif background_color == 'white':
        color_map = plt.cm.YlOrBr  # @UndefinedVariable
        #color_map = plt.cm.Greys_r,  # @UndefinedVariable

    if len(dimen_indices_plot) == 2:
        subplot = fig.add_subplot(111, axisbg=background_color)
        fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03, wspace=0.03)

        subplot.set_xlim(position_limits[0])
        subplot.set_ylim(position_limits[1])

        subplot.set_xlabel('{} $[\\rm kpc]$'.format(dimen_label[dimen_indices_plot[0]]))
        subplot.set_ylabel('{} $[\\rm kpc]$'.format(dimen_label[dimen_indices_plot[1]]))

        # smooth image
        histogramss, xs, ys = np.histogram2d(
            positions[:, dimen_indices_plot[0]], positions[:, dimen_indices_plot[1]],
            position_bin_number, position_limits,
            normed=False,
            weights=weights,
        )

        # convert to surface density
        histogramss /= np.diff(xs)[0] * np.diff(ys)[0]

        masks = (histogramss > 0)
        Say.say('histogram min, med, max = {:.3e}, {:.3e}, {:.3e}'.format(
                histogramss[masks].min(), np.median(histogramss[masks]), histogramss[masks].max()))

        image_limits_use = [histogramss[masks].min(), histogramss[masks].max()]
        if image_limits is not None and len(image_limits):
            if image_limits[0] is not None:
                image_limits_use[0] = image_limits[0]
            if image_limits[1] is not None:
                image_limits_use[1] = image_limits[1]

        subplot.imshow(
            histogramss.transpose(),
            norm=colors.LogNorm(),
            cmap=color_map,
            aspect='auto',
            interpolation='none',
            #interpolation='bilinear',
            #interpolation='bicubic',
            #interpolation='gaussian',
            extent=np.concatenate(position_limits),
            vmin=image_limits[0], vmax=image_limits[1],
        )

        # standard method
        """
        _histogramss, _xs, _ys, _Image = subplot.hist2d(
            positions[:, dimen_indices_plot[0]], positions[:, dimen_indices_plot[1]],
            weights=weights, range=position_limits, bins=position_bin_number,
            norm=colors.LogNorm(),
            #cmap=plt.cm.YlOrBr,  # @UndefinedVariable
            cmap=plt.get_cmap('test'),
            vmin=image_limits[0], vmax=image_limits[1],
        )
        """

        # plot average of property
        """
        histogramss, xs, ys = np.histogram2d(
            positions[:, dimen_indices_plot[0]], positions[:, dimen_indices_plot[1]],
            position_bin_number, position_limits, weights=None,
            normed=False)

        #histogramss = ut.math.Fraction.get_fraction(weight_grid, grid_number)
        subplot.imshow(
            histogramss.transpose(),
            #norm=colors.LogNorm(),
            cmap=plt.cm.YlOrBr,  # @UndefinedVariable
            aspect='auto',
            #interpolation='nearest',
            interpolation='none',
            extent=np.concatenate(position_limits),
            vmin=np.min(weights), vmax=np.max(weights),
        )
        """

        #fig.colorbar(_Image)

        # plot halos
        if hal is not None:
            for hal_position, hal_radius in zip(hal_positions, hal_radiuss):
                print(hal_position, hal_radius)
                circle = plt.Circle(
                    hal_position[dimen_indices_plot], hal_radius, color='w', linewidth=1,
                    fill=False)
                subplot.add_artist(circle)

        fig.gca().set_aspect('equal')

    elif len(dimen_indices_plot) == 3:
        #position_limits *= 0.999  # ensure that tick labels do not overlap
        position_limits[0, 0] *= 0.994
        position_limits[1, 0] *= 0.994

        fig, subplots = plt.subplots(2, 2, num=figure_index, sharex=True, sharey=True,
                                     axisbg=background_color)
        fig.subplots_adjust(left=0.17, right=0.96, top=0.97, bottom=0.13, hspace=0.03, wspace=0.03)

        fig.gca().set_aspect('equal')

        plot_dimen_iss = [
            [dimen_indices_plot[0], dimen_indices_plot[1]],
            [dimen_indices_plot[0], dimen_indices_plot[2]],
            [dimen_indices_plot[1], dimen_indices_plot[2]],
        ]

        subplot_iss = [
            [0, 0],
            [1, 0],
            [1, 1],
        ]

        for plot_i, plot_dimen_is in enumerate(plot_dimen_iss):
            subplot_is = subplot_iss[plot_i]
            subplot = subplots[subplot_is[0], subplot_is[1]]

            subplot.set_xlim(position_limits[0])
            subplot.set_ylim(position_limits[1])

            if subplot_is == [0, 0]:
                subplot.set_ylabel(dimen_label[plot_dimen_is[1]] + ' $[\\rm kpc]$')
            elif subplot_is == [1, 0]:
                subplot.set_xlabel(dimen_label[plot_dimen_is[0]] + ' $[\\rm kpc]$')
                subplot.set_ylabel(dimen_label[plot_dimen_is[1]] + ' $[\\rm kpc]$')
            elif subplot_is == [1, 1]:
                subplot.set_xlabel(dimen_label[plot_dimen_is[0]] + ' $[\\rm kpc]$')

            histogramss, xs, ys = np.histogram2d(
                positions[:, plot_dimen_is[0]], positions[:, plot_dimen_is[1]],
                position_bin_number, position_limits,
                normed=False,
                weights=weights,
            )

            # convert to surface density
            histogramss /= np.diff(xs)[0] * np.diff(ys)[0]

            masks = (histogramss > 0)
            Say.say('histogram min, med, max = {:.3e}, {:.3e}, {:.3e}'.format(
                    histogramss[masks].min(), np.median(histogramss[masks]),
                    histogramss[masks].max()))

            image_limits_use = np.array([histogramss[masks].min(), histogramss[masks].max()])
            if image_limits is not None and len(image_limits):
                if image_limits[0] is not None:
                    image_limits_use[0] = image_limits[0]
                if image_limits[1] is not None:
                    image_limits_use[1] = image_limits[1]

            subplot.imshow(
                histogramss.transpose(),
                norm=colors.LogNorm(),
                cmap=plt.cm.YlOrBr,  # @UndefinedVariable
                #aspect='auto',
                interpolation='nearest',
                #interpolation='bilinear',
                #interpolation='bicubic',
                #interpolation='gaussian',
                extent=np.concatenate(position_limits),
                vmin=image_limits[0], vmax=image_limits[1],
            )

            # default method
            """
            histogramss, _xs, _ys, _Image = subplot.hist2d(
                positions[:, plot_dimen_is[0]], positions[:, plot_dimen_is[1]], weights=weights,
                range=position_limits, bins=position_bin_number, norm=colors.LogNorm(),
                cmap=plt.cm.YlOrBr)  # @UndefinedVariable
            """

            #fig.colorbar(_Image)  # , ax=subplot)

            # plot halos
            if hal is not None:
                for hal_position, hal_radius in zip(hal_positions, hal_radiuss):
                    circle = plt.Circle(
                        hal_position[plot_dimen_is], hal_radius, linewidth=1, fill=False)
                    subplot.add_artist(circle)

                circle = plt.Circle((0, 0), 10, color='b', fill=False)
                subplot.add_artist(circle)

            subplot.axis('equal')

    #plt.tight_layout(pad=0.02)

    plot_name = spec_name + '.position'
    for dimen_i in dimen_indices_plot:
        plot_name += '.' + dimen_label[dimen_i]
    plot_name += '_d.{:.0f}_z.{:0.1f}'.format(distance_max, part.snapshot['redshift'])

    if add_image_limits:
        plot_name += '_i.{:.1f}-{:.1f}'.format(
            log10(image_limits_use[0]), log10(image_limits_use[1]))
        #plot_name += '_i.{}-{}'.format(
        #    ut.io.get_string_for_exponential(image_limits_use[0], 0),
        #    ut.io.get_string_for_exponential(image_limits_use[1], 0))

    if add_simulation_name:
        plot_name = part.info['simulation.name'].replace(' ', '.') + '_' + plot_name

    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# general property analysis
#===================================================================================================
def plot_property_distribution(
    parts, spec_name='gas', prop_name='density', prop_limits=[], prop_bin_width=None,
    prop_bin_number=100, prop_scaling='log', prop_statistic='probability',
    distance_limits=[], center_positions=None, center_velocities=None,
    other_prop_limits={}, part_indicess=None,
    axis_y_limits=[], axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot distribution of property.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    spec_name : string : particle species
    prop_name : string : property name
    prop_limits : list : min and max limits of property
    prop_bin_width : float : width of property bin (use this or prop_bin_number)
    prop_bin_number : int : number of property bins within limits (use this or prop_bin_width)
    prop_scaling : string : scaling of property: 'log', 'lin'
    prop_statistic : string : statistic to plot: 'probability', 'histogram'
    distance_limits : list : min and max limits for distance from galaxy
    center_positions : array or list of arrays : position[s] of galaxy center[s]
    center_velocities : array or list of arrays : velocity[s] of galaxy center[s]
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_property_distribution)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions)
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)
    if 'velocity' in prop_name:
        center_velocities = ut.particle.parse_property(parts, 'velocity', center_velocities)

    Stat = ut.statistic.StatisticClass()

    for part_i, part in enumerate(parts):
        if part_indicess[part_i] is not None and len(part_indicess[part_i]):
            part_indices = part_indicess[part_i]
        else:
            part_indices = ut.array.arange_length(part[spec_name]['position'].shape[0])

        if other_prop_limits:
            part_indices = ut.catalog.get_indices_catalog(
                part[spec_name], other_prop_limits, part_indices)

        if distance_limits:
            distances = ut.coordinate.get_distances(
                'scalar', part[spec_name]['position'][part_indices], center_positions[part_i],
                part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}
            part_indices = part_indices[ut.array.get_indices(distances, distance_limits)]

        if 'velocity' in prop_name:
            orb = ut.particle.get_orbit_dictionary(
                part, spec_name, center_positions[part_i], center_velocities[part_i], part_indices,
                include_hubble_flow=True, scalarize=True)
            prop_values = orb[prop_name]
        else:
            prop_values = part[spec_name].prop(prop_name, part_indices)

        Say.say('keeping {} {} particles'.format(prop_values.size, spec_name))

        Stat.append_to_dictionary(
            prop_values, prop_limits, prop_bin_width, prop_bin_number, prop_scaling)

        Stat.print_statistics(-1)
        print()

    colors = ut.plot.get_colors(len(parts))

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03, wspace=0.03)

    subplot.set_xlim(ut.plot.get_axis_limits(prop_values, prop_scaling, prop_limits))

    y_values = [Stat.distr[prop_statistic][part_i] for part_i in range(len(parts))]
    subplot.set_ylim(ut.plot.get_axis_limits(y_values, axis_y_scaling, axis_y_limits))

    subplot.set_xlabel(ut.plot.get_label(prop_name, species=spec_name, get_units=True))
    subplot.set_ylabel(
        ut.plot.get_label(
            prop_name, prop_statistic, spec_name, get_symbol=True, get_units=False,
            get_log=prop_scaling))

    plot_func = ut.plot.get_plot_function(subplot, prop_scaling, axis_y_scaling)
    for part_i, part in enumerate(parts):
        plot_func(Stat.distr['bin.mid'][part_i], Stat.distr[prop_statistic][part_i],
                  color=colors[part_i], alpha=0.7, linewidth=2.5,
                  label=part.info['simulation.name'])

    # redshift legend
    #legend_z = None
    legend_z = subplot.legend(
        [plt.Line2D((0, 0), (0, 0), linestyle='')],
        ['$z={:.1f}$'.format(parts[0].snapshot['redshift'])],
        loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    # property legend
    if len(parts) > 1 and parts[0].info['simulation.name']:
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    plot_name = spec_name + '.' + prop_name + '_distr_z.{:.1f}'.format(part.info['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_property_v_property(
    part, spec_name='gas',
    x_prop_name='density', x_prop_limits=[], x_prop_scaling='log',
    y_prop_name='temperature', y_prop_limits=[], y_prop_scaling='log',
    prop_bin_number=300, weight_by_mass=True, cut_percent=0,
    host_distance_limitss=[20, 300], center_position=None,
    other_prop_limits={}, part_indices=None,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot property v property.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    spec_name : string : particle species
    x_prop_name : string : property name for x-axis
    x_prop_limits : list : min and max limits to impose on x_prop_name
    x_prop_scaling : string : 'log', 'lin'
    y_prop_name : string : property name for y-axis
    y_prop_limits : list : min and max limits to impose on y_prop_name
    y_prop_scaling : string : 'log', 'lin'
    prop_bin_number : int : number of bins for histogram along each axis
    weight_by_mass : boolean : whether to weight property by particle mass
    host_distance_limitss : list : min and max limits for distance from galaxy
    center_position : array : position of galaxy center
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indices : array : indices of particles from which to select
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    center_position = ut.particle.parse_property(part, 'position', center_position)

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.arange_length(part[spec_name].prop(x_prop_name))

    if other_prop_limits:
        part_indices = ut.catalog.get_indices_catalog(
            part[spec_name], other_prop_limits, part_indices)

    if len(center_position) and len(host_distance_limitss):
        distances = ut.coordinate.get_distances(
            'scalar', center_position, part[spec_name]['position'][part_indices],
            part.info['box.length']) * part.snapshot['scalefactor']
        part_indices = part_indices[ut.array.get_indices(distances, host_distance_limitss)]

    x_prop_values = part[spec_name].prop(x_prop_name, part_indices)
    y_prop_values = part[spec_name].prop(y_prop_name, part_indices)
    masses = None
    if weight_by_mass:
        masses = part[spec_name].prop('mass', part_indices)

    part_indices = ut.array.arange_length(part_indices)

    if x_prop_limits:
        part_indices = ut.array.get_indices(x_prop_values, x_prop_limits, part_indices)

    if y_prop_limits:
        part_indices = ut.array.get_indices(y_prop_values, y_prop_limits, part_indices)

    if cut_percent > 0:
        x_limits = ut.array.get_limits(x_prop_values[part_indices], cut_percent=cut_percent)
        y_limits = ut.array.get_limits(y_prop_values[part_indices], cut_percent=cut_percent)
        part_indices = ut.array.get_indices(x_prop_values, x_limits, part_indices)
        part_indices = ut.array.get_indices(y_prop_values, y_limits, part_indices)

    x_prop_values = x_prop_values[part_indices]
    y_prop_values = y_prop_values[part_indices]
    if weight_by_mass:
        masses = masses[part_indices]

    if 'log' in x_prop_scaling:
        x_prop_values = ut.math.get_log(x_prop_values)

    if 'log' in y_prop_scaling:
        y_prop_values = ut.math.get_log(y_prop_values)

    print(x_prop_values.size, y_prop_values.size)

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03, wspace=0.03)

    axis_x_limits = ut.plot.get_axis_limits(x_prop_values, 'lin', x_prop_limits)
    axis_y_limits = ut.plot.get_axis_limits(y_prop_values, 'lin', y_prop_limits)

    axis_x_label = ut.plot.get_label(
        x_prop_name, species=spec_name, get_units=True, get_symbol=True, get_log=x_prop_scaling)
    subplot.set_xlabel(axis_x_label, fontsize=30)

    axis_y_label = ut.plot.get_label(
        y_prop_name, species=spec_name, get_units=True, get_symbol=True, get_log=y_prop_scaling)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    _valuess, _xs, _ys, _Image = plt.hist2d(
        x_prop_values, y_prop_values, prop_bin_number, [axis_x_limits, axis_y_limits],
        norm=colors.LogNorm(), weights=masses,
        cmin=None, cmax=None,
        cmap=plt.cm.YlOrBr,  # @UndefinedVariable
    )

    """
    _valuess, xs, ys = np.histogram2d(
        x_prop_values, y_prop_values, prop_bin_number,
        #[axis_x_limits, axis_y_limits],
        weights=masses)

    subplot.imshow(
        _valuess.transpose(),
        norm=colors.LogNorm(),
        cmap=plt.cm.YlOrBr,  # @UndefinedVariable
        aspect='auto',
        #interpolation='nearest',
        interpolation='none',
        extent=(axis_x_limits[0], axis_x_limits[1], axis_y_limits[0], axis_y_limits[1]),
        #vmin=valuess.min(), vmax=valuess.max(),
        label=label,
    )
    """
    plt.colorbar()

    if host_distance_limitss is not None and len(host_distance_limitss):
        label = ut.plot.get_label_distance('host.distance', host_distance_limitss)

        # distance legend
        legend = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle=':')], [label],
            loc='best', prop=FontProperties(size=18))
        legend.get_frame().set_alpha(0.5)

    # plt.tight_layout(pad=0.02)

    plot_name = (spec_name + '.' + y_prop_name + '_v_' + x_prop_name + '_z.{:.1f}'.format(
                 part.info['redshift']))
    if host_distance_limitss is not None and len(host_distance_limitss):
        plot_name += '_d.{:.0f}-{:.0f}'.format(host_distance_limitss[0], host_distance_limitss[1])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_property_v_distance(
    parts, species='dark',
    prop_name='mass', prop_statistic='sum', prop_scaling='log', weight_by_mass=False,
    distance_limits=[0.1, 300], distance_bin_width=0.02, distance_bin_number=None,
    distance_scaling='log',
    dimension_number=3, rotation_vectors=None, axis_distance_max=Inf, other_axis_distance_max=None,
    center_positions=None, center_velocities=None,
    other_prop_limits={}, part_indicess=None,
    axis_y_limits=[], distance_reference=None, label_redshift=True,
    get_values=False, write_plot=False, plot_directory='.', figure_index=1):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshot)
    species : string or list : species to compute total mass of
        options: 'dark', 'star', 'gas', 'baryon', 'total'
    prop_name : string : property to get profile of
    prop_statistic : string : statistic/type to plot:
        sum, sum.cum, density, density.cum, vel.circ, sum.fraction, sum.cum.fraction, med, ave
    prop_scaling : string : scaling for property (y-axis): 'log', 'lin'
    weight_by_mass : boolean : whether to weight property by particle mass
    distance_limits : list : min and max distance for binning
    distance_bin_width : float : width of distance bin
    distance_bin_number : int : number of bins between limits
    distance_scaling : string : 'log', 'lin'
    dimension_number : int : number of spatial dimensions for profile
        note : if 1, get profile along minor axis, if 2, get profile along 2 major axes
    rotation_vectors : array : eigen-vectors to define rotation
    axis_distance_max : float : maximum distance to use in defining principal axes {kpc physical}
    other_axis_distance_max : float :
        maximum distance along other axis[s] to keep particles {kpc physical}
    center_positions : array or list of arrays : position of center for each particle catalog
    center_velocities : array or list of arrays : velocity of center for each particle catalog
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    axis_y_limits : list : limits to impose on y-axis
    distance_reference : float : reference distance at which to draw vertical line
    label_redshift : boolean : whether to label redshift
    get_values : boolean : whether to return values plotted
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions)
    if 'velocity' in prop_name:
        center_velocities = ut.particle.parse_property(parts, 'velocity', center_velocities)
    else:
        center_velocities = [center_velocities for _ in center_positions]
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, width=distance_bin_width, number=distance_bin_number,
        dimension_number=dimension_number)

    SpeciesProfile = SpeciesProfileClass()
    pros = []

    for part_i, part in enumerate(parts):
        pros_part = SpeciesProfile.get_profiles(
            part, species, prop_name, prop_statistic, weight_by_mass, DistanceBin,
            center_positions[part_i], center_velocities[part_i], rotation_vectors,
            axis_distance_max, other_axis_distance_max, other_prop_limits, part_indicess[part_i])

        pros.append(pros_part)

        #if part_i > 0:
        #    print(pros[part_i][prop_name] / pros[0][prop_name])

        #print(pros_part[species][prop_statistic])

    # plot ----------
    fig = plt.figure(figure_index)
    fig.clf()
    plt.minorticks_on()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03, wspace=0.03)

    subplot.set_xlim(distance_limits)
    y_values = [pro[species][prop_statistic] for pro in pros]
    axis_y_limits = ut.plot.get_axis_limits(y_values, prop_scaling, axis_y_limits)
    subplot.set_ylim(axis_y_limits)

    #subplot.set_xlabel('radius $r$ $[\\rm kpc\,physical]$')
    subplot.set_xlabel('radius $r$ $[\\mathrm{kpc}]$', fontsize=30)

    if prop_statistic == 'vel.circ':
        label_prop_name = 'vel.circ'
    else:
        label_prop_name = prop_name
    axis_y_label = ut.plot.get_label(
        label_prop_name, prop_statistic, species, dimension_number, get_symbol=True, get_units=True)
    #if prop_statistic == 'vel.circ':
    #    axis_y_label = 'circular velocity ' + axis_y_label
    subplot.set_ylabel(axis_y_label, fontsize=30)

    plot_func = ut.plot.get_plot_function(subplot, distance_scaling, prop_scaling)
    colors = ut.plot.get_colors(len(parts))

    if 'fraction' in prop_statistic or 'beta' in prop_name or 'velocity.rad' in prop_name:
        if 'fraction' in prop_statistic:
            y_values = [1, 1]
        elif 'beta' in prop_name:
            y_values = [0, 0]
        elif 'velocity.rad' in prop_name:
            y_values = [0, 0]
        plot_func(distance_limits, y_values, color='black', linestyle=':', alpha=0.5, linewidth=2)

    if len(pros) == 1:
        alpha = 0.9
        linewidth = 3.5
    else:
        alpha = 0.6
        linewidth = 2.5

    for part_i, pro in enumerate(pros):
        plot_func(pro[species]['distance'], pro[species][prop_statistic], color=colors[part_i],
                  linestyle='-', alpha=alpha, linewidth=linewidth,
                  label=parts[part_i].info['simulation.name'])

    if distance_reference is not None:
        plot_func([distance_reference, distance_reference], axis_y_limits,
                  color='black', linestyle=':', alpha=0.6)

    # redshift legend
    legend_z = None
    if label_redshift:
        legend_z = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$z={:.1f}$'.format(parts[0].snapshot['redshift'])],
            loc='lower left', prop=FontProperties(size=16))
        legend_z.get_frame().set_alpha(0.5)

    if len(parts) > 1 and parts[0].info['simulation.name']:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    plot_name = (species + '.' + prop_name + '.' + prop_statistic + '_v_dist_z.{:.1f}'.format(
                 part.info['redshift']))
    plot_name = plot_name.replace('.sum', '')
    plot_name = plot_name.replace('mass.vel.circ', 'vel.circ')
    plot_name = plot_name.replace('mass.density', 'density')
    ut.plot.parse_output(write_plot, plot_directory, plot_name)

    if get_values:
        if len(parts) == 1:
            pros = pros[0]
        return pros


#===================================================================================================
# mass and star-formation history
#===================================================================================================
def get_time_bin_dictionary(
    time_kind='redshift', time_limits=[0, 10], time_width=0.01, time_scaling='lin', Cosmology=None):
    '''
    .
    '''
    assert time_kind in ['time', 'time.lookback', 'redshift', 'scalefactor']

    time_limits = np.array(time_limits)

    if 'log' in time_scaling:
        if time_kind == 'redshift':
            time_limits += 1  # convert to z + 1 so log is well-defined
        times = 10 ** np.arange(
            log10(time_limits.min()), log10(time_limits.max()) + time_width, time_width)
        if time_kind == 'redshift':
            times -= 1
    else:
        times = np.arange(time_limits.min(), time_limits.max() + time_width, time_width)

    # if input limits is reversed, get reversed array
    if time_limits[1] < time_limits[0]:
        times = times[::-1]

    time_dict = {}

    if 'time' in time_kind:
        if 'lookback' in time_kind:
            time_dict['time.lookback'] = times
            time_dict['time'] = Cosmology.get_time_from_redshift(0) - times
        else:
            time_dict['time'] = times
            time_dict['time.lookback'] = Cosmology.get_time_from_redshift(0) - times
        time_dict['redshift'] = Cosmology.convert_time('redshift', 'time', time_dict['time'])
        time_dict['scalefactor'] = 1 / (1 + time_dict['redshift'])

    else:
        if 'redshift' in time_kind:
            time_dict['redshift'] = times
            time_dict['scalefactor'] = 1 / (1 + time_dict['redshift'])
        elif 'scalefactor' in time_kind:
            time_dict['scalefactor'] = times
            time_dict['redshift'] = 1 / time_dict['scalefactor'] - 1
        time_dict['time'] = Cosmology.get_time_from_redshift(time_dict['redshift'])
        time_dict['time.lookback'] = Cosmology.get_time_from_redshift(0) - time_dict['time']

    return time_dict


def get_star_form_history(
    part, time_kind='redshift', time_limits=[0, 8], time_width=0.1, time_scaling='lin',
    distance_limits=None, center_position=None, other_prop_limits=None, part_indices=None):
    '''
    Get array of times and star-formation rate at each time.

    Parameters
    ----------
    part : dict : dictionary of particles
    time_kind : string : time kind to use: time, time.lookback, redshift
    time_limits : list : min and max limits of time_kind to impose
    time_width : float : width of time_kind bin (in units set by time_scaling)
    time_scaling : string : scaling of time_kind: 'log', 'lin'
    distance_limits : list : min and max limits of galaxy distance to select star particles
    center_position : list : position of galaxy centers {kpc comoving}
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indices : array : indices of star particles to select

    Returns
    -------
    time_bins : array : times at midpoint of bin {Gyr or redshift, according to time_kind}
    dm_dt_in_bins : array : star-formation rate at each time bin mid {M_sun / yr}
    mass_cum_in_bins : array : cumulative mass at each time bin mid {M_sun}
    '''
    species = 'star'

    if part_indices is None:
        part_indices = ut.array.arange_length(part[species]['mass'])

    if other_prop_limits:
        part_indices = ut.catalog.get_indices_catalog(
            part['star'], other_prop_limits, part_indices)

    center_position = ut.particle.parse_property(part, 'position', center_position)

    if (center_position is not None and len(center_position) and
            distance_limits is not None and len(distance_limits)):
        distances = ut.coordinate.get_distances(
            'scalar', part['star']['position'][part_indices], center_position,
            part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}
        part_indices = part_indices[ut.array.get_indices(distances, distance_limits)]

    # get star particle formation times, sorted from earliest
    part_indices_sort = part_indices[np.argsort(part[species].prop('form.time', part_indices))]
    form_times = part[species].prop('form.time', part_indices_sort)
    masses = part[species]['mass'][part_indices_sort]
    masses_cum = np.cumsum(masses)

    # get time bins, ensure are ordered from earliest
    time_dict = get_time_bin_dictionary(
        time_kind, time_limits, time_width, time_scaling, part.Cosmology)
    time_bins = np.sort(time_dict['time'])

    mass_cum_in_bins = np.interp(time_bins, form_times, masses_cum)
    mass_difs = np.diff(mass_cum_in_bins)
    time_difs = np.diff(time_bins)
    dm_dt_in_bins = mass_difs / time_difs / ut.const.giga  # convert to {M_sun / yr}

    # convert to midpoints of bins
    mass_cum_in_bins = mass_cum_in_bins[: mass_cum_in_bins.size - 1] + 0.5 * mass_difs

    # account for stellar mass loss, crudely assuming instantaneous recycling
    dm_dt_in_bins /= 0.7
    #mass_cum_in_bins += mass_difs * (1 / 0.7 - 1)

    for k in time_dict:
        time_dict[k] = time_dict[k][: time_dict[k].size - 1] + 0.5 * np.diff(time_dict[k])

    # ensure ordering jives with ordering of input limits
    if time_dict['time'][0] > time_dict['time'][1]:
        dm_dt_in_bins = dm_dt_in_bins[::-1]
        mass_cum_in_bins = mass_cum_in_bins[::-1]

    sfh = {}
    for k in time_dict:
        sfh[k] = time_dict[k]
    sfh['form.rate'] = dm_dt_in_bins
    sfh['form.rate.specific'] = dm_dt_in_bins / mass_cum_in_bins
    sfh['mass'] = mass_cum_in_bins
    sfh['mass.normalized'] = mass_cum_in_bins / mass_cum_in_bins.max()
    sfh['particle.number'] = form_times.size

    return sfh


def plot_star_form_history(
    parts=None, sfh_kind='rate',
    time_kind='time.lookback', time_limits=[13.8, 0], time_width=0.2, time_scaling='lin',
    distance_limits=[0, 10], center_positions=None, other_prop_limits={}, part_indicess=None,
    axis_y_limits=[], axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot star-formation rate history v time_kind.
    Note: assumes instantaneous recycling of 30% for mass, should fix this for mass lass v time.

    Parameters
    ----------
    parts : dict or list : catalog[s] of particles
    sfh_kind : string : star form kind to plot: 'rate', 'rate.specific', 'mass', 'mass.normalized'
    time_kind : string : time kind to use: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to get
    time_width : float : width of time_kind bin
    time_scaling : string : scaling of time_kind: 'log', 'lin'
    distance_limits : list : min and max limits of distance to select star particles
    center_positions : list or list of lists : position[s] of galaxy centers {kpc comoving}
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array : part_indices of particles from which to select
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : scaling of y-axis: 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_star_form_history)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions)
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)

    time_limits = np.array(time_limits)
    if time_limits[1] is None:
        time_limits[1] = parts[0].snapshot[time_kind]

    sfh = {}

    for part_i, part in enumerate(parts):
        sfh_p = get_star_form_history(
            part, time_kind, time_limits, time_width, time_scaling,
            distance_limits, center_positions[part_i], other_prop_limits, part_indicess[part_i])

        if part_i == 0:
            for k in sfh_p:
                sfh[k] = []  # initialize

        for k in sfh_p:
            sfh[k].append(sfh_p[k])

        Say.say('star.mass max = {:.3e}'.format(sfh_p['mass'].max()))

    if time_kind == 'redshift' and 'log' in time_scaling:
        time_limits += 1  # convert to z + 1 so log is well-defined

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.86, bottom=0.15, hspace=0.03, wspace=0.03)

    ut.plot.make_axis_time(
        subplot, time_kind, time_limits, time_scaling, label_axis_2=True, Cosmology=part.Cosmology,
        fontsize=26)

    subplot.set_ylim(ut.plot.get_axis_limits(sfh[sfh_kind], axis_y_scaling, axis_y_limits))

    if 'mass' in sfh_kind:
        axis_y_label = ut.plot.get_label('star.mass', get_symbol=True, get_units=True)
    else:
        axis_y_label = ut.plot.get_label('sfr', get_symbol=True, get_units=True)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    colors = ut.plot.get_colors(len(parts))

    plot_func = ut.plot.get_plot_function(subplot, time_scaling, axis_y_scaling)

    for part_i, part in enumerate(parts):
        plot_func(sfh[time_kind][part_i], sfh[sfh_kind][part_i],
                  linewidth=3.0, color=colors[part_i], alpha=0.9,
                  label=part.info['simulation.name'])

    # redshift legend
    legend_z = None
    if parts[0].snapshot['redshift'] > 0.01:
        legend_z = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$z={:.1f}$'.format(parts[0].snapshot['redshift'])],
            loc='lower left', prop=FontProperties(size=18))
        legend_z.get_frame().set_alpha(0.5)

    # property legend
    if len(parts) > 1 and parts[0].info['simulation.name']:
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=18))
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    sfh_name = 'star' + '.' + sfh_kind + '.history'
    plot_name = '{}_v_{}_z.{:.1f}'.format(sfh_name, time_kind, part.info['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_star_form_history_galaxies(
    part, hal=None, gals=None,
    mass_kind='star.mass.part', mass_limits=[1e5, 1e9], other_prop_limits={}, hal_indices=None,
    sfh_kind='mass.normalized', sfh_limits=[], sfh_scaling='lin',
    time_kind='time.lookback', time_limits=[13.7, 0], time_width=0.2, time_scaling='lin',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot star-formation rate history v time_kind.
    Note: assumes instantaneous recycling of 30% of mass, should fix this for mass lass v time.

    Parameters
    ----------
    part : dict : catalog of particles
    hal : dict : catalog of halos at snapshot
    gals : dict : SFHs of observed galaxies in the Local Group
    mass_kind : string : mass kind by which to select halos
    mass_limits : list : min and max limits to impose on mass_kind
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    hal_indices : index or array : index[s] of halo[s] whose particles to plot
    sfh_kind : string : star form kind to plot: 'rate', 'rate.specific', 'mass', 'mass.normalized'
    sfh_limits : list : min and max limits for y-axis
    sfh_scaling : string : scailng of y-axis: 'log', 'lin'
    time_kind : string : time kind to plot: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to plot
    time_width : float : width of time_kind bin
    time_scaling : string : scaling of time_kind: 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_star_form_history_galaxies)

    time_limits = np.array(time_limits)
    if time_limits[0] is None:
        time_limits[0] = part.snapshot[time_kind]
    if time_limits[1] is None:
        time_limits[1] = part.snapshot[time_kind]

    sfh = None
    if hal is not None:
        #import ipdb; ipdb.set_trace()
        if hal_indices is None or not len(hal_indices):
            hal_indices = ut.array.get_indices(hal.prop('star.number'), [2, Inf])

        if mass_limits is not None and len(mass_limits):
            hal_indices = ut.array.get_indices(hal.prop(mass_kind), mass_limits, hal_indices)

        if other_prop_limits:
            hal_indices = ut.catalog.get_indices_catalog(
                hal, other_prop_limits, hal_indices)

        hal_indices = hal_indices[np.argsort(hal.prop(mass_kind, hal_indices))]

        print('halo number = {}'.format(hal_indices.size))

        sfh = {}

        for hal_ii, hal_i in enumerate(hal_indices):
            part_indices = hal.prop('star.indices', hal_i)
            sfh_h = get_star_form_history(
                part, time_kind, time_limits, time_width, time_scaling, part_indices=part_indices)

            if hal_ii == 0:
                for k in sfh_h:
                    sfh[k] = []  # initialize

            for k in sfh_h:
                sfh[k].append(sfh_h[k])

            Say.say('id = {:8d}, star.mass = {:.3e}, particle.number = {}'.format(
                    hal_i, sfh_h['mass'].max(), part_indices.size))
            #print(hal.prop('position', hal_i))

        for k in sfh:
            sfh[k] = np.array(sfh[k])

        sfh['mass.normalized.median'] = np.median(sfh['mass.normalized'], 0)

    if time_kind == 'redshift' and 'log' in time_scaling:
        time_limits += 1  # convert to z + 1 so log is well-defined

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.86, bottom=0.15, hspace=0.03, wspace=0.03)

    ut.plot.make_axis_time(
        subplot, time_kind, time_limits, time_scaling, label_axis_2=True, Cosmology=part.Cosmology,
        fontsize=28)

    y_values = None
    if sfh is not None:
        y_values = sfh[sfh_kind]
    subplot.set_ylim(ut.plot.get_axis_limits(y_values, sfh_scaling, sfh_limits))

    if 'mass' in sfh_kind:
        if 'normalized' in sfh_kind:
            axis_y_label = '$M_{\\rm star}(z)\, / \, M_{\\rm star}(z=0)$'
        else:
            axis_y_label = ut.plot.get_label('star.mass', get_symbol=True, get_units=True)
    else:
        axis_y_label = ut.plot.get_label('sfr', get_symbol=True, get_units=True)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    if hal is not None:
        colors = ut.plot.get_colors(len(hal_indices))
    elif gals is not None:
        colors = ut.plot.get_colors(len(gals))

    plot_func = ut.plot.get_plot_function(subplot, time_scaling, sfh_scaling)

    label = None
    if hal is not None:
        for hal_ii, hal_i in enumerate(hal_indices):
            label = '$M_{{\\rm star}}={}\,M_\odot$'.format(
                ut.io.get_string_for_exponential(sfh['mass'][hal_ii][-1], 0))
            plot_func(sfh[time_kind][hal_ii], sfh[sfh_kind][hal_ii],
                      linewidth=3.0, color=colors[hal_ii], alpha=0.5, label=label)

    if gals is not None:
        for gal_i, gal_name in enumerate(gals):
            if hal is not None:
                color = 'black'
                linewidth = 2.0
                alpha = 0.15
                linestyle = '-'
            else:
                color = colors[gal_i]
                linewidth = 3.0
                alpha = 0.5
                linestyle = '-'
            plot_func(gals[gal_name][time_kind], gals[gal_name][sfh_kind],
                      linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color)

    #plot_func(sfh['time'][0], sfh['mass.normalized.median'],
    #          linewidth=4.0, color='black', alpha=0.5)

    # redshift legend
    legend_z = None
    if part.snapshot['redshift'] > 0.01:
        legend_z = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$z={:.1f}$'.format(part.snapshot['redshift'])],
            loc='lower left', prop=FontProperties(size=16))
        legend_z.get_frame().set_alpha(0.5)

    # property legend
    if part.info['simulation.name'] and label is not None:
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    sf_name = 'star' + '.' + sfh_kind
    plot_name = '{}_v_{}'.format(sf_name, time_kind)
    if gals is not None:
        plot_name += '_lg'
    if hal is not None:
        plot_name += '_z.{:.1f}'.format(part.snapshot['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# galaxy mass and radius at snapshots
#===================================================================================================
def write_galaxy_properties_v_time(simulation_directory='.', redshifts=[], species=['star']):
    '''
    Read snapshots and store dictionary of host galaxy properties (such as mass and radius)
    at snapshots.

    Parameters
    ----------
    simulation_directory : string : root directory of simulation
    redshifts : array-like : redshifts at which to get properties
        'all' = read and store all snapshots
    species : string or list : species to read and get properties of

    Returns
    -------
    dictionary of host galaxy properties at input redshifts
    '''
    from . import gizmo_io

    star_distance_max = 15

    properties_read = ['mass', 'position']

    mass_percents = [50, 90]

    simulation_directory = ut.io.get_path(simulation_directory)

    gal = {
        'index': [],
        'redshift': [],
        'scalefactor': [],
        'time': [],
        'time.lookback': [],
    }

    for spec_name in species:
        gal['{}.position'.format(spec_name)] = []
        for mass_percent in mass_percents:
            gal['{}.radius.{:.0f}'.format(spec_name, mass_percent)] = []
            gal['{}.mass.{:.0f}'.format(spec_name, mass_percent)] = []

    if redshifts == 'all' or redshifts is None or redshifts == []:
        Snapshot = ut.simulation.SnapshotClass()
        Snapshot.read_snapshots('snapshot_times.txt', simulation_directory)
        redshifts = Snapshot['redshift']
    else:
        if np.isscalar(redshifts):
            redshifts = [redshifts]

    redshifts = np.sort(redshifts)

    for _zi, redshift in enumerate(redshifts):
        part = gizmo_io.Gizmo.read_snapshot(
            species, 'redshift', redshift, simulation_directory, property_names=properties_read,
            force_float32=True)

        for k in ['index', 'redshift', 'scalefactor', 'time', 'time.lookback']:
            gal[k].append(part.snapshot[k])

        # get position and velocity
        gal['star.position'].append(part.center_position)

        for spec_name in species:
            for mass_percent in mass_percents:
                gal_radius, gal_mass = ut.particle.get_galaxy_radius_mass(
                    part, spec_name, 'mass.percent', mass_percent, star_distance_max)
                gal['{}.radius.{:.0f}'.format(spec_name, mass_percent)].append(gal_radius)
                gal['{}.mass.{:.0f}'.format(spec_name, mass_percent)].append(gal_mass)

    for prop in gal:
        gal[prop] = np.array(gal[prop])

    ut.io.pickle_object(simulation_directory + 'galaxy_properties_v_time', 'write', gal)

    return gal


def plot_galaxy_property_v_time(
    gals=None, sfhs=None, Cosmology=None,
    prop_name='star.mass',
    time_kind='redshift', time_limits=[0, 8], time_scaling='lin', snapshot_subsample_factor=1,
    axis_y_limits=[], axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot host galaxy property v time_kind, using tabulated dictionary of properties of progenitor
    across snapshots.

    Parameters
    ----------
    gals : dict : tabulated dictionary of host galaxy properties
    sfhs : dict : tabulated dictinnary of star-formation histories (computed at single snapshot)
    prop_name : string : star formation history kind to plot:
        'rate', 'rate.specific', 'mass', 'mass.normalized'
    time_kind : string : time kind to use: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to get
    time_scaling : string : scaling of time_kind: 'log', 'lin'
    snapshot_subsample_factor : int : factor by which to sub-sample snapshots from gals
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : scaling of y-axis: 'log', 'lin'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    #Say = ut.io.SayClass(plot_galaxy_property_v_time)

    if gals is not None and isinstance(gals, dict):
        gals = [gals]

    if sfhs is not None and isinstance(sfhs, dict):
        sfhs = [sfhs]

    time_limits = np.array(time_limits)
    if time_limits[0] is None:
        time_limits[0] = gals[0][time_kind].min()
    if time_limits[1] is None:
        time_limits[1] = gals[0][time_kind].max()

    if time_kind == 'redshift' and 'log' in time_scaling:
        time_limits += 1  # convert to z + 1 so log is well-defined

    # plot ----------
    plt.minorticks_on()
    fig = plt.figure(figure_index)
    fig.clf()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=figure_index, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.86, bottom=0.15, hspace=0.03, wspace=0.03)

    ut.plot.make_axis_time(
        subplot, time_kind, time_limits, time_scaling, label_axis_2=True, Cosmology=Cosmology,
        fontsize=26)

    y_values = []
    if gals is not None:
        y_values.append(gals[0][prop_name])
    if sfhs is not None:
        y_values.append(sfhs[0][time_kind])
    subplot.set_ylim(ut.plot.get_axis_limits(y_values, axis_y_scaling, axis_y_limits))

    if 'mass' in prop_name:
        axis_y_label = ut.plot.get_label('star.mass', get_symbol=True, get_units=True)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    #colors = ut.plot.get_colors(len(gals))

    plot_func = ut.plot.get_plot_function(subplot, time_scaling, axis_y_scaling)

    if gals is not None:
        for _gal_i, gal in enumerate(gals):
            plot_func(
                gal[time_kind][::snapshot_subsample_factor],
                gal[prop_name][::snapshot_subsample_factor],
                linewidth=3.0, alpha=0.9,
                #color=colors[gal_i],
                color=ut.plot.get_color('blue.mid'),
                label='main progenitor',
            )

    if sfhs is not None:
        for _sfh_i, sfh in enumerate(sfhs):
            plot_func(
                sfh[time_kind], sfh['mass'],
                '--', linewidth=3.0, alpha=0.9,
                #color=colors[sfh_i],
                color=ut.plot.get_color('orange.mid'),
                label='SFH computed at $z=0$',
            )

    # property legend
    #if len(gals) > 1 and gals[0].info['simulation.name']:
    legend_prop = subplot.legend(loc='best', prop=FontProperties(size=18))
    legend_prop.get_frame().set_alpha(0.5)

    #plt.tight_layout(pad=0.02)

    plot_name = 'galaxy_{}_v_{}'.format(prop_name, time_kind)
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# disk mass and radius over time, for shea
#===================================================================================================
def get_galaxy_mass_profiles_v_redshift(
    directory='.', redshifts=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0],
    parts=None):
    '''
    Read snapshots and store dictionary of galaxy/halo position, velocity, size, mass at input
    scale-factors.

    Parameters
    ----------
    directory : string : directory of snapshot files
    redshifts : array-like : redshifts at which to get properties

    Returns
    -------
    dictionary of galaxy/halo properties at each redshift
    '''
    from . import gizmo_io

    star_distance_max = 20
    dark_distance_max = 50

    property_names = ['mass', 'position', 'velocity', 'potential']

    species_read = ['star', 'dark']
    spec_name = 'star'
    mass_percents = [90]

    gal = {
        'index': [],
        'redshift': [],
        'scalefactor': [],
        'time': [],

        'position': [],
        'velocity': [],
        'dark.position': [],
        'dark.velocity': [],
        'rotation.tensor': [],
        'axis.ratio': [],

        'profile.3d.distance': [],
        'profile.3d.density': [],

        'profile.minor.distance': [],
        'profile.minor.density': [],

        'profile.major.distance': [],
        'profile.major.density': [],
    }

    for mass_percent in mass_percents:
        mass_percent_name = '{:.0f}'.format(mass_percent)

        gal['radius.3d.' + mass_percent_name] = []
        gal['mass.3d.' + mass_percent_name] = []
        gal['profile.3d.distance'] = []

        gal['radius.major.' + mass_percent_name] = []
        gal['mass.major.' + mass_percent_name] = []

        gal['radius.minor.' + mass_percent_name] = []
        gal['mass.minor.' + mass_percent_name] = []

    for zi, redshift in enumerate(redshifts):
        if parts is not None and len(parts):
            part = parts[zi]
        else:
            part = gizmo_io.Gizmo.read_snapshot(
                species_read, 'redshift', redshift, directory, property_names=property_names,
                force_float32=True)

        for k in ['redshift', 'scalefactor', 'time']:
            gal[k].append(part.snapshot[k])

        # get position and velocity
        gal['position'].append(part.center_position)
        gal['velocity'].append(part.center_velocity)

        gal['dark.position'].append(ut.particle.get_center_position(part, 'dark', 'potential'))
        gal['dark.velocity'].append(
            ut.particle.get_center_velocity(part, 'dark', distance_max=dark_distance_max))

        # get radius_90 as fiducial
        gal_radius_90, _gal_mass_90 = ut.particle.get_galaxy_radius_mass(
            part, spec_name, 'mass.percent', mass_percent, star_distance_max)

        rotation_vectors, _eigen_values, axis_ratios = ut.particle.get_principal_axes(
            part, spec_name, gal_radius_90, scalarize=True)

        gal['rotation.tensor'].append(rotation_vectors)
        gal['axis.ratio'].append(axis_ratios)

        for mass_percent in mass_percents:
            mass_percent_name = '{:.0f}'.format(mass_percent)

            gal_radius, gal_mass = ut.particle.get_galaxy_radius_mass(
                part, spec_name, 'mass.percent', mass_percent, star_distance_max)
            gal['radius.3d.' + mass_percent_name].append(gal_radius)
            gal['mass.3d.' + mass_percent_name].append(gal_mass)

            gal_radius_minor, gal_mass_minor = ut.particle.get_galaxy_radius_mass(
                part, spec_name, 'mass.percent', mass_percent, star_distance_max,
                axis_kind='minor', rotation_vectors=rotation_vectors,
                other_axis_distance_max=gal_radius_90)
            gal['radius.minor.' + mass_percent_name].append(gal_radius_minor)
            gal['mass.minor.' + mass_percent_name].append(gal_mass_minor)

            gal_radius_major, gal_mass_major = ut.particle.get_galaxy_radius_mass(
                part, spec_name, 'mass.percent', mass_percent, star_distance_max,
                axis_kind='major', rotation_vectors=rotation_vectors,
                other_axis_distance_max=gal_radius_minor)
            gal['radius.major.' + mass_percent_name].append(gal_radius_major)
            gal['mass.major.' + mass_percent_name].append(gal_mass_major)

        pro = plot_property_v_distance(
            part, spec_name, 'mass', 'density', 'log', False, [0.1, 20], 0.1, None, 'log', 3,
            rotation_vectors=rotation_vectors, other_axis_distance_max=1, get_values=True)

        for k in ['distance', 'density']:
            gal['profile.3d.' + k].append(pro[spec_name][k])

        pro = plot_property_v_distance(
            part, spec_name, 'mass', 'density', 'log', False, [0.1, 20], 0.1, None, 'log', 1,
            rotation_vectors=rotation_vectors, other_axis_distance_max=1, get_values=True)

        for k in ['distance', 'density']:
            gal['profile.minor.' + k].append(pro[spec_name][k])

        pro = plot_property_v_distance(
            part, spec_name, 'mass', 'density', 'log', False, [0.1, 20], 0.1, None, 'log', 2,
            rotation_vectors=rotation_vectors, other_axis_distance_max=1, get_values=True)

        for k in ['distance', 'density']:
            gal['profile.major.' + k].append(pro[spec_name][k])

    for prop in gal:
        gal[prop] = np.array(gal[prop])

    return gal


def print_galaxy_mass_v_redshift(gal):
    '''
    Print galaxy/halo position, velocity, size, mass over time for Shea.

    Parameters
    ----------
    gal : dict : dictionary of galaxy properties across snapshots
    '''
    print('# redshift scale-factor time[Gyr] ', end='')
    print('star_position(x,y,z)[kpc comov] ', end='')
    print('star_velocity(x,y,z)[km/s phys] dark_velocity(x,y,z)[km/s phys] ', end='')
    print('r_50[kpc phys] star_mass_50[M_sun] gas_mass_50[M_sun] dark_mass_50[M_sun] ', end='')
    print('r_90[kpc phys] star_mass_90[M_sun] gas_mass_90[M_sun] dark_mass_90[M_sun]', end='\n')

    for ti in range(gal['redshift'].size):
        print('{:.5f} {:.5f} {:.5f} '.format(
              gal['redshift'][ti], gal['scalefactor'][ti], gal['time'][ti]), end='')
        print('{:.3f} {:.3f} {:.3f} '.format(
              gal['position'][ti][0], gal['position'][ti][1], gal['position'][ti][2]), end='')
        print('{:.3f} {:.3f} {:.3f} '.format(
              gal['star.velocity'][ti][0], gal['star.velocity'][ti][1],
              gal['star.velocity'][ti][2]), end='')
        print('{:.3f} {:.3f} {:.3f} '.format(
              gal['dark.velocity'][ti][0], gal['dark.velocity'][ti][1],
              gal['dark.velocity'][ti][2]), end='')
        print('{:.3e} {:.3e} {:.3e} {:.3e} '.format(
              gal['radius.50'][ti], gal['star.mass.50'][ti], gal['gas.mass.50'][ti],
              gal['dark.mass.50'][ti]), end='')
        print('{:.3e} {:.3e} {:.3e} {:.3e}'.format(
              gal['radius.90'][ti], gal['star.mass.90'][ti], gal['gas.mass.90'][ti],
              gal['dark.mass.90'][ti]), end='\n')


#===================================================================================================
# compare simulations
#===================================================================================================
class CompareSimulationsClass(ut.io.SayClass):
    '''
    .
    '''
    def __init__(self):
        '''
        Set directories and names of simulations to read.
        '''
        self.simulation_names = [
            # original FIRE
            ['/work/02769/arwetzel/fire/m12i_ref12', 'r12 FIRE n100'],

            # first ref13
            ['fb-aniso-angle-max/m12i_ref13', 'r13 aniso anglemax n100'],

            ['fb-aniso/m12i_ref12_fb-volume', 'r12 aniso volume'],
            ['fb-aniso/m12i_ref12', 'r12 aniso'],
            ['fb-aniso/m12i_ref13', 'r13 aniso'],
            ['/scratch/01799/phopkins/m12i_hybrid_test/quadratic_test', 'r12 mom-cons'],

            ['fb-iso/m12i_ref11', 'r11 iso'],
            ['fb-iso/m12i_ref12', 'r12 iso'],
            ['fb-iso/m12i_ref13', 'r13 iso'],

            ['fb-iso/m12i_ref12_sfn100', 'r12 iso n100'],
            ['fb-iso/m12i_ref13_sfn100', 'r13 iso n100'],

            # test adaptive resolution
            #['fb-iso/m12i_ref12_res-adapt', 'r12 iso res-adapt'],
            #['fb-iso/m12i_ref13_res-adapt', 'r13 iso res-adapt'],

            # test maximum radius of coupling
            #['fb-iso-radius-max/m12i_ref12_rmax1kpc', 'r12 iso rmax1kpc'],
            #['fb-iso-radius-max/m12i_ref13_rmax1kpc', 'r13 iso rmax1kpc'],
            #['fb-iso-radius-max/m12i_ref12_rmax10h', 'r12 iso rmax10h'],
            #['fb-iso-radius-max/m12i_ref13_rmax10h', 'r13 iso rmax10h'],

            # compare different halos
            ['m12i/fb-iso/m12i_ref12', 'latte r12'],
            ['m12b/m12b_ref12', 'breve r12'],
            ['m12m/m12m_ref12', 'macchiato r12'],
            ['m12c/m12c_ref12', 'cappuccino r12'],
            ['m12f/m12f_ref12', 'flatwhite r12'],

        ]

    def read_simulations(
        self, simulation_names=None, redshift=0, species='all',
        property_names=['mass', 'position', 'form.time'], force_float32=True):
        '''
        Read snapshots from simulations.
        '''
        from . import gizmo_io

        if simulation_names is None:
            simulation_names = self.simulation_names

        if not isinstance(simulation_names, collections.OrderedDict):
            simulation_names = collections.OrderedDict(simulation_names)

        parts = []
        directories = []
        for directory in simulation_names:
            try:
                part = gizmo_io.Gizmo.read_snapshot(
                    species, 'redshift', redshift, directory, property_names=property_names,
                    simulation_name=simulation_names[directory], force_float32=force_float32)

                if 'velocity' in property_names:
                    gizmo_io.Gizmo.assign_orbit(part, 'gas')

                parts.append(part)
                directories.append(directory)
            except:
                self.say('! could not read snapshot at z = {:.3f} in {}'.format(
                         redshift, directory))

        if not len(parts):
            self.say('! could not read any snapshots at z = {:.3f}'.format(redshift))
            return

        if 'mass' in property_names and 'star' in part:
            for part, directory in zip(parts, directories):
                print('{} star.mass = {:.3e}'.format(directory, part['star']['mass'].sum()))

        return parts

    def plot_simulations(
        self, parts=None, simulation_names=None, redshifts=[6, 5, 4, 3, 2, 1.5, 1, 0.5, 0],
        species='all', property_names=['mass', 'position', 'form.time'], force_float32=True):
        '''
        Plot various properties, comparing all simulations at fixed redshift.
        '''
        if np.isscalar(redshifts):
            redshifts = [redshifts]

        for redshift in redshifts:
            if parts is None:
                parts = self.read_simulations(
                    simulation_names, redshift, species, property_names, force_float32)

            plot_property_v_distance(
                parts, 'total', 'mass', 'vel.circ', 'lin', False, [0.1, 300], 0.1,
                axis_y_limits=[0, None], write_plot=True)

            plot_property_v_distance(
                parts, 'total', 'mass', 'sum.cum', 'log', False, [1, 300], 0.1,
                axis_y_limits=[None, None], write_plot=True)

            plot_property_v_distance(
                parts, 'dark', 'mass', 'sum.cum', 'log', False, [1, 300], 0.1,
                axis_y_limits=[None, None], write_plot=True)

            plot_property_v_distance(
                parts, 'dark', 'mass', 'density', 'log', False, [0.1, 30], 0.1,
                axis_y_limits=[None, None], write_plot=True)

            if 'gas' in parts[0]:
                plot_property_v_distance(
                    parts, 'baryon', 'mass', 'sum.cum.fraction', 'lin', False, [10, 2000], 0.1,
                    axis_y_limits=[0, 2], write_plot=True)

                plot_property_v_distance(
                    parts, 'gas', 'mass', 'sum.cum', 'log', False, [1, 300], 0.1,
                    axis_y_limits=[None, None], write_plot=True)

            if 'star' in parts[0]:
                plot_property_v_distance(
                    parts, 'star', 'mass', 'sum.cum', 'log', False, [1, 300], 0.1,
                    axis_y_limits=[None, None], write_plot=True)

                plot_property_v_distance(
                    parts, 'star', 'mass', 'density', 'log', False, [0.1, 30], 0.1,
                    axis_y_limits=[None, None], write_plot=True)

            if 'velocity' in property_names:
                plot_property_v_distance(
                    parts, 'gas', 'host.velocity.rad', 'average', 'lin', True, [1, 300], 0.25,
                    axis_y_limits=[None, None], write_plot=True)

            if 'form.time' in property_names and redshift <= 4:
                plot_star_form_history(
                    parts, 'mass', 'redshift', [0, 6], 0.2, 'lin', distance_limits=[0, 15],
                    axis_y_limits=[None, None], write_plot=True)

            for part in parts:
                for spec_name in ['star', 'gas']:
                    if spec_name in part:
                        plot_image(
                            part, spec_name, [0, 1, 2], [0, 1, 2], 15, 0.05,
                            image_limits=[3e7, 1e10], align_principal_axes=True,
                            write_plot=True, add_simulation_name=True)

CompareSimulations = CompareSimulationsClass()
