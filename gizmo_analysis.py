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
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator
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
        total metallicity of star prior to event, relative to solar = sun_metal_mass_fraction
    normalize : boolean : whether to normalize yields to be mass fractions (instead of masses)

    Returns
    -------
    yields : ordered dictionary : yield mass {M_sun} or mass fraction for each element
        can covert to regular dictionary via dict(yields) or list of values via yields.values()
    '''
    sun_metal_mass_fraction = 0.02  # total metal mass fraction of sun that Gizmo assumes

    element_dict = collections.OrderedDict()
    element_dict['metals'] = 0
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
                yield_dict['metals'] += yield_dict[k]

    elif event_kind == 'supernova.ii':
        # yields from Nomoto et al 2006, IMF averaged
        # rates from Starburst99
        # in Gizmo, these occur from 3.4 to 37.53 Myr after formation
        # from 3.4 to 10.37 Myr, rate / M_sun = 5.408e-10 yr ^ -1
        # from 10.37 to 37.53 Myr, rate / M_sun = 2.516e-10 yr ^ -1
        ejecta_mass = 10.5  # {M_sun}

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
        # 5.3e-14 + 1.6e-11 * exp(-0.5 * ((star_age - 0.05) / 0.01) *
        #                         ((star_age - 0.05) / 0.01)) yr ^ -1
        ejecta_mass = 1.4  # {M_sun}

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
    event_kind : string : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float : total metallicity of star prior to event, relative to solar
    normalize : boolean : whether to normalize yields to be mass fractions (instead of masses)
    axis_y_scaling : string : scaling along y-axis: 'log', 'linear'
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
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.96, top=0.94, bottom=0.14)
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
            y_label = 'yield $[M_\odot]$'
        subplots[si].set_ylabel(y_label, fontsize=28)
        subplots[si].set_xlabel('element', fontsize=28)
        #fig.set_ylabel(y_label, fontsize=26)
        #fig.set_xlabel('element', fontsize=26)

        for yi in yield_indices:
            if yield_values[yi] > 0:
                subplot.plot(
                    yield_indices[yi], yield_values[yi], 'o', markersize=14, color=colors[yi])
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

    plot_name = 'element.yields_{}_z.{:.2f}'.format(event_kind, star_metallicity)
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


class SpeciesProfileClass(ut.io.SayClass):
    '''
    Get profiles of either summation or stastitics (such as average, median) of given property for
    given particle species.
    '''
    def get_profiles(
        self, part, species=['all'], prop_name='', prop_statistic='sum', weight_by_mass=False,
        DistanceBin=None, center_position=None, center_velocity=None, rotation_vectors=None,
        axis_distance_max=Inf, other_axis_distance_limits=None, other_prop_limits={},
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
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles {kpc physical}
        other_prop_limits : dict : dictionary with properties as keys and limits as values
        part_indicess : array (species number x particle number) :
            indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if 'sum' in prop_statistic or 'vel.circ' in prop_statistic or 'density' in prop_statistic:
            pros = self.get_sum_profiles(
                part, species, prop_name, DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_limits, other_prop_limits, part_indicess)
        else:
            pros = self.get_statistics_profiles(
                part, species, prop_name, weight_by_mass, DistanceBin, center_position,
                center_velocity, rotation_vectors, axis_distance_max, other_axis_distance_limits,
                other_prop_limits, part_indicess)

        for k in pros:
            if '.cum' in prop_statistic or 'vel.circ' in prop_statistic:
                pros[k]['distance'] = pros[k]['distance.cum']
                pros[k]['log distance'] = pros[k]['log distance.cum']
            else:
                pros[k]['distance'] = pros[k]['distance.mid']
                pros[k]['log distance'] = pros[k]['log distance.mid']

        return pros

    def get_sum_profiles(
        self, part, species=['all'], prop_name='mass', DistanceBin=None, center_position=None,
        rotation_vectors=None, axis_distance_max=Inf, other_axis_distance_limits=None,
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
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles {kpc physical}
        other_prop_limits : dict : dictionary with properties as keys and limits as values
        part_indicess : array (species number x particle number) :
            indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if 'gas' in species and 'consume.time' in prop_name:
            pros_mass = self.get_sum_profiles(
                part, species, 'mass', DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_limits, other_prop_limits, part_indicess)

            pros_sfr = self.get_sum_profiles(
                part, species, 'sfr', DistanceBin, center_position, rotation_vectors,
                axis_distance_max, other_axis_distance_limits, other_prop_limits, part_indicess)

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
                    distances = distancess[1]
                    other_distances = distancess[0]
                elif DistanceBin.dimension_number == 2:
                    distances = distancess[0]
                    other_distances = distancess[1]

                if (other_axis_distance_limits is not None and
                        (min(other_axis_distance_limits) > 0 or
                         max(other_axis_distance_limits) < Inf)):
                    masks = ((other_distances >= min(other_axis_distance_limits)) *
                             (other_distances < max(other_axis_distance_limits)))
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

        return pros

    def get_statistics_profiles(
        self, part, species=['all'], prop_name='', weight_by_mass=True, DistanceBin=None,
        center_position=None, center_velocity=None, rotation_vectors=None, axis_distance_max=Inf,
        other_axis_distance_limits=None, other_prop_limits={}, part_indicess=None):
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
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles {kpc physical}
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
                try:
                    part_indices = ut.array.arange_length(part[spec_name].prop(prop_name))
                except:
                    part_indices = ut.array.arange_length(part[spec_name].prop('mass'))

            if other_prop_limits:
                part_indices = ut.catalog.get_indices_catalog(
                    part[spec_name], other_prop_limits, part_indices)

            masses = None
            if weight_by_mass:
                masses = part[spec_name].prop('mass', part_indices)

            if 'velocity' in prop_name:
                distance_vectors = ut.coordinate.get_distances(
                    'vector', part[spec_name]['position'][part_indices], center_position,
                    part.info['box.length']) * part.snapshot['scalefactor']  # {kpc physical}

                velocity_vectors = ut.coordinate.get_velocity_differences(
                    'vector', part[spec_name]['velocity'][part_indices], center_velocity, True,
                    part[spec_name]['position'][part_indices], center_position,
                    part.snapshot['scalefactor'], part.snapshot['time.hubble'],
                    part.info['box.length'])

                pro = DistanceBin.get_velocity_profile(distance_vectors, velocity_vectors, masses)

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

                    if (other_axis_distance_limits is not None and
                            min(other_axis_distance_limits) > 0 and
                            max(other_axis_distance_limits) < Inf):
                        masks = ((other_distances >= min(other_axis_distance_limits)) *
                                 (other_distances < max(other_axis_distance_limits)))
                        distances = distances[masks]
                        masses = masses[masks]
                        prop_values = prop_values[masks]

                pros[spec_name] = DistanceBin.get_statistics_profile(distances, prop_values, masses)

        return pros


#===================================================================================================
# diagnostic
#===================================================================================================
def plot_mass_contamination(
    part,
    distance_limits=[1, 2000], distance_bin_width=0.02, distance_bin_number=None,
    distance_scaling='log',
    halo_radius=None, scale_to_halo_radius=False, center_position=None,
    axis_y_limits=[0.0001, 3], axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot contamination from low-resolution particles v distance from center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    distance_limits : list : min and max limits for distance from galaxy
    distance_bin_width : float : width of each distance bin (in units of distance_scaling)
    distance_bin_number : int : number of distance bins
    distance_scaling : string : 'log', 'linear'
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_position : array : position of galaxy/halo center
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : scaling of y-axis: 'log', 'linear'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    species_test = ['dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6', 'gas', 'star']
    species_reference = 'dark'

    virial_kind = '200m'
    if halo_radius is None:
        halo_radius = np.nan

    Say = ut.io.SayClass(plot_mass_contamination)

    species_test = ut.particle.parse_species(part, species_test)
    center_position = ut.particle.parse_property(part, 'position', center_position)

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, distance_bin_width, distance_bin_number)

    profile_mass = collections.OrderedDict()
    profile_mass[species_reference] = {}
    for spec_name in species_test:
        profile_mass[spec_name] = {}

    profile_mass_ratio = {}
    profile_number = {}

    for spec_name in profile_mass:
        distances = ut.coordinate.get_distances(
            'scalar', part[spec_name]['position'], center_position, part.info['box.length'])
        distances *= part.snapshot['scalefactor']  # convert to {kpc physical}
        if scale_to_halo_radius:
            distances /= halo_radius
        profile_mass[spec_name] = DistanceBin.get_sum_profile(distances, part[spec_name]['mass'])

    for spec_name in species_test:
        mass_ratio_bin = profile_mass[spec_name]['sum'] / profile_mass[species_reference]['sum']
        mass_ratio_cum = (profile_mass[spec_name]['sum.cum'] /
                          profile_mass[species_reference]['sum.cum'])
        profile_mass_ratio[spec_name] = {'bin': mass_ratio_bin, 'cum': mass_ratio_cum}
        profile_number[spec_name] = {
            'bin': np.int64(np.round(profile_mass[spec_name]['sum'] / part[spec_name]['mass'][0])),
            'cum': np.int64(np.round(profile_mass[spec_name]['sum.cum'] /
                                     part[spec_name]['mass'][0])),
        }

    # print diagnostics
    if scale_to_halo_radius:
        distances_halo = profile_mass[species_test[0]]['distance.cum']
        distances_phys = distances_halo * halo_radius
    else:
        distances_phys = profile_mass[species_test[0]]['distance.cum']
        distances_halo = distances_phys / halo_radius

    species_dark = [spec_name for spec_name in species_test if 'dark' in spec_name]

    for spec_name in species_dark:
        Say.say(spec_name)
        if profile_mass[spec_name]['sum.cum'][-1] == 0:
            Say.say('  none. yay!')
            continue

        if scale_to_halo_radius:
            print_string = '  d/R_halo < {:.2f}, d < {:.2f} kpc: '
        else:
            print_string = '  d < {:.2f} kpc, d/R_halo < {:.2f}: '
        print_string += 'mass_frac = {:.3f}, mass = {:.2e}, number = {:.0f}'

        for dist_i in range(profile_mass[spec_name]['sum.cum'].size):
            if profile_mass[spec_name]['sum.cum'][dist_i] > 0:
                if scale_to_halo_radius:
                    distance_0 = distances_halo[dist_i]
                    distance_1 = distances_phys[dist_i]
                else:
                    distance_0 = distances_phys[dist_i]
                    distance_1 = distances_halo[dist_i]

                Say.say(print_string.format(
                        distance_0, distance_1,
                        profile_mass_ratio[spec_name]['cum'][dist_i],
                        profile_mass[spec_name]['sum.cum'][dist_i],
                        profile_number[spec_name]['cum'][dist_i]))

                if spec_name != 'dark.2':
                    # print only 1 distance bin for lower-resolution particles
                    break

    print()
    print('contamination summary')
    spec_name = 'dark.2'
    dist_i_halo = np.searchsorted(distances_phys, halo_radius)
    print('* {} {} particles within R_halo'.format(
          profile_number[spec_name]['cum'][dist_i_halo], spec_name))
    dist_i = np.where(profile_number[spec_name]['cum'] > 0)[0][0]
    print('* {} closest d = {:.1f} kpc, {:.1f} R_halo'.format(
          spec_name, distances_phys[dist_i], distances_halo[dist_i]))
    dist_i = np.where(profile_mass_ratio[spec_name]['cum'] > 0.001)[0][0]
    print('* {} mass_ratio = 0.1% at d < {:.1f} kpc, {:.1f} R_halo'.format(
          spec_name, distances_phys[dist_i], distances_halo[dist_i]))
    dist_i = np.where(profile_mass_ratio[spec_name]['cum'] > 0.01)[0][0]
    print('* {} mass_ratio = 1% at d < {:.1f} kpc, {:.1f} R_halo'.format(
          spec_name, distances_phys[dist_i], distances_halo[dist_i]))
    for spec_name in species_dark:
        if spec_name != 'dark.2' and profile_number[spec_name]['cum'][dist_i_halo] > 0:
            print('! {} {} particles within R_halo'.format(
                  profile_number[spec_name]['cum'][dist_i_halo], spec_name))
            dist_i = np.where(profile_number[spec_name]['cum'] > 0)[0][0]
            print('! {} closest d = {:.1f} kpc, {:.1f} R_halo'.format(
                  spec_name, distances_phys[dist_i], distances_halo[dist_i]))
    print()

    if write_plot is None:
        return

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.96, top=0.96, bottom=0.14)

    ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None, axis_y_scaling, axis_y_limits)

    subplot.set_ylabel(
        '$M_{{\\rm species}} / M_{{\\rm {}}}$'.format(species_reference), fontsize=30)
    if scale_to_halo_radius:
        axis_x_label = '$d \, / \, R_{{\\rm {}}}$'.format(virial_kind)
    else:
        axis_x_label = 'distance $[\\rm kpc]$'
    subplot.set_xlabel(axis_x_label, fontsize=30)

    colors = ut.plot.get_colors(len(species_test), use_black=False)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        subplot.plot([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    for spec_i, spec_name in enumerate(species_test):
        subplot.plot(
            DistanceBin.mids, profile_mass_ratio[spec_name]['bin'], color=colors[spec_i], alpha=0.7,
            label=spec_name)

    legend = subplot.legend(loc='best', prop=FontProperties(size=16))
    legend.get_frame().set_alpha(0.7)

    distance_name = 'dist'
    if halo_radius and scale_to_halo_radius:
        distance_name += '.' + virial_kind
    plot_name = 'mass.ratio_v_{}_z.{:.2f}'.format(distance_name, part.snapshot['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_metal_v_distance(
    parts, spec_name='gas',
    metal_kind='massfraction.metals', axis_y_scaling='log', axis_y_limits=[None, None],
    distance_limits=[10, 3000], distance_bin_width=0.1, distance_bin_number=None,
    distance_scaling='log',
    halo_radius=None, scale_to_halo_radius=False, center_positions=None,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot metallicity (in bin or cumulative) of gas or stars v distance from galaxy.

    Parameters
    ----------
    part : dict or list : catalog[s] of particles at snapshot
    spec_name : string : particle species
    metal_kind : string : 'massfraction.X' or 'mass.X'
    axis_y_scaling : string : scaling of y-axis: 'log', 'linear'
    distance_limits : list : min and max limits for distance from galaxy
    distance_bin_width : float : width of each distance bin (in units of distance_scaling)
    distance_bin_number : int : number of distance bins
    distance_scaling : string : scaling of distance: 'log', 'linear'
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_positions : array : position[s] of galaxy center[s] {kpc comoving}
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    virial_kind = '200m'

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions)

    distance_limits_use = np.array(distance_limits)
    if halo_radius and scale_to_halo_radius:
        distance_limits_use *= halo_radius

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits_use, distance_bin_width, distance_bin_number)

    metal_values = []
    for part_i, part in enumerate(parts):
        distances = ut.coordinate.get_distances(
            'scalar', part[spec_name]['position'], center_positions[part_i],
            part.info['box.length'])
        distances *= part.snapshot['scalefactor']  # convert to {kpc physical}

        metal_mass_kind = metal_kind.replace('massfraction.', 'mass.')
        metal_masses = part[spec_name].prop(metal_mass_kind)

        pro_metal = DistanceBin.get_sum_profile(distances, metal_masses, get_fraction=True)

        if 'massfraction' in metal_kind:
            pro_mass = DistanceBin.get_sum_profile(distances, part[spec_name]['mass'])
            if '.cum' in metal_kind:
                metal_values.append(pro_metal['sum.cum'] / pro_mass['sum.cum'])
            else:
                metal_values.append(pro_metal['sum'] / pro_mass['sum'])
        elif 'mass' in metal_kind:
            if '.cum' in metal_kind:
                metal_values.append(pro_metal['sum.cum'])
            else:
                metal_values.append(pro_metal['sum'])

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.96, top=0.96, bottom=0.14)

    ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None,
        axis_y_scaling, axis_y_limits, metal_values)

    metal_mass_label = 'M_{{\\rm Z,{}}}'.format(spec_name)
    radius_label = '(r)'
    if '.cum' in metal_kind:
        radius_label = '(< r)'
    if 'massfraction' in metal_kind:
        axis_y_label = '${}{} \, / \, M_{{\\rm {}}}{}$'.format(
            metal_mass_label, radius_label, spec_name, radius_label)
    elif 'mass' in metal_kind:
        # axis_y_label = '${}(< r) \, / \, M_{{\\rm Z,tot}}$'.format(metal_mass_label)
        axis_y_label = '${}{} \, [M_\odot]$'.format(metal_mass_label, radius_label)
    #axis_y_label = '$Z \, / \, Z_\odot$'
    subplot.set_ylabel(axis_y_label, fontsize=30)

    if scale_to_halo_radius:
        axis_x_label = '$d \, / \, R_{{\\rm {}}}$'.format(virial_kind)
    else:
        axis_x_label = 'distance $[\\mathrm{kpc}]$'
    subplot.set_xlabel(axis_x_label, fontsize=26)

    colors = ut.plot.get_colors(len(parts), use_black=False)

    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        subplot.plot([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    for part_i, part in enumerate(parts):
        subplot.plot(
            xs, metal_values[part_i], color=colors[part_i], alpha=0.8,
            label=part.info['simulation.name'])

    if len(parts):
        legend = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend.get_frame().set_alpha(0.7)

    distance_name = 'dist'
    if halo_radius and scale_to_halo_radius:
        distance_name += '.' + virial_kind
    plot_name = '{}.{}_v_{}_z.{:.2f}'.format(spec_name, metal_kind, distance_name,
                                             part.info['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# visualize
#===================================================================================================
def plot_image(
    part, spec_name='dark', dimen_indices_plot=[0, 1, 2], dimen_indices_select=[0, 1, 2],
    distance_max=1000, distance_bin_width=1, distance_bin_number=None, center_position=None,
    weight_prop_name='mass', other_prop_limits={}, part_indices=None, subsample_factor=None,
    align_principal_axes=False, use_column_units=None, image_limits=[None, None],
    background_color='white',
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
    use_column_units : boolean : whether to convert to particle number / cm ^ 2
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
    BYW = colors.LinearSegmentedColormap('byw', ut.plot.cmap_dict['BlackYellowWhite'])
    plt.register_cmap(cmap=BYW)
    BBW = colors.LinearSegmentedColormap('bbw', ut.plot.cmap_dict['BlackBlueWhite'])
    plt.register_cmap(cmap=BBW)

    if background_color == 'black':
        if 'dark' in spec_name:
            color_map = plt.get_cmap('bbw')
        elif spec_name == 'star':
            color_map = plt.get_cmap('byw')
    elif background_color == 'white':
        color_map = plt.cm.YlOrBr  # @UndefinedVariable
        #color_map = plt.cm.Greys_r,  # @UndefinedVariable

    if len(dimen_indices_plot) == 2:
        fig, subplot = ut.plot.make_figure(
            figure_index, left=0.17, right=0.96, top=0.96, bottom=0.14,
            background_color=background_color)

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

        # convert to column density
        if use_column_units:
            histogramss *= ut.const.hydrogen_per_sun * ut.const.kpc_per_cm ** 2

        masks = (histogramss > 0)
        Say.say('histogram min, med, max = {:.3e}, {:.3e}, {:.3e}'.format(
                histogramss[masks].min(), np.median(histogramss[masks]), histogramss[masks].max()))

        image_limits_use = [histogramss[masks].min(), histogramss[masks].max()]
        if image_limits is not None and len(image_limits):
            if image_limits[0] is not None:
                image_limits_use[0] = image_limits[0]
            if image_limits[1] is not None:
                image_limits_use[1] = image_limits[1]

        _Image = subplot.imshow(
            histogramss.transpose(),
            norm=colors.LogNorm(),
            cmap=color_map,
            aspect='auto',
            #interpolation='none',
            interpolation='nearest',
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

        fig.gca().set_aspect('equal')

        fig.colorbar(_Image)

        # plot halos
        if hal is not None:
            for hal_position, hal_radius in zip(hal_positions, hal_radiuss):
                print(hal_position, hal_radius)
                circle = plt.Circle(
                    hal_position[dimen_indices_plot], hal_radius, color='w', linewidth=1,
                    fill=False)
                subplot.add_artist(circle)

    elif len(dimen_indices_plot) == 3:
        #position_limits *= 0.999  # ensure that tick labels do not overlap
        position_limits[0, 0] *= 0.99
        position_limits[1, 0] *= 0.99

        fig, subplots = ut.plot.make_figure(
            figure_index, [2, 2], left=0.17, right=0.96, top=0.97, bottom=0.13,
            background_color=background_color)

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

                circle = plt.Circle((0, 0), 10, color='black', fill=False)
                subplot.add_artist(circle)

            subplot.axis('equal')

    plot_name = spec_name
    if weight_prop_name:
        plot_name += '.{}'.format(weight_prop_name)
    plot_name += '.position'

    for dimen_i in dimen_indices_plot:
        plot_name += '.' + dimen_label[dimen_i]
    plot_name += '_d.{:.0f}'.format(distance_max)
    plot_name += '_z.{:.2f}'.format(part.snapshot['redshift'])

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
    parts, spec_name='gas',
    prop_name='density', prop_limits=[], prop_bin_width=None, prop_bin_number=100,
    prop_scaling='log', prop_statistic='probability',
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
    prop_scaling : string : scaling of property: 'log', 'linear'
    prop_statistic : string : statistic to plot:
        'probability', 'probability.cum', 'histogram', 'histogram.cum'
    distance_limits : list : min and max limits for distance from galaxy
    center_positions : array or list of arrays : position[s] of galaxy center[s]
    center_velocities : array or list of arrays : velocity[s] of galaxy center[s]
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : 'log', 'linear'
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
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.18, right=0.95, top=0.96, bottom=0.16)

    y_values = np.array([Stat.distr[prop_statistic][part_i] for part_i in range(len(parts))])
    ut.plot.set_axes_scaling_limits(
        subplot, prop_scaling, prop_limits, prop_values, axis_y_scaling, axis_y_limits, y_values)

    subplot.set_xlabel(ut.plot.get_label(prop_name, species=spec_name, get_units=True))
    subplot.set_ylabel(
        ut.plot.get_label(
            prop_name, prop_statistic, spec_name, get_symbol=True, get_units=False,
            get_log=prop_scaling))

    for part_i, part in enumerate(parts):
        subplot.plot(Stat.distr['bin.mid'][part_i], Stat.distr[prop_statistic][part_i],
                     color=colors[part_i], alpha=0.8, linewidth=3.0,
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

    plot_name = spec_name + '.' + prop_name + '_distr_z.{:.2f}'.format(part.info['redshift'])
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
    x_prop_scaling : string : 'log', 'linear'
    y_prop_name : string : property name for y-axis
    y_prop_limits : list : min and max limits to impose on y_prop_name
    y_prop_scaling : string : 'log', 'linear'
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
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.18, right=0.95, top=0.96, bottom=0.16)

    axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot, x_prop_scaling, x_prop_limits, x_prop_values,
        y_prop_scaling, y_prop_limits, y_prop_values)

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

    plot_name = (spec_name + '.' + y_prop_name + '_v_' + x_prop_name + '_z.{:.2f}'.format(
                 part.info['redshift']))
    if host_distance_limitss is not None and len(host_distance_limitss):
        plot_name += '_d.{:.0f}-{:.0f}'.format(host_distance_limitss[0], host_distance_limitss[1])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_property_v_distance(
    parts, species='dark',
    prop_name='mass', prop_statistic='sum', prop_scaling='log', weight_by_mass=False,
    prop_limits=[],
    distance_limits=[0.1, 300], distance_bin_width=0.02, distance_bin_number=None,
    distance_scaling='log',
    dimension_number=3, rotation_vectors=None,
    axis_distance_max=Inf, other_axis_distance_limits=None,
    center_positions=None, center_velocities=None,
    other_prop_limits={}, part_indicess=None,
    distance_reference=None, plot_nfw=False, label_redshift=True,
    get_values=False, write_plot=False, plot_directory='.', figure_index=1):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshots)
    species : string or list : species to compute total mass of
        options: 'dark', 'star', 'gas', 'baryon', 'total'
    prop_name : string : property to get profile of
    prop_statistic : string : statistic/type to plot:
        'sum, sum.cum, density, density.cum, vel.circ, sum.fraction, sum.cum.fraction,
        median, average'
    prop_scaling : string : scaling for property (y-axis): 'log', 'linear'
    weight_by_mass : boolean : whether to weight property by particle mass
    prop_limits : list : limits to impose on y-axis
    distance_limits : list : min and max distance for binning
    distance_bin_width : float : width of distance bin
    distance_bin_number : int : number of bins between limits
    distance_scaling : string : 'log', 'linear'
    dimension_number : int : number of spatial dimensions for profile
        note : if 1, get profile along minor axis, if 2, get profile along 2 major axes
    rotation_vectors : array : eigen-vectors to define rotation
    axis_distance_max : float : maximum distance to use in defining principal axes {kpc physical}
    other_axis_distance_limits : float :
        min and max distances along other axis[s] to keep particles {kpc physical}
    center_positions : array or list of arrays : position of center for each particle catalog
    center_velocities : array or list of arrays : velocity of center for each particle catalog
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    distance_reference : float : reference distance at which to draw vertical line
    plot_nfw : boolean : whether to overplot NFW profile: density ~ 1 / r
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
            axis_distance_max, other_axis_distance_limits, other_prop_limits, part_indicess[part_i])

        pros.append(pros_part)

        #if part_i > 0:
        #    print(pros[part_i][prop_name] / pros[0][prop_name])

        #print(pros_part[species][prop_statistic])

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.18, right=0.95, top=0.96, bottom=0.16)

    y_values = [pro[species][prop_statistic] for pro in pros]
    _axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None, prop_scaling, prop_limits, y_values)

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

    colors = ut.plot.get_colors(len(parts))

    if 'fraction' in prop_statistic or 'beta' in prop_name or 'velocity.rad' in prop_name:
        if 'fraction' in prop_statistic:
            y_values = [1, 1]
        elif 'beta' in prop_name:
            y_values = [0, 0]
        elif 'velocity.rad' in prop_name:
            y_values = [0, 0]
        subplot.plot(
            distance_limits, y_values, color='black', linestyle=':', alpha=0.5, linewidth=2)

    if distance_reference is not None:
        subplot.plot([distance_reference, distance_reference], axis_y_limits,
                     color='black', linestyle=':', alpha=0.6)

    if plot_nfw:
        pro = pros[0]
        distances_nfw = pro[species]['distance']
        # normalize to outermost distance bin
        densities_nfw = np.ones(pro[species]['distance'].size) * pro[species][prop_statistic][-1]
        densities_nfw *= pro[species]['distance'][-1] / pro[species]['distance']
        subplot.plot(distances_nfw, densities_nfw, color='black', linestyle=':', alpha=0.6)

    # plot profiles of objects
    if len(pros) == 1:
        alpha = 1.0
        linewidth = 3.5
    else:
        alpha = 0.7
        linewidth = 2.5

    for part_i, pro in enumerate(pros):
        print(pro[species][prop_statistic])
        linestyle = '-'
        color = colors[part_i]
        if 'res-adapt' in parts[part_i].info['simulation.name']:
            linestyle = '--'
            color = colors[part_i - 1]
        masks = pro[species][prop_statistic] != 0  # plot only non-zero values
        subplot.plot(pro[species]['distance'][masks], pro[species][prop_statistic][masks],
                     color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth,
                     label=parts[part_i].info['simulation.name'])

    # redshift legend
    legend_z = None
    if label_redshift:
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

    plot_name = (species + '.' + prop_name + '.' + prop_statistic + '_v_dist_z.{:.2f}'.format(
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
# properties of halos
#===================================================================================================
def assign_vel_circ_at_radius(
    hal, part, radius=0.4, sort_prop_name='vel.circ.max', sort_prop_value_min=20,
    halo_number_max=100, host_distance_limits=[1, 310]):
    '''
    .
    '''
    Say = ut.io.SayClass(assign_vel_circ_at_radius)

    his = ut.array.get_indices(hal.prop('mass.bound/mass.200m'), [0.1, Inf])
    his = ut.array.get_indices(hal['host.distance'], host_distance_limits, his)
    his = ut.array.get_indices(hal[sort_prop_name], [sort_prop_value_min, Inf], his)
    Say.say('{} halos within limits'.format(his.size))

    his = his[np.argsort(hal[sort_prop_name][his])]
    his = his[::-1][: halo_number_max]

    mass_key = 'vel.circ.rad.{:.1f}'.format(radius)
    hal[mass_key] = np.zeros(hal['total.mass'].size)
    dark_mass = np.median(part['dark']['mass'])

    for hii, hi in enumerate(his):
        if hii > 0 and hii % 10 == 0:
            ut.io.print_flush(hii)
        pis = ut.particle.get_indices_within_distances(
            part, 'dark', [0, radius], hal['position'][hi], scalarize=True)
        hal[mass_key][hi] = ut.halo_property.get_circular_velocity(pis.size * dark_mass, radius)


def plot_vel_circ_v_radius_halos(
    parts=None, hals=None, hal_indicess=None, part_indicesss=None,
    gal=None,
    total_mass_limits=None, star_mass_limits=[1e5, Inf], host_distance_limits=[1, 310],
    sort_prop_name='vel.circ.max', sort_prop_value_min=15, halo_number_max=20,
    vel_circ_limits=[0, 50], vel_circ_scaling='linear',
    pros=None,
    radius_limits=[0.1, 3], radius_bin_width=0.1, radius_bin_number=None, radius_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    .
    '''
    if isinstance(hals, dict):
        hals = [hals]
    if hal_indicess is not None:
        if np.isscalar(hal_indicess):
            hal_indicess = [hal_indicess]
        if np.isscalar(hal_indicess[0]):
            hal_indicess = [hal_indicess]

    Say = ut.io.SayClass(plot_vel_circ_v_radius_halos)

    hiss = None
    if hals is not None:
        hiss = []
        for cat_i, hal in enumerate(hals):
            his = None
            if hal_indicess is not None:
                his = hal_indicess[cat_i]
            his = ut.array.get_indices(hal.prop('mass.bound/mass.200m'), [0.1, Inf], his)
            his = ut.array.get_indices(hal['total.mass'], total_mass_limits, his)
            his = ut.array.get_indices(hal['host.distance'], host_distance_limits, his)

            if 'star.indices' in hal:
                his = ut.array.get_indices(hal['star.mass.part'], star_mass_limits, his)
            else:
                his = ut.array.get_indices(hal[sort_prop_name], [sort_prop_value_min, Inf], his)
                his = his[np.argsort(hal[sort_prop_name][his])[::-1]]
                his = his[: halo_number_max]

                Say.say('{} halos with {} [min, max] = [{:.3f}, {:.3f}]'.format(
                        his.size, sort_prop_name,
                        hal[sort_prop_name][his[0]], hal[sort_prop_name][his[-1]]))

            hiss.append(his)

    gal_indices = None
    if gal is not None:
        gal_indices = ut.array.get_indices(gal['star.mass'], star_mass_limits)
        gal_indices = ut.array.get_indices(gal['host.distance'], host_distance_limits, gal_indices)
        gal_indices = gal_indices[gal['host.name'][gal_indices] == 'MW'.encode()]

    pros = plot_property_v_distance_halos(
        parts, hals, hiss, part_indicesss,
        gal, gal_indices,
        'total', 'mass', 'vel.circ', vel_circ_scaling, False, vel_circ_limits,
        radius_limits, radius_bin_width, radius_bin_number, radius_scaling, 3,
        pros,
        None, False,
        write_plot, plot_directory, figure_index)

    """
    plot_property_v_distance_halos(
        parts, hals, hiss, part_indicesss
        gal, gal_indices,
        'star', 'velocity.tot', 'std.cum', vel_circ_scaling, True, vel_circ_limits,
        radius_limits, radius_bin_width, radius_bin_number, radius_scaling, 3, None, False,
        write_plot, plot_directory, figure_index)
    """

    return pros


def plot_property_v_distance_halos(
    parts=None, hals=None, hal_indicess=None, part_indicesss=None,
    gal=None, gal_indices=None,
    species='total',
    prop_name='mass', prop_statistic='vel.circ', prop_scaling='linear', weight_by_mass=False,
    prop_limits=[],
    distance_limits=[0.1, 3], distance_bin_width=0.1, distance_bin_number=None,
    distance_scaling='log', dimension_number=3,
    pros=None,
    distance_reference=None, label_redshift=True,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    parts : dict or list : catalog[s] of particles at snapshot
    hals : dict or list : catalog[s] of halos at snapshot
    hal_indicess : array (halo catalog number x halo number) : indices of halos to plot
    part_indicesss : array (halo catalog number x halo number x particle number) :
    gal : dict : catalog of observed galaxies
    gal_indices : array : indices of galaxies to plot
    species : string or list : species to compute total mass of
        options: 'dark', 'star', 'gas', 'baryon', 'total'
    prop_name : string : property to get profile of
    prop_statistic : string : statistic/type to plot:
        'sum', sum.cum, density, density.cum, vel.circ, sum.fraction, sum.cum.fraction, med, ave'
    prop_scaling : string : scaling for property (y-axis): 'log', 'linear'
    weight_by_mass : boolean : whether to weight property by particle mass
    prop_limits : list : limits to impose on y-axis
    distance_limits : list : min and max distance for binning
    distance_bin_width : float : width of distance bin
    distance_bin_number : int : number of bins between limits
    distance_scaling : string : 'log', 'linear'
    dimension_number : int : number of spatial dimensions for profile
    distance_reference : float : reference distance at which to draw vertical line
    label_redshift : boolean : whether to label redshift
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    if isinstance(hals, dict):
        hals = [hals]
    if hal_indicess is not None:
        if np.isscalar(hal_indicess):
            hal_indicess = [hal_indicess]
        if np.isscalar(hal_indicess[0]):
            hal_indicess = [hal_indicess]
    if isinstance(parts, dict):
        parts = [parts]

    # widen so curves extend to edge of figure
    distance_limits_bin = [distance_limits[0] - distance_bin_width,
                           distance_limits[1] + distance_bin_width]

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits_bin, width=distance_bin_width,
        number=distance_bin_number, dimension_number=dimension_number)

    SpeciesProfile = SpeciesProfileClass()

    if pros is None:
        pros = []
        if hals is not None:
            for cat_i, hal in enumerate(hals):
                part = parts[cat_i]
                hal_indices = hal_indicess[cat_i]

                if species == 'star' and 'star' in part:
                    position_kind = 'star.position'
                    velocity_kind = 'star.velocity'
                elif species == 'dark' and 'dark' in part:
                    position_kind = 'dark.position'
                    velocity_kind = 'dark.velocity'
                else:
                    position_kind = 'position'
                    velocity_kind = 'velocity'

                pros_cat = []

                for hal_i in hal_indices:
                    if part_indicesss is not None:
                        part_indices = part_indicesss[cat_i][hal_i]
                    elif species == 'star' and 'star.indices' in hal:
                        part_indices = hal['star.indices'][hal_i]
                    elif species == 'dark' and 'dark.indices' in hal:
                        part_indices = hal['dark.indices'][hal_i]
                    else:
                        part_indices = None

                    pro_hal = SpeciesProfile.get_profiles(
                        part, species, prop_name, prop_statistic, weight_by_mass, DistanceBin,
                        hal[position_kind][hal_i], hal[velocity_kind][hal_i],
                        part_indicess=part_indices)

                    pros_cat.append(pro_hal)
                pros.append(pros_cat)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.18, right=0.95, top=0.96, bottom=0.16)

    y_values = []
    for pro_cat in pros:
        for pro_hal in pro_cat:
            y_values.append(pro_hal[species][prop_statistic])

    ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None, prop_scaling, prop_limits, y_values)

    if 'log' in distance_scaling:
        subplot.xaxis.set_ticks([0.1, 0.2, 0.3, 0.5, 1, 2])
        subplot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    else:
        subplot.xaxis.set_minor_locator(AutoMinorLocator(2))

    #subplot.yaxis.set_minor_locator(AutoMinorLocator(5))

    subplot.set_xlabel('radius $r$ $[\\mathrm{kpc}]$', fontsize=30)
    if prop_statistic in ['vel.circ']:
        label_prop_name = prop_statistic
    else:
        label_prop_name = prop_name
    axis_y_label = ut.plot.get_label(
        label_prop_name, prop_statistic, species, dimension_number, get_symbol=True, get_units=True)
    #if prop_statistic == 'vel.circ':
    #    axis_y_label = 'circular velocity ' + axis_y_label
    subplot.set_ylabel(axis_y_label, fontsize=30)

    # draw reference values
    if 'fraction' in prop_statistic or 'beta' in prop_name or 'velocity.rad' in prop_name:
        if 'fraction' in prop_statistic:
            y_values = [1, 1]
        elif 'beta' in prop_name:
            y_values = [0, 0]
        elif 'velocity.rad' in prop_name:
            y_values = [0, 0]
        subplot.plot(
            distance_limits, y_values, color='black', linestyle=':', alpha=0.5, linewidth=2)

    if distance_reference is not None:
        subplot.plot([distance_reference, distance_reference], prop_limits,
                     color='black', linestyle=':', alpha=0.6)

    # draw simulation halos
    if hals is not None:
        colors = ut.plot.get_colors(len(hals))
        for cat_i, hal in enumerate(hals):
            hal_indices = hal_indicess[cat_i]
            for hal_ii, hal_i in enumerate(hal_indices):
                color = colors[cat_i]
                linewidth = 1.9
                alpha = 0.5
                if pros[cat_i][hal_ii][species][prop_statistic][0] > 12:  # dark vel.circ
                    color = ut.plot.get_color('blue.lite')
                    linewidth = 3.0
                    alpha = 0.8
                if species == 'star':
                    linewidth = 2.0
                    alpha = 0.6
                    color = ut.plot.get_color('orange.mid')
                    if pros[cat_i][hal_ii][species][prop_statistic][0] > 27:
                        color = ut.plot.get_color('orange.lite')
                        linewidth = 3.5
                        alpha = 0.9

                subplot.plot(
                    pros[cat_i][hal_ii][species]['distance'],
                    pros[cat_i][hal_ii][species][prop_statistic],
                    color=color,
                    linestyle='-', alpha=alpha, linewidth=linewidth,
                    #label=parts[part_i].info['simulation.name'],
                )

    # draw observed galaxies
    if gal is not None:
        alpha = 0.7
        linewidth = 2.0
        gis = ut.array.get_indices(gal['star.radius.50'], distance_limits, gal_indices)
        gis = gis[gal['host.name'][gis] == 'MW'.encode()]
        print(gal['vel.circ.50'][gis] / gal['star.vel.std'][gis])
        for gal_i in gis:
            subplot.errorbar(
                gal['star.radius.50'][gal_i],
                gal['vel.circ.50'][gal_i],
                [[gal['vel.circ.50.err.lo'][gal_i]], [gal['vel.circ.50.err.hi'][gal_i]]],
                color='black', marker='s', markersize=10, alpha=alpha,
                linewidth=2.5, capthick=2.5,
            )

    # redshift legend
    legend_z = None
    if label_redshift:
        legend_z = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$z={:.1f}$'.format(parts[0].snapshot['redshift'])],
            loc='lower left', prop=FontProperties(size=16))
        legend_z.get_frame().set_alpha(0.5)

    """
    if len(parts) > 1 and parts[0].info['simulation.name']:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)
    """

    plot_name = species + '.' + prop_name + '.' + prop_statistic + '_v_dist'
    if parts is not None:
        plot_name += '_z.{:.2f}'.format(parts[0].info['redshift'])
    plot_name = plot_name.replace('.sum', '')
    plot_name = plot_name.replace('mass.vel.circ', 'vel.circ')
    plot_name = plot_name.replace('mass.density', 'density')
    ut.plot.parse_output(write_plot, plot_directory, plot_name)

    return pros


#===================================================================================================
# mass and star-formation history
#===================================================================================================
def get_time_bin_dictionary(
    time_kind='redshift', time_limits=[0, 10], time_width=0.01, time_scaling='linear',
    Cosmology=None):
    '''
    Get dictionary of time bin information.

    Parameters
    ----------
    time_kind : string : time metric to use: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to impose
    time_width : float : width of time_kind bin (in units set by time_scaling)
    time_scaling : string : scaling of time_kind: 'log', 'linear'
    Cosmology : class : cosmology class, to convert between time metrics

    Returns
    -------
    time_dict : dict
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
    part, time_kind='redshift', time_limits=[0, 8], time_width=0.1, time_scaling='linear',
    distance_limits=None, center_position=None, other_prop_limits=None, part_indices=None):
    '''
    Get array of times and star-formation rate at each time.

    Parameters
    ----------
    part : dict : dictionary of particles
    time_kind : string : time metric to use: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to impose
    time_width : float : width of time_kind bin (in units set by time_scaling)
    time_scaling : string : scaling of time_kind: 'log', 'linear'
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
    time_kind='time.lookback', time_limits=[0, 13.8], time_width=0.2, time_scaling='linear',
    distance_limits=[0, 10], center_positions=None, other_prop_limits={}, part_indicess=None,
    sfh_limits=[], sfh_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot star-formation rate history v time_kind.
    Note: assumes instantaneous recycling of 30% for mass, should fix this for mass lass v time.

    Parameters
    ----------
    parts : dict or list : catalog[s] of particles
    sfh_kind : string : star form kind to plot: 'rate', 'rate.specific', 'mass', 'mass.normalized'
    time_kind : string : time kind to use: 'time', 'time.lookback' (wrt z = 0), 'redshift'
    time_limits : list : min and max limits of time_kind to get
    time_width : float : width of time_kind bin
    time_scaling : string : scaling of time_kind: 'log', 'linear'
    distance_limits : list : min and max limits of distance to select star particles
    center_positions : list or list of lists : position[s] of galaxy centers {kpc comoving}
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    part_indicess : array : part_indices of particles from which to select
    sfh_limits : list : min and max limits for y-axis
    sfh_scaling : string : scaling of y-axis: 'log', 'linear'
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
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.95, top=0.86, bottom=0.15)

    ut.plot.make_axis_time(
        subplot, time_kind, time_limits, time_scaling, label_axis_2=True, Cosmology=part.Cosmology,
        fontsize=30)

    y_values = None
    if sfh is not None:
        y_values = sfh[sfh_kind]
    ut.plot.set_axes_scaling_limits(
        subplot, y_scaling=sfh_scaling, y_limits=sfh_limits, y_values=y_values)

    if sfh_kind == 'mass.normalized':
        axis_y_label = '$M_{\\rm star}(z)\, / \, M_{\\rm star}(z=0)$'
    else:
        axis_y_label = ut.plot.get_label('star.' + sfh_kind, get_symbol=True, get_units=True)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    colors = ut.plot.get_colors(len(parts))

    for part_i, part in enumerate(parts):
        tis = (sfh[sfh_kind][part_i] > 0)
        if time_kind in ['redshift', 'time.lookback']:
            tis *= (sfh[time_kind][part_i] >= parts[0].snapshot[time_kind] * 0.99)
        else:
            tis *= (sfh[time_kind][part_i] <= parts[0].snapshot[time_kind] * 1.01)
        subplot.plot(sfh[time_kind][part_i][tis], sfh[sfh_kind][part_i][tis],
                     linewidth=2.5, color=colors[part_i], alpha=0.8,
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

    sfh_name = 'star' + '.' + sfh_kind + '.history'
    plot_name = '{}_v_{}_z.{:.2f}'.format(sfh_name, time_kind, part.info['redshift'])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


def plot_star_form_history_galaxies(
    part=None, hal=None, gal=None,
    mass_kind='star.mass.part', mass_limits=[1e5, 1e9], other_prop_limits={}, hal_indices=None,
    sfh_kind='mass.normalized', sfh_limits=[], sfh_scaling='linear',
    time_kind='time.lookback', time_limits=[13.7, 0], time_width=0.2, time_scaling='linear',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot star-formation history v time_kind.
    Note: assumes instantaneous recycling of 30% of mass, should fix this for mass lass v time.

    Parameters
    ----------
    part : dict : catalog of particles
    hal : dict : catalog of halos at snapshot
    gal : dict : catalog of galaxies in the Local Group with SFHs
    mass_kind : string : mass kind by which to select halos
    mass_limits : list : min and max limits to impose on mass_kind
    other_prop_limits : dict : dictionary with properties as keys and limits as values
    hal_indices : index or array : index[s] of halo[s] whose particles to plot
    sfh_kind : string : star form kind to plot: 'rate', 'rate.specific', 'mass', 'mass.normalized'
    sfh_limits : list : min and max limits for y-axis
    sfh_scaling : string : scailng of y-axis: 'log', 'linear'
    time_kind : string : time kind to plot: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to plot
    time_width : float : width of time_kind bin
    time_scaling : string : scaling of time_kind: 'log', 'linear'
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_star_form_history_galaxies)

    time_limits = np.array(time_limits)
    if part is not None:
        if time_limits[0] is None:
            time_limits[0] = part.snapshot[time_kind]
        if time_limits[1] is None:
            time_limits[1] = part.snapshot[time_kind]

    sfh = None
    if hal is not None:
        if hal_indices is None or not len(hal_indices):
            hal_indices = ut.array.get_indices(hal.prop('star.number'), [2, Inf])

        if mass_limits is not None and len(mass_limits):
            hal_indices = ut.array.get_indices(hal.prop(mass_kind), mass_limits, hal_indices)

        if other_prop_limits:
            hal_indices = ut.catalog.get_indices_catalog(hal, other_prop_limits, hal_indices)

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

            Say.say(
                'id = {:8d}, star.mass = {:.3e}, particle.number = {}, distance = {:.0f}'.format(
                    hal_i, sfh_h['mass'].max(), part_indices.size,
                    hal.prop('host.distance', hal_i)))
            #print(hal.prop('position', hal_i))

        for k in sfh:
            sfh[k] = np.array(sfh[k])

        sfh['mass.normalized.median'] = np.median(sfh['mass.normalized'], 0)

    if time_kind == 'redshift' and 'log' in time_scaling:
        time_limits += 1  # convert to z + 1 so log is well-defined

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.95, top=0.86, bottom=0.15)

    ut.plot.make_axis_time(
        subplot, time_kind, time_limits, time_scaling, label_axis_2=True, Cosmology=part.Cosmology,
        fontsize=30)

    y_values = None
    if sfh is not None:
        y_values = sfh[sfh_kind]
    ut.plot.set_axes_scaling_limits(
        subplot, y_scaling=sfh_scaling, y_limits=sfh_limits, y_values=y_values)
    subplot.xaxis.set_minor_locator(AutoMinorLocator(2))

    if sfh_kind == 'mass.normalized':
        axis_y_label = '$M_{\\rm star}(z)\, / \, M_{\\rm star}(z=0)$'
    else:
        axis_y_label = ut.plot.get_label('star.' + sfh_kind, get_symbol=True, get_units=True)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    if hal is not None:
        colors = ut.plot.get_colors(len(hal_indices))
    elif gal is not None:
        colors = ut.plot.get_colors(len(gal.sfh))

    label = None

    # draw observed galaxies
    if gal is not None:
        import string
        gal_names = np.array(list(gal.sfh.keys()))
        gal_indices = [gal['name.to.index'][gal_name] for gal_name in gal_names]
        gal_names_sort = gal_names[np.argsort(gal['star.mass'][gal_indices])]

        for gal_i, gal_name in enumerate(gal_names_sort):
            linestyle = '-'
            if hal is not None:
                color = 'black'
                linewidth = 1.0 + 0.25 * gal_i
                alpha = 0.2
                label = None
            else:
                color = colors[gal_i]
                linewidth = 1.25 + 0.25 * gal_i
                alpha = 0.45
                label = string.capwords(gal_name)
                label = label.replace('Canes Venatici I', 'CVn I').replace('Ii', 'II')

                print(label)
            subplot.plot(gal.sfh[gal_name][time_kind], gal.sfh[gal_name][sfh_kind],
                         linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color,
                         label=label)

    # draw simulated galaxies
    if hal is not None:
        for hal_ii, hal_i in enumerate(hal_indices):
            linewidth = 2.5 + 0.1 * hal_ii
            #linewidth = 3.0
            label = '$M_{{\\rm star}}={}\,M_\odot$'.format(
                ut.io.get_string_for_exponential(sfh['mass'][hal_ii][-1], 0))
            subplot.plot(sfh[time_kind][hal_ii], sfh[sfh_kind][hal_ii],
                         linewidth=linewidth, color=colors[hal_ii], alpha=0.55, label=label)

    #subplot.plot(sfh['time'][0], sfh['mass.normalized.median'],
    #             linewidth=4.0, color='black', alpha=0.5)

    # redshift legend
    legend_z = None
    if part is not None and part.snapshot['redshift'] > 0.03:
        legend_z = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')],
            ['$z={:.1f}$'.format(part.snapshot['redshift'])],
            loc='lower left', prop=FontProperties(size=16))
        legend_z.get_frame().set_alpha(0.5)

    # property legend
    if label is not None:
        legend_prop = subplot.legend(
            loc='best', prop=FontProperties(size=15), handlelength=1.3, labelspacing=0.1)
        legend_prop.get_frame().set_alpha(0.5)
        if legend_z:
            subplot.add_artist(legend_z)

    sf_name = 'star' + '.' + sfh_kind
    plot_name = '{}_v_{}'.format(sf_name, time_kind)
    if gal is not None:
        plot_name += '_lg'
    if hal is not None:
        plot_name += '_z.{:.2f}'.format(part.snapshot['redshift'])
    if 'host.distance' in other_prop_limits:
        plot_name += '_d.{:.0f}-{:.0f}'.format(
            other_prop_limits['host.distance'][0], other_prop_limits['host.distance'][1])
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# use halo catalog
#===================================================================================================
def explore_galaxy(
    hal, hal_index=None, part=None, species_plot=['star'],
    distance_max=None, distance_bin_width=0.25, distance_bin_number=None, plot_only_members=False,
    write_plot=False, plot_directory='.'):
    '''
    Print and plot several properties of galaxies in list.

    Parameters
    ----------
    hal : dict : catalog of halos at snapshot
    part : dict : catalog of particles at snapshot
    distance_max : float : max distance (radius) for galaxy image
    distance_bin_width : float : length of pixel for galaxy image
    distance_bin_number : int : number of pixels for galaxy image
    plot_only_members : boolean : whether to plat only particles that are members of halo
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    '''
    from rockstar import rockstar_analysis

    rockstar_analysis.print_properties_halo(part, hal, hal_index)

    hi = hal_index

    if part is not None:
        if not distance_max and 'star.radius.50' in hal:
            distance_max = 4 * hal.prop('star.radius.50', hi)

        if 'star' in species_plot and 'star' in part and 'star.indices' in hal:
            if plot_only_members:
                part_indices = hal.prop('star.indices', hi)
            else:
                part_indices = None

            plot_image(
                part, 'star', [0, 1, 2], [0, 1, 2], distance_max, distance_bin_width,
                distance_bin_number, hal.prop('star.position', hi), 'mass',
                part_indices=part_indices,
                write_plot=write_plot, plot_directory=plot_directory, figure_index=1)

            plot_property_distribution(
                part, 'star', 'velocity.tot', [0, None], 2, None, 'linear', 'histogram',
                [], hal.prop('star.position', hi), hal.prop('star.velocity', hi), {}, part_indices,
                [0, None], 'linear', write_plot, plot_directory, figure_index=2)

            try:
                element_name = 'metallicity.iron'
                hal.prop(element_name)
            except:
                element_name = 'metallicity.metals'

            plot_property_distribution(
                part, 'star', element_name, [1e-4, 1], 0.1, None, 'log', 'histogram',
                [], None, None, {}, part_indices,
                [0, None], 'linear', write_plot, plot_directory, figure_index=3)

            plot_property_v_distance(
                part, 'star', 'mass', 'density', 'log', False, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=hal.prop('star.position', hi), part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                write_plot=write_plot, plot_directory=plot_directory, figure_index=4)

            plot_property_v_distance(
                part, 'star', 'mass', 'sum.cum', 'log', False, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=hal.prop('star.position', hi), part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                write_plot=write_plot, plot_directory=plot_directory, figure_index=5)

            plot_property_v_distance(
                part, 'star', 'velocity.tot', 'std.cum', 'linear', True, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=hal.prop('star.position', hi),
                center_velocities=hal.prop('star.velocity', hi),
                part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                write_plot=write_plot, plot_directory=plot_directory, figure_index=6)

            plot_property_v_distance(
                part, 'star', element_name, 'median', 'linear', True, None,
                [0.1, distance_max], 0.2, None, 'log', 3,
                center_positions=hal.prop('star.position', hi), part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                write_plot=write_plot, plot_directory=plot_directory, figure_index=7)

            plot_star_form_history(
                part, 'mass.normalized', 'time.lookback', [13.6, 0], 0.2, 'linear', [], None, {},
                part_indices, [0, 1], 'linear', write_plot, plot_directory, figure_index=8)

        if 'dark' in species_plot and 'dark' in part and 'dark.indices' in hal:
            if plot_only_members:
                part_indices = hal.prop('dark.indices', hi)
            else:
                part_indices = None

            if 'star.position' in hal:
                center_position = hal.prop('star.position', hi)
                center_velocity = hal.prop('star.velocity', hi)
            elif 'star.position' in hal:
                center_position = hal.prop('dark.position', hi)
                center_velocity = hal.prop('dark.velocity', hi)
            else:
                center_position = hal.prop('position', hi)
                center_velocity = hal.prop('velocity', hi)

            if 'star.radius.50' in hal:
                distance_reference = hal.prop('star.radius.50', hi)
            else:
                distance_reference = None

            plot_property_v_distance(
                part, 'dark', 'mass', 'density', 'log', False, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=center_position, part_indicess=part_indices,
                distance_reference=distance_reference,
                write_plot=write_plot, plot_directory=plot_directory, figure_index=10)

            plot_property_v_distance(
                part, 'dark', 'velocity.tot', 'std.cum', 'linear', True, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=center_position, center_velocities=center_velocity,
                part_indicess=part_indices,
                distance_reference=distance_reference,
                write_plot=write_plot, plot_directory=plot_directory, figure_index=11)

            plot_property_v_distance(
                part, 'dark', 'mass', 'vel.circ', 'linear', True, None,
                [0.1, distance_max], 0.1, None, 'log', 3,
                center_positions=center_position, part_indicess=part_indices,
                distance_reference=distance_reference,
                write_plot=write_plot, plot_directory=plot_directory, figure_index=12)

        if 'gas' in species_plot and 'gas' in part and 'gas.indices' in hal:
            part_indices = None
            if plot_only_members:
                part_indices = hal.prop('gas.indices', hi)

            if part_indices is None or len(part_indices) >= 3:
                plot_image(
                    part, 'gas', [0, 1, 2], [0, 1, 2], distance_max, distance_bin_width,
                    distance_bin_number, hal.prop('star.position', hi), 'mass',
                    part_indices=part_indices,
                    write_plot=write_plot, plot_directory=plot_directory, figure_index=20)

                plot_image(
                    part, 'gas', [0, 1, 2], [0, 1, 2], distance_max, distance_bin_width,
                    distance_bin_number, hal.prop('star.position', hi), 'mass.neutral',
                    part_indices=part_indices,
                    write_plot=write_plot, plot_directory=plot_directory, figure_index=21)
            else:
                fig = plt.figure(10)
                fig.clf()
                fig = plt.figure(11)
                fig.clf()


def plot_density_profile_halo(
    part,
    hal=None, hal_index=None, center_position=None,
    species='star',
    distance_limits=[0.1, 2], distance_bin_width=0.1, distance_bin_number=None,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot density profile for single halo/center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    hal : dict : catalog of halos at snapshot
    hal_index : int : index of halo in catalog
    center_position : array : position to center profile (to use instead of halo position)
    distance_max : float : max distance (radius) for galaxy image
    distance_bin_width : float : length of pixel for galaxy image
    distance_bin_number : int : number of pixels for galaxy image
    plot_only_members : boolean : whether to plat only particles that are members of halo
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    '''
    distance_scaling = 'log'
    dimension_number = 3

    if center_position is None:
        center_positions = []
        #center_positions.append(hal['position'][hal_index])
        if 'star.position' in hal and hal['star.position'][hal_index][0] > 0:
            center_positions.append(hal['star.position'][hal_index])
    else:
        center_positions = [center_position]

    parts = [part]
    if len(center_positions) == 2:
        parts = [part, part]

    if 'star.radius.50' in hal and hal['star.radius.50'][hal_index] > 0:
        distance_reference = hal['star.radius.50'][hal_index]
    else:
        distance_reference = None

    plot_property_v_distance(
        parts, species, 'mass', 'density', 'log', False, None,
        distance_limits, distance_bin_width, distance_bin_number, distance_scaling,
        dimension_number,
        center_positions=center_positions, part_indicess=None,
        distance_reference=distance_reference,
        write_plot=write_plot, plot_directory=plot_directory, figure_index=figure_index)


def plot_density_profiles_halos(
    part, hal, hal_indices,
    species='dark', density_limits=None,
    distance_limits=[0.05, 1], distance_bin_width=0.2, distance_bin_number=None,
    plot_only_members=False,
    write_plot=False, plot_directory='.', figure_index=0):
    '''
    write_plot : boolean : whether to write figure to file
    plot_directory : string : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    parts = []
    center_positions = []
    part_indicess = None
    for hal_i in hal_indices:
        parts.append(part)
        if 'star.position' in hal:
            center_positions.append(hal.prop('star.position', hal_i))
            if plot_only_members:
                part_indicess.append(hal.prop(species + '.indices', hal_i))
        else:
            center_positions.append(hal.prop('position', hal_i))
            if plot_only_members:
                part_indicess.append(hal.prop(species + '.indices', hal_i))

    plot_property_v_distance(
        parts, species, 'mass', 'density', 'log', False, density_limits,
        distance_limits, distance_bin_width, distance_bin_number, 'log', 3,
        center_positions=center_positions, part_indicess=part_indicess,
        write_plot=write_plot, plot_directory=plot_directory, figure_index=figure_index)


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
        part = gizmo_io.Read.read_snapshot(
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
    time_kind='redshift', time_limits=[0, 8], time_scaling='linear', snapshot_subsample_factor=1,
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
    time_scaling : string : scaling of time_kind: 'log', 'linear'
    snapshot_subsample_factor : int : factor by which to sub-sample snapshots from gals
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : string : scaling of y-axis: 'log', 'linear'
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
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.17, right=0.95, top=0.86, bottom=0.15)

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

    if gals is not None:
        for _gal_i, gal in enumerate(gals):
            subplot.plot(
                gal[time_kind][::snapshot_subsample_factor],
                gal[prop_name][::snapshot_subsample_factor],
                linewidth=3.0, alpha=0.9,
                #color=colors[gal_i],
                color=ut.plot.get_color('blue.mid'),
                label='main progenitor',
            )

    if sfhs is not None:
        for _sfh_i, sfh in enumerate(sfhs):
            subplot.plot(
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

    plot_name = 'galaxy_{}_v_{}'.format(prop_name, time_kind)
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# disk mass and radius over time, for

#===================================================================================================
def get_galaxy_mass_profiles_v_redshift(
    directory='.', redshifts=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0],
    parts=None):
    '''
    Read snapshots and store dictionary of galaxy/halo position, velocity, size, mass at input
    scale-factors, for Shea.

    Parameters
    ----------
    directory : string : directory of snapshot files
    redshifts : array-like : redshifts at which to get properties

    Returns
    -------
    dictionary of galaxy/halo properties at each redshift
    '''
    from . import gizmo_io

    species_read = ['star', 'dark']
    properties_read = ['mass', 'position', 'velocity', 'potential']

    star_distance_max = 20
    dark_distance_max = 50

    profile_spec_name = 'star'
    profile_mass_percents = [50, 90]

    gal = {
        'index': [],  # snapshot index
        'redshift': [],  # snapshot redshift
        'scalefactor': [],  # snapshot scale-factor
        'time': [],  # snapshot time [Gyr]
        'time.lookback': [],  # snapshot lookback time [Gyr]

        'star.position': [],  # position of galaxy (star) center [kpc comoving]
        'star.velocity': [],  # center-of-mass velocity of stars within R_50 [km/s physical]
        'dark.position': [],  # position of DM center [kpc comoving]
        'dark.velocity': [],  # center-of-mass velocity of DM within 0.5 * R_200m [km/s physical]

        'rotation.tensor': [],  # rotation tensor of disk
        'axis.ratio': [],  # axis ratios of disk

        'profile.3d.distance': [],  # distance bins in 3-D [kpc physical]
        'profile.3d.density': [],  # density, in 3-D [M_sun / kpc ^ 3]

        'profile.major.distance': [],  # distance bins along major (R) axis [kpc physical]
        'profile.major.density': [],  # surface density, in 2-D [M_sun / kpc ^ 2]

        'profile.minor.bulge.distance': [],  # distance bins along minor (z) axis [kpc physical]
        'profile.minor.bulge.density': [],  # density, in 1-D [M_sun / kpc]

        'profile.minor.disk.distance': [],  # distance bins along minor (z) axis [kpc physical]
        'profile.minor.disk.density': [],  # density, in 1-D [M_sun / kpc]
    }

    for mass_percent in profile_mass_percents:
        mass_percent_name = '{:.0f}'.format(mass_percent)

        gal['radius.3d.' + mass_percent_name] = []  # stellar R_{50,90} in 3-D [kpc physical]
        gal['mass.3d.' + mass_percent_name] = []  # associated stellar mass [M_sun}

        gal['radius.major.' + mass_percent_name] = []  # stellar R_{50,90} along major axis
        gal['mass.major.' + mass_percent_name] = []  # associated stellar mass [M_sun]

        gal['radius.minor.' + mass_percent_name] = []  # stellar R_{50,90} along minor axis
        gal['mass.minor.' + mass_percent_name] = []  # associated stellar mass [M_sun]

    for zi, redshift in enumerate(redshifts):
        if parts is not None and len(parts):
            part = parts[zi]
        else:
            part = gizmo_io.Read.read_snapshot(
                species_read, 'redshift', redshift, directory, property_names=properties_read,
                force_float32=True)

        for k in ['index', 'redshift', 'scalefactor', 'time', 'time.lookback']:
            gal[k].append(part.snapshot[k])

        # get position and velocity
        gal['star.position'].append(part.center_position)
        gal['star.velocity'].append(part.center_velocity)

        gal['dark.position'].append(ut.particle.get_center_position(part, 'dark', 'potential'))
        gal['dark.velocity'].append(
            ut.particle.get_center_velocity(part, 'dark', distance_max=dark_distance_max))

        # get radius_90 as fiducial
        gal_radius_90, _gal_mass_90 = ut.particle.get_galaxy_radius_mass(
            part, profile_spec_name, 'mass.percent', mass_percent, star_distance_max)

        rotation_vectors, _eigen_values, axis_ratios = ut.particle.get_principal_axes(
            part, profile_spec_name, gal_radius_90, scalarize=True)

        gal['rotation.tensor'].append(rotation_vectors)
        gal['axis.ratio'].append(axis_ratios)

        for mass_percent in profile_mass_percents:
            mass_percent_name = '{:.0f}'.format(mass_percent)

            gal_radius, gal_mass = ut.particle.get_galaxy_radius_mass(
                part, profile_spec_name, 'mass.percent', mass_percent, star_distance_max)
            gal['radius.3d.' + mass_percent_name].append(gal_radius)
            gal['mass.3d.' + mass_percent_name].append(gal_mass)

            gal_radius_minor, gal_mass_minor = ut.particle.get_galaxy_radius_mass(
                part, profile_spec_name, 'mass.percent', mass_percent, star_distance_max,
                axis_kind='minor', rotation_vectors=rotation_vectors,
                other_axis_distance_limits=[0, gal_radius_90])
            gal['radius.minor.' + mass_percent_name].append(gal_radius_minor)
            gal['mass.minor.' + mass_percent_name].append(gal_mass_minor)

            gal_radius_major, gal_mass_major = ut.particle.get_galaxy_radius_mass(
                part, profile_spec_name, 'mass.percent', mass_percent, star_distance_max,
                axis_kind='major', rotation_vectors=rotation_vectors,
                other_axis_distance_limits=[0, gal_radius_minor])
            gal['radius.major.' + mass_percent_name].append(gal_radius_major)
            gal['mass.major.' + mass_percent_name].append(gal_mass_major)

        pro = plot_property_v_distance(
            part, profile_spec_name, 'mass', 'density', 'log', False, None,
            [0.05, 20], 0.1, None, 'log', 3, get_values=True)
        for k in ['distance', 'density']:
            gal['profile.3d.' + k].append(pro[profile_spec_name][k])

        pro = plot_property_v_distance(
            part, profile_spec_name, 'mass', 'density', 'log', False, None,
            [0.05, 20], 0.1, None, 'log', 2,
            rotation_vectors=rotation_vectors, other_axis_distance_limits=[0, 1], get_values=True)
        for k in ['distance', 'density']:
            gal['profile.major.' + k].append(pro[profile_spec_name][k])

        pro = plot_property_v_distance(
            part, profile_spec_name, 'mass', 'density', 'log', False, None,
            [0.05, 20], 0.1, None, 'log', 1,
            rotation_vectors=rotation_vectors, other_axis_distance_limits=[0, 0.05],
            get_values=True)
        for k in ['distance', 'density']:
            gal['profile.minor.bulge.' + k].append(pro[profile_spec_name][k])

        pro = plot_property_v_distance(
            part, profile_spec_name, 'mass', 'density', 'log', False, None,
            [0.05, 20], 0.1, None, 'log', 1,
            rotation_vectors=rotation_vectors, other_axis_distance_limits=[1, 10], get_values=True)
        for k in ['distance', 'density']:
            gal['profile.minor.disk.' + k].append(pro[profile_spec_name][k])

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
              gal['star.position'][ti][0], gal['star.position'][ti][1],
              gal['star.position'][ti][2]), end='')
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
    Plot different simulations on same figure for comparison.
    '''
    def __init__(self):
        '''
        Set directories and names of simulations to read.
        '''
        self.simulation_names = [
            # original FIRE
            ['/work/02769/arwetzel/fire/m12i_ref12', 'm12i r12 fire1'],

            # symmetric
            ['m12i/m12i_ref12', 'm12i r12'],
            ['fb-sym/m12i_ref13', 'm12i r13'],

            # different halos
            ['m12i/m12i_ref12', 'm12i r12'],
            ['m12b/m12b_ref12', 'm12b r12'],
            ['m12m/m12m_ref12', 'm12m r12'],
            ['m12c/m12c_ref12', 'm12c r12'],
            ['m12f/m12f_ref12', 'm12f r12'],
            ['m12q/m12q_ref12', 'm12q r12'],

        ]

    def read_simulations(
        self, simulation_names=None, redshift=0, species='all',
        property_names=['mass', 'position', 'form.time'], force_float32=True):
        '''
        Read snapshots from simulations.

        Parameters
        ----------
        simulation_names : list : list of simulation directories and name/label for figure.
        redshift : float
        species : string or list : particle species to read
        property_names : string or list : names of properties to read
        force_float32 : boolean : whether to force positions to be 32-bit
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
                part = gizmo_io.Read.read_snapshot(
                    species, 'redshift', redshift, directory, property_names=property_names,
                    simulation_name=simulation_names[directory], force_float32=force_float32)

                if 'velocity' in property_names:
                    gizmo_io.assign_orbit(part, 'gas')

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

    def plot_profiles(
        self, parts=None, distance_bin_width=0.1,
        simulation_names=None, redshifts=[6, 5, 4, 3, 2, 1.5, 1, 0.5, 0],
        species='all', property_names=['mass', 'position', 'form.time'], force_float32=True):
        '''
        Plot profiles of various properties, comparing all simulations at each redshift.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        distance_bin_width : float : width of distance bin
        simulation_names : list : list of simulation directories and name/label for figure.
        redshifts : float or list
        species : string or list : particle species to read
        property_names : string or list : names of properties to read
        force_float32 : boolean : whether to force positions to be 32-bit
        '''
        if isinstance(parts, dict):
            parts = [parts]

        if np.isscalar(redshifts):
            redshifts = [redshifts]

        if parts is not None and len(redshifts) > 1:
            self.say('! input particles at single snapshot but also input more than one redshift')
            return

        for redshift in redshifts:
            if parts is None or len(redshifts) > 1:
                parts = self.read_simulations(
                    simulation_names, redshift, species, property_names, force_float32)

            plot_property_v_distance(
                parts, 'total', 'mass', 'vel.circ', 'linear', False, [0, None],
                [0.1, 300], distance_bin_width, write_plot=True)

            plot_property_v_distance(
                parts, 'total', 'mass', 'sum.cum', 'log', False, [None, None],
                [1, 300], distance_bin_width, write_plot=True)

            plot_property_v_distance(
                parts, 'dark', 'mass', 'sum.cum', 'log', False, [None, None],
                [1, 300], distance_bin_width, write_plot=True)

            plot_property_v_distance(
                parts, 'dark', 'mass', 'density', 'log', False, [None, None],
                [0.1, 30], distance_bin_width, write_plot=True)

            if 'gas' in parts[0]:
                plot_property_v_distance(
                    parts, 'baryon', 'mass', 'sum.cum.fraction', 'linear', False, [0, 2],
                    [10, 2000], distance_bin_width, write_plot=True)

                plot_property_v_distance(
                    parts, 'gas', 'mass', 'sum.cum', 'log', False, [None, None],
                    [1, 300], distance_bin_width, write_plot=True)

            if 'star' in parts[0]:
                plot_property_v_distance(
                    parts, 'star', 'mass', 'sum.cum', 'log', False, [None, None],
                    [1, 300], distance_bin_width, write_plot=True)

                plot_property_v_distance(
                    parts, 'star', 'mass', 'density', 'log', False, [None, None],
                    [0.1, 30], distance_bin_width, write_plot=True)

            if 'velocity' in property_names:
                plot_property_v_distance(
                    parts, 'gas', 'host.velocity.rad', 'average', 'linear', True, [None, None],
                    [1, 300], 0.25, write_plot=True)

            if 'form.time' in property_names and redshift <= 4:
                plot_star_form_history(
                    parts, 'mass', 'redshift', [0, 6], 0.2, 'linear',
                    distance_limits=[0, 15], sfh_limits=[None, None], write_plot=True)

                plot_star_form_history(
                    parts, 'form.rate', 'time.lookback', [0, 13], 0.5, 'linear',
                    distance_limits=[0, 15], sfh_limits=[None, None], write_plot=True)

                plot_star_form_history(
                    parts, 'form.rate.specific', 'time.lookback', [0, 13], 0.5, 'linear',
                    distance_limits=[0, 15], sfh_limits=[None, None], write_plot=True)

            self.plot_images(parts, redshifts=redshift)

    def plot_images(
        self, parts=None,
        distance_max=15, distance_bin_width=0.05, image_limits=[10 ** 7, 10 ** 10],
        align_principal_axes=True, simulation_names=None,
        redshifts=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.45, 0.4, 0.35, 0.3,
                   0.25, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0],
        species=['star', 'gas'], property_names=['mass', 'position'], force_float32=True):
        '''
        Plot images of simulations at each snapshot.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        distance_max : float : maximum distance from center to plot
        distance_bin_width : float : distance bin width (pixel size)
        image_limits : list : min and max limits for image dyanmic range
        align_principal_axes : boolean : whether to align plot axes with principal axes
        simulation_names : list : list of simulation directories and name/label for figure.
        redshifts : float or list
        species : string or list : particle species to read
        property_names : string or list : names of properties to read
        force_float32 : boolean : whether to force positions to be 32-bit
        '''
        if np.isscalar(redshifts):
            redshifts = [redshifts]

        if parts is not None and len(redshifts) > 1:
            self.say('! input particles at single snapshot but also input more than one redshift')
            return

        for redshift in redshifts:
            if parts is None or len(redshifts) > 1:
                parts = self.read_simulations(
                    simulation_names, redshift, species, property_names, force_float32)

            for part in parts:
                for spec_name in ['star', 'gas']:
                    if spec_name in part:
                        plot_image(
                            part, spec_name, [0, 1, 2], [0, 1, 2], distance_max, distance_bin_width,
                            image_limits=image_limits, align_principal_axes=align_principal_axes,
                            write_plot=True, add_simulation_name=True)

CompareSimulations = CompareSimulationsClass()


def compare_resolution(
    parts=None, simulation_names=[],
    redshifts=[0], distance_limits=[0.01, 20], distance_bin_width=0.1):
    '''
    .
    '''
    from . import gizmo_io

    if not simulation_names:
        simulation_names = [
            ['m12i_ref11_dm', 'm12i r11 dm'],
            ['m12i_ref11_dm_res-adapt', 'm12i r11 dm res-adapt'],
            ['m12i_ref12_dm', 'm12i r12 dm'],
            ['m12i_ref12_dm_res-adapt', 'm12i r12 dm res-adapt'],
            ['m12i_ref13_dm', 'm12i r13 dm'],
            ['m12i_ref13_dm_res-adapt', 'm12i r13 dm res-adapt'],
            #['/work/02769/arwetzel/m12/m12i/tests/m12i_ref14_dm_res-low', 'm12i r14 dm'],
        ]

    if np.isscalar(redshifts):
        redshifts = [redshifts]

    if parts is None:
        parts = []
        for simulation_dir, simulation_name in simulation_names:
            for redshift in redshifts:
                assign_center = True
                if 'ref14' in simulation_dir:
                    assign_center = False
                part = gizmo_io.Read.read_snapshot(
                    'dark', 'redshift', redshift, simulation_dir, simulation_name=simulation_name,
                    property_names=['position', 'mass'], assign_center=assign_center,
                    force_float32=True)
                if 'ref14' in simulation_dir:
                    part.center_position = np.array([41820.015, 44151.745, 46272.818],
                                                    dtype=np.float32)
                if len(redshifts) > 1:
                    part.info['simulation.name'] += ' z=%.1f'.format(redshift)

                parts.append(part)

    plot_property_v_distance(
        parts, 'dark', 'mass', 'vel.circ', 'log', False, [None, None],
        distance_limits, distance_bin_width, write_plot=True)

    plot_property_v_distance(
        parts, 'dark', 'mass', 'density', 'log', False, [None, None],
        distance_limits, distance_bin_width, write_plot=True)

    plot_property_v_distance(
        parts, 'dark', 'mass', 'density*r', 'log', False, [None, None],
        distance_limits, distance_bin_width, write_plot=True)

    return parts
