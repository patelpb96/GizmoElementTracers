'''
Analysis of Gizmo simulations.

Masses in {M_sun}, positions in {kpc comoving}, distances and radii in {kpc physical}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf  # @UnusedImport
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
import copy
# local ----
from utilities import utility as ut
from utilities import constants as const
from utilities import plot


#===================================================================================================
# diagnostic
#===================================================================================================
def plot_mass_contamination(
    part, distance_lim=[1, 2000], distance_bin_wid=0.02, distance_bin_num=None,
    distance_scaling='log', halo_radius=None, scale_to_halo_radius=False, center_position=[],
    axis_y_scaling='log', write_plot=False, plot_directory='.'):
    '''
    Plot lower resolution particle contamination v distance from input center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    distance_lim : list : min and max limits for distance from galaxy
    distance_bin_wid : float : width of each distance bin (in units of distance_scaling)
    distance_bin_num : int : number of distance bins
    distance_scaling : string : lin or log
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_position : array : position of galaxy/halo center
    axis_y_scaling : string : scaling of y-axis: lin, log
    write_plot : boolean : whether to write plot to file
    plot_directory : string : directory to put plot
    '''
    species_test = ['dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6', 'gas', 'star']

    species_ref = 'dark'

    Say = ut.io.SayClass(plot_mass_contamination)

    species_test_t = []
    for spec_test in species_test:
        if spec_test in part:
            species_test_t.append(spec_test)
        else:
            Say.say('! no %s in particle dictionary' % spec_test)
    species_test = species_test_t

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    distance_lim_use = np.array(distance_lim)
    if halo_radius and scale_to_halo_radius:
        distance_lim_use *= halo_radius

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, distance_lim_use, distance_bin_wid, distance_bin_num)

    pros = {species_ref: {}}
    for spec in species_test:
        pros[spec] = {}

    ratios = {}

    for spec in pros:
        distances = ut.coord.distance(
            'scalar', part[spec]['position'], center_position, part.info['box.length'])
        distances *= part.snapshot['scale-factor']  # convert to {kpc physical}
        pros[spec] = DistanceBin.get_histogram_profile(distances, part[spec]['mass'])

    for spec in species_test:
        mass_ratio_bin = pros[spec]['hist'] / pros[species_ref]['hist']
        mass_ratio_cum = pros[spec]['hist.cum'] / pros[species_ref]['hist.cum']
        ratios[spec] = {'bin': mass_ratio_bin, 'cum': mass_ratio_cum}
        """
        for dist_bin_i in xrange(DistanceBin.num):
            dist_bin_lim = DistanceBin.get_bin_limit('lin', dist_bin_i)
            Say.say('dist = [%.3f, %.3f]: mass ratio (bin, cum) = (%.5f, %.5f)' %
                    (dist_bin_lim[0], dist_bin_lim[1],
                     mass_ratio_bin[dist_bin_i], mass_ratio_cum[dist_bin_i]))
            if mass_ratio_bin[dist_bin_i] >= 1.0:
                break
        """

    # print diagnostics
    spec = 'dark.2'
    Say.say('%s cumulative mass/number:' % spec)
    distances = pros[spec]['distance.cum']
    print_string = '  d < %.3f kpc: cumulative contamination mass = %.2e, number = %d'
    if scale_to_halo_radius:
        distances /= halo_radius
        print_string = '  d/R_halo < %.3f: mass = %.2e, number = %d'
    for dist_i in xrange(pros[spec]['hist.cum'].size):
        if pros[spec]['hist.cum'][dist_i] > 0:
            Say.say(print_string % (distances[dist_i], pros[spec]['hist.cum'][dist_i],
                                    pros[spec]['hist.cum'][dist_i] / part[spec]['mass'][0]))

    # plot ----------
    colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    #fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    subplot.set_xlim(distance_lim)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim([0.0001, 3])

    subplot.set_ylabel('$M_{\\rm spec} / M_{\\rm %s}$' % species_ref, fontsize=20)
    if scale_to_halo_radius:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, axis_y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    for spec_i, spec in enumerate(species_test):
        plot_func(xs, ratios[spec]['bin'], color=colors[spec_i], alpha=0.6, label=spec)

    legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    legend.get_frame().set_alpha(0.7)

    # plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory, create_path=True)
        dist_name = 'dist'
        if halo_radius and scale_to_halo_radius:
            dist_name += '.200m'
        plot_name = 'mass.ratio_v_%s_z.%.1f.pdf' % (dist_name, part.snapshot['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def plot_metal_v_distance(
    part, species='gas',
    distance_lim=[10, 3000], distance_bin_wid=0.1, distance_bin_num=None, distance_scaling='log',
    halo_radius=None, scale_to_halo_radius=False, center_position=[],
    plot_kind='metallicity', axis_y_scaling='log', write_plot=False, plot_directory='.'):
    '''
    Plot metallicity (in bin or cumulative) of gas or stars v distance from galaxy.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string : particle species
    distance_lim : list : min and max limits for distance from galaxy
    distance_bin_wid : float : width of each distance bin (in units of distance_scaling)
    distance_bin_num : int : number of distance bins
    distance_scaling : string : lin or log
    halo_radius : float : radius of halo {kpc physical}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    center_position : array : position of galaxy center {kpc comoving}
    plot_kind : string : metallicity or metal.mass.cum
    axis_y_scaling : string : scaling of y-axis
    write_plot : boolean : whether to write plot to file
    plot_directory : string : directory to put plot
    '''
    metal_index = 0  # overall metallicity

    Say = ut.io.SayClass(plot_metal_v_distance)

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    distance_lim_use = np.array(distance_lim)
    if halo_radius and scale_to_halo_radius:
        distance_lim_use *= halo_radius

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, distance_lim_use, distance_bin_wid, distance_bin_num)

    distances = ut.coord.distance(
        'scalar', part[species]['position'], center_position, part.info['box.length'])
    distances *= part.snapshot['scale-factor']  # convert to {kpc physical}

    metal_masses = part[species]['metallicity'][:, metal_index] * part[species]['mass']
    metal_masses /= 0.02  # convert to {wrt Solar}

    pro_metal = DistanceBin.get_histogram_profile(distances, metal_masses, get_fraction=True)
    if 'metallicity' in plot_kind:
        pro_mass = DistanceBin.get_histogram_profile(distances, part['gas']['mass'])
        ys = pro_metal['mass'] / pro_mass['mass']
        axis_y_lim = np.clip(plot.get_limits(ys), 0.0001, 10)
    elif 'metal.mass.cum' in plot_kind:
        ys = pro_metal['fraction.cum']
        axis_y_lim = [0.001, 1]

    # plot ----------
    # colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, num=1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03, wspace=0.03)

    subplot.set_xlim(distance_lim)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim(axis_y_lim)

    if 'metallicity' in plot_kind:
        subplot.set_ylabel('$Z \, / \, Z_\odot$', fontsize=20)
    elif 'metal.mass.cum' in plot_kind:
        subplot.set_ylabel('$M_{\\rm Z}(< r) \, / \, M_{\\rm Z,tot}$', fontsize=20)

    if scale_to_halo_radius:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'

    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, axis_y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    plot_func(xs, ys, color='blue', alpha=0.6)

    # legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    # legend.get_frame().set_alpha(0.7)

    plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        dist_name = 'dist'
        if halo_radius and scale_to_halo_radius:
            dist_name += '.200m'
        plot_name = plot_kind + '_v_' + dist_name + '_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


#===================================================================================================
# visualize
#===================================================================================================
def plot_positions(
    part, species='dark', dimen_indices=[0, 1], distance_max=1000, distance_bin_wid=1,
    center_position=[], subsample_factor=None, write_plot=False, plot_directory='.'):
    '''
    Visualize the positions of given partcle species, using either a single panel for 2 axes or
    3 panels for all axes.

    Parameters
    ----------
    part : dict : catalog of particles
    species : string : particle species to plot
    dimen_indices : list : which dimensions to plot
        if 2, plot one v other, if 3, plot all via 3 panels
    distance_max : float : maximum distance from center to plot
    distance_bin_wid : float : length of histogram bin
    center_position : array-like : position of center
    subsample_factor : int : factor by which periodically to sub-sample particles
    write_plot : boolean : whether to write plot to file
    plot_directory : string : where to put plot file
    '''
    dimen_labels = {0: 'x', 1: 'y', 2: 'z'}

    positions = [[] for _ in xrange(part[species]['position'].shape[1])]
    for dimen_i in dimen_indices:
        positions[dimen_i] = np.array(part[species]['position'][:, dimen_i])

    if subsample_factor > 1:
        for dimen_i in dimen_indices:
            positions[dimen_i] = positions[dimen_i][::subsample_factor]

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    if center_position is not None and len(center_position):
        masks = positions[dimen_indices[0]] < Inf
        for dimen_i in dimen_indices:
            positions[dimen_i] -= center_position[dimen_i]
            positions[dimen_i] *= part.snapshot['scale-factor']
            masks *= (positions[dimen_i] <= distance_max) * (positions[dimen_i] >= -distance_max)

        for dimen_i in dimen_indices:
            positions[dimen_i] = positions[dimen_i][masks]
        masses = part[species]['mass'][masks]
    else:
        position_dif_max = 0
        for dimen_i in dimen_indices:
            position_dif = np.max(positions[dimen_i]) - np.min(positions[dimen_i])
            if position_dif > position_dif_max:
                position_dif_max = position_dif
        distance_max = 0.5 * position_dif_max
        masses = part[species]['mass']

    position_bin_num = int(np.round(2 * distance_max / distance_bin_wid))
    position_lims = np.array([[-distance_max, distance_max], [-distance_max, distance_max]])

    # plot ----------
    plt.clf()

    if len(dimen_indices) == 2:
        plt.minorticks_on()
        fig = plt.figure(1)
        subplot = fig.add_subplot(111)
        fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

        subplot.set_xlim(position_lims[0])
        subplot.set_ylim(position_lims[1])

        subplot.set_xlabel('position %s $[\\rm kpc\,physical]$' % dimen_labels[dimen_indices[0]])
        subplot.set_ylabel('position %s $[\\rm kpc\,physical]$' % dimen_labels[dimen_indices[1]])

        H = subplot.hist2d(positions[dimen_indices[0]], positions[dimen_indices[1]], weights=masses,
                           range=position_lims, bins=position_bin_num, norm=LogNorm(),
                           cmap=plt.cm.Greens)  # @UndefinedVariable

        fig.colorbar(H[3])

    elif len(dimen_indices) == 3:
        #position_lims *= 0.99999  # ensure that tick labels do not overlap
        position_lims[0, 0] *= 0.9999
        position_lims[1, 0] *= 0.9999

        fig, subplots = plt.subplots(2, 2, num=1, sharex=True, sharey=True)
        fig.subplots_adjust(left=0.17, right=0.96, top=0.97, bottom=0.13, hspace=0.03, wspace=0.03)

        plot_dimen_iss = [
            [dimen_indices[0], dimen_indices[1]],
            [dimen_indices[0], dimen_indices[2]],
            [dimen_indices[1], dimen_indices[2]],
        ]

        subplot_iss = [
            [0, 0],
            [1, 0],
            [1, 1],
        ]

        for plot_i, plot_dimen_is in enumerate(plot_dimen_iss):
            subplot_is = subplot_iss[plot_i]
            subplot = subplots[subplot_is[0], subplot_is[1]]
            subplot.set_ylim([position_lims[1, 0] * 0.9, position_lims[1, 1]])

            if subplot_is == [0, 0]:
                subplot.set_ylabel('%s $[\\rm kpc\,phys]$' % dimen_labels[plot_dimen_is[1]])
            elif subplot_is == [1, 0]:
                subplot.set_xlabel('%s $[\\rm kpc\,phys]$' % dimen_labels[plot_dimen_is[0]])
                subplot.set_ylabel('%s $[\\rm kpc\,phys]$' % dimen_labels[plot_dimen_is[1]])
            elif subplot_is == [1, 1]:
                subplot.set_xlabel('%s $[\\rm kpc\,phys]$' % dimen_labels[plot_dimen_is[0]])

            subplot.set_xlim(position_lims[0])
            subplot.set_ylim(position_lims[1])

            H = subplot.hist2d(
                positions[plot_dimen_is[0]], positions[plot_dimen_is[1]], weights=masses,
                range=position_lims, bins=position_bin_num, norm=LogNorm(),
                cmap=plt.cm.Greens)  # @UndefinedVariable
            #fig.colorbar(H[3])  #, ax=subplot)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = 'test'
        plt.savefig(plot_directory + plot_name, format='pdf')
    else:
        plt.show(block=False)


#===================================================================================================
# analysis utility
#===================================================================================================
def parse_center_positions(parts, center_positions):
    '''
    Parameters
    ----------
    parts : dict or list : catalog[s] of particles
    center_positions : list or list of lists : position[s] of center[s]
    '''
    if isinstance(parts, dict):
        parts = [parts]

    if center_positions is None or np.ndim(center_positions) == 1:
        if not len(center_positions):
            center_positions = []
            for part in parts:
                if len(part.center_position):
                    center_positions.append(part.center_position)
                else:
                    center_positions.append([])
        else:
            center_positions = [center_positions]
    else:
        center_positions = center_positions

    return center_positions


def get_species_histogram_profiles(
    part, species=['all'], prop_name='mass', center_position=[], DistanceBin=None):
    '''
    Get dictionary of profiles of mass/density (or any summed quantity) for each particle species.

    Parameters
    ----------
    part : dict : catalog of particles
    species : string or list : species to compute total mass of
    center_position : list : center position
    DistanceBin : class : distance bin class
    '''
    pros = {}

    Fraction = ut.math.FractionClass()

    # ensure is list even if just one species
    if np.isscalar(species):
        species = [species]
    else:
        species = copy.copy(species)

    if species == ['all'] or species == ['total'] or species == ['baryon']:
        species = ['dark', 'gas', 'star']
        #species = part.keys()
        if 'dark.2' in part:
            species.append('dark.2')

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    for spec in species:
        if 'mass' in prop_name:
            positions, prop_vals = ut.particle.get_species_positions_masses(part, spec)

            if np.isscalar(prop_vals):
                prop_vals = np.zeros(positions.shape[0], dtype=prop_vals.dtype) + prop_vals

        else:
            positions = part[spec]['position']
            prop_vals = part[spec][prop_name]

        distances = ut.coord.distance('scalar', positions, center_position, part.info['box.length'])
        distances *= part.snapshot['scale-factor']  # convert to {kpc physical}

        pros[spec] = DistanceBin.get_histogram_profile(distances, prop_vals)

    props = [pro_prop for pro_prop in pros[species[0]] if 'distance' not in pro_prop]
    props_dist = [pro_prop for pro_prop in pros[species[0]] if 'distance' in pro_prop]

    if prop_name == 'mass':
        # create dictionary for baryonic mass
        if 'star' in species or 'gas' in species:
            spec_new = 'baryon'
            pros[spec_new] = {}
            for spec in np.intersect1d(species, ['star', 'gas']):
                for pro_prop in props:
                    if pro_prop not in pros[spec_new]:
                        pros[spec_new][pro_prop] = np.array(pros[spec][pro_prop])
                    elif 'log' in pro_prop:
                        pros[spec_new][pro_prop] = ut.math.get_log(
                            10 ** pros[spec_new][pro_prop] + 10 ** pros[spec][pro_prop])
                    else:
                        pros[spec_new][pro_prop] += pros[spec][pro_prop]

            for pro_prop in props_dist:
                pros[spec_new][pro_prop] = pros[species[0]][pro_prop]
            species.append(spec_new)

        if len(species) > 1:
            # create dictionary for total mass
            spec_new = 'total'
            pros[spec_new] = {}
            for spec in np.setdiff1d(species, ['baryon', 'total']):
                for pro_prop in props:
                    if pro_prop not in pros[spec_new]:
                        pros[spec_new][pro_prop] = np.array(pros[spec][pro_prop])
                    elif 'log' in pro_prop:
                        pros[spec_new][pro_prop] = ut.math.get_log(
                            10 ** pros[spec_new][pro_prop] + 10 ** pros[spec][pro_prop])
                    else:
                        pros[spec_new][pro_prop] += pros[spec][pro_prop]

            for pro_prop in props_dist:
                pros[spec_new][pro_prop] = pros[species[0]][pro_prop]
            species.append(spec_new)

            # create mass fraction wrt total mass
            for spec in np.setdiff1d(species, ['total']):
                for pro_prop in ['hist', 'hist.cum']:
                    pros[spec][pro_prop + '.fraction'] = Fraction.get_fraction(
                        pros[spec][pro_prop], pros['total'][pro_prop])

                    if spec == 'baryon':
                        # units of cosmic baryon fraction
                        pros[spec][pro_prop + '.fraction'] /= (part.Cosmo['omega_baryon'] /
                                                               part.Cosmo['omega_matter'])

        # create circular velocity = sqrt (G m(<r) / r)
        for spec in species:
            pros[spec]['vel.circ'] = (pros[spec]['hist.cum'] / pros[spec]['distance.cum'] *
                                      const.grav_kpc_msun_yr)
            pros[spec]['vel.circ'] = np.sqrt(pros[spec]['vel.circ'])
            pros[spec]['vel.circ'] *= const.km_per_kpc * const.yr_per_sec

    return pros


def get_species_statistics_profiles(
    part, species=['all'], prop_name='', center_position=[], DistanceBin=None,
    weight_by_mass=False):
    '''
    Get dictionary of profiles of statistics (such as median, average) for given property for each
    particle species.

    Parameters
    ----------
    part : dict : catalog of particles
    species : string or list : species to compute total mass of
    prop_name : string : name of property to get statistics of
    center_position : list : center position
    DistanceBin : class : distance bin class
    weight_by_mass : boolean : whether to weight property by species mass
    '''
    pros = {}

    # ensure is list even if just one species
    if np.isscalar(species):
        species = [species]

    if species == ['all'] or species == ['total']:
        species = ['dark', 'gas', 'star', 'dark.2']
        #species = part.keys()
    elif species == ['baryon']:
        species = ['gas', 'star']

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    for spec in species:
        # {kpc comoving}
        distances = ut.coord.distance(
            'scalar', part[spec]['position'], center_position, part.info['box.length'])
        distances *= part.snapshot['scale-factor']  # convert to {kpc physical}

        if weight_by_mass:
            masses = part[spec]['mass']
        else:
            masses = None
        pros[spec] = DistanceBin.get_statistics_profile(distances, part[spec][prop_name], masses)

    return pros


#===================================================================================================
# general property analysis
#===================================================================================================
def plot_property_distribution(
    parts, species='gas', prop_name='density', prop_scaling='log', prop_lim=[], prop_bin_wid=None,
    prop_bin_num=100, prop_statistic='probability', center_positions=[], distance_lim=[],
    axis_y_scaling='log', axis_y_lim=[], write_plot=False, plot_directory='.'):
    '''
    Plot distribution of property.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string : particle species
    prop_name : string : property name
    prop_scaling : string : ling or log
    prop_lim : list : min and max limits of property
    prop_bin_wid : float : width of property bin (use this or prop_bin_num)
    prop_bin_num : int : number of property bins within limits (use this or prop_bin_wid)
    prop_statistic : string : statistic to plot: probability,
    center_positions : array or list of arrays : position[s] of galaxy center[s]
    distance_lim : list : min and max limits for distance from galaxy
    axis_y_scaling : string : lin or log
    axis_y_lim : list : min and max limits for y-axis
    write_plot : boolean : whether to write plot to file
    plot_directory : string : directory to put plot
    '''
    Say = ut.io.SayClass(plot_property_distribution)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = parse_center_positions(parts, center_positions)

    prop_name_dict = prop_name
    if prop_name == 'density.num':
        prop_name_dict = 'density'

    Stat = ut.math.StatisticClass()

    for part_i, part in enumerate(parts):
        prop_vals = part[species][prop_name_dict]

        if distance_lim:
            distances = ut.coord.distance(
                'scalar', part[species]['position'], center_positions[part_i],
                part.info['box.length'])
            distances *= part.snapshot['scale-factor']  # {kpc physical}
            prop_is = ut.array.elements(distances, distance_lim)
            prop_vals = prop_vals[prop_is]

        Say.say('keeping %s %s particles' % (prop_vals.size, species))

        if prop_name == 'density.num':
            # convert to {cm ^ -3 physical}
            prop_vals *= const.proton_per_sun * const.kpc_per_cm ** 3

        #Say.say('%s %s range = %s' %
        #        (prop_name, prop_scaling, ut.array.get_limits(prop_vals, digit_num=2)))

        Stat.append_to_dictionary(prop_vals, prop_lim, prop_bin_wid, prop_bin_num, prop_scaling)

        Stat.print_statistics(-1)

    colors = plot.get_colors(len(parts))

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlim(prop_lim)
    if not axis_y_lim:
        y_vals = [Stat.distr[prop_statistic][part_i] for part_i in xrange(len(parts))]
        axis_y_lim = plot.get_limits(y_vals, axis_y_scaling, exclude_zero=True)
    subplot.set_ylim(axis_y_lim)

    subplot.set_xlabel(plot.get_label(prop_name, species=species, get_units=True))
    subplot.set_ylabel(plot.get_label(prop_name, prop_statistic, species, get_symbol=True,
                                      get_units=False, draw_log=prop_scaling))

    #import ipdb; ipdb.set_trace()

    plot_func = plot.get_plot_function(subplot, prop_scaling, axis_y_scaling)
    for part_i, part in enumerate(parts):
        plot_func(Stat.distr['bin.mid'][part_i], Stat.distr[prop_statistic][part_i],
                  color=colors[part_i], alpha=0.5, linewidth=2, label=part.info['simulation.name'])

    # redshift legend
    legend_z = subplot.legend([plt.Line2D((0, 0), (0, 0), linestyle='.')],
                              ['$z=%.1f$' % parts[0].snapshot['redshift']],
                              loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    if len(parts) > 1:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = species + '.' + prop_name + '_distr_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def plot_property_v_property(
    part, species='gas',
    prop_x_name='density', prop_x_scaling='log', prop_x_lim=[],
    prop_y_name='temperature', prop_y_scaling='log', prop_y_lim=[],
    prop_bin_num=200, center_position=[], distance_lim=[20, 300],
    write_plot=False, plot_directory='.'):
    '''
    Plot property v property.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string : particle species
    prop_x_name : string : property name for x-axis
    prop_x_scaling : string : lin or log
    prop_y_name : string : property name for y-axis
    prop_y_scaling : string : lin or log
    center_position : array : position of galaxy center
    distance_lim : list : min and max limits for distance from galaxy
    write_plot : boolean : whether to write plot to file
    plot_directory : string : directory to put plot
    '''
    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    if len(center_position) and len(distance_lim):
        distances = ut.coord.distance(
            'scalar', part[species]['position'], center_position, part.info['box.length'])
        distances *= part.snapshot['scale-factor']
        masks = ut.array.elements(distances, distance_lim)
    else:
        masks = np.arange(part[species][prop_x_name].size, dtype=np.int32)

    prop_x_vals = part[species][prop_x_name][masks]
    prop_y_vals = part[species][prop_y_name][masks]

    if prop_x_lim:
        indices = ut.array.elements(prop_x_vals, prop_x_lim)
        prop_x_vals = prop_x_vals[indices]
        prop_y_vals = prop_y_vals[indices]

    if prop_y_lim:
        indices = ut.array.elements(prop_y_vals, prop_y_lim)
        prop_x_vals = prop_x_vals[indices]
        prop_y_vals = prop_y_vals[indices]

    if 'log' in prop_x_scaling:
        prop_x_vals = ut.math.get_log(prop_x_vals)

    if 'log' in prop_y_scaling:
        prop_y_vals = ut.math.get_log(prop_y_vals)

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)

    subplot.set_xlabel(plot.get_label(prop_x_name, species=species, get_units=True, get_symbol=True,
                                      draw_log=prop_x_scaling))

    subplot.set_ylabel(plot.get_label(prop_y_name, species=species, get_units=True, get_symbol=True,
                                      draw_log=prop_y_scaling))

    plt.hist2d(prop_x_vals, prop_y_vals, bins=prop_bin_num, norm=LogNorm(),
               cmap=plt.cm.Greens)  # @UndefinedVariable
    plt.colorbar()

    # plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = (species + '.' + prop_y_name + '_v_' + prop_x_name + '_z.%.1f.pdf' %
                     part.info['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
    else:
        plt.show(block=False)


def plot_property_v_distance(
    parts, species='dark', prop_name='mass', prop_statistic='hist', prop_scaling='log',
    weight_by_mass=False,
    distance_scaling='log', distance_lim=[0.1, 300], distance_bin_wid=0.02, distance_bin_num=None,
    center_positions=[], axis_y_lim=[], write_plot=False, plot_directory='.'):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshot)
    species : string or list : species to compute total mass of
        options: dark, star, gas, baryon, total
    prop_name : string : property to get profile of
    prop_statistic : string : statistic/type to plot
        options: hist, hist.cum, density, density.cum, vel.circ, hist.fraction, hist.cum.fraction,
            med, ave
    prop_scaling : string : scaling for property (y-axis): lin, log
    weight_by_mass : boolean : whether to weight property by species mass
    distance_scaling : string : lin or log
    distance_lim : list : min and max distance for binning
    distance_bin_wid : float : width of distance bin
    distance_bin_num : int : number of bins between limits
    center_positions : list : center position for each particle catalog
    axis_y_lim : list : limits to impose on y-axis
    write_plot : boolean : whether to write plot to file
    plot_directory : string
    '''
    Say = ut.io.SayClass(plot_property_v_distance)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = parse_center_positions(parts, center_positions)

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, distance_lim, width=distance_bin_wid, number=distance_bin_num,
        dimension_num=3)

    pros = []
    for part_i, part in enumerate(parts):
        if prop_name in ['mass', 'sfr']:
            pros_part = get_species_histogram_profiles(
                part, species, prop_name, center_positions[part_i], DistanceBin)
        elif 'gas' in species and 'consume.time' in prop_name:
            pros_part_mass = get_species_histogram_profiles(
                part, species, 'mass', center_positions[part_i], DistanceBin)
            pros_part_sfr = get_species_histogram_profiles(
                part, species, 'sfr', center_positions[part_i], DistanceBin)

            pros_part = pros_part_sfr
            for k in pros_part_sfr['gas']:
                if 'distance' not in k:
                    pros_part['gas'][k] = pros_part_mass['gas'][k] / pros_part_sfr['gas'][k] / 1e9
        else:
            pros_part = get_species_statistics_profiles(
                part, species, prop_name, center_positions[part_i], DistanceBin, weight_by_mass)

        pros.append(pros_part)

        #if part_i > 0:
        #    print(pros[part_i][prop_name] / pros[0][prop_name])

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlim(distance_lim)
    if not axis_y_lim:
        y_vals = [pro[species][prop_statistic] for pro in pros]
        axis_y_lim = plot.get_limits(y_vals, prop_scaling)
        if prop_name == 'consume.time':
            axis_y_lim = plot.get_limits(
                pros[0][species][prop_statistic][pros[0][species][prop_statistic] < 10],
                prop_scaling)
    subplot.set_ylim(axis_y_lim)

    subplot.set_xlabel('radius $r$ $[\\rm kpc\,physical]$')
    label_y = plot.get_label(prop_name, prop_statistic, species, get_symbol=True, get_units=True)
    subplot.set_ylabel(label_y)

    plot_func = plot.get_plot_function(subplot, distance_scaling, prop_scaling)
    colors = plot.get_colors(len(parts))

    if 'fraction' in prop_statistic:
        plot_func(distance_lim, [1, 1], color='black', linestyle=':', alpha=0.5, linewidth=2)

    for part_i, pro in enumerate(pros):
        plot_func(pro[species]['distance'], pro[species][prop_statistic], color=colors[part_i],
                  linestyle='-', alpha=0.5, linewidth=2,
                  label=parts[part_i].info['simulation.name'])

    # redshift legend
    legend_z = subplot.legend([plt.Line2D((0, 0), (0, 0), linestyle='.')],
                              ['$z=%.1f$' % parts[0].snapshot['redshift']],
                              loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    if len(parts) > 1:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = (species + '.' + prop_name + '.' + prop_statistic +
                     '_v_dist_z.%.1f.pdf' % part.info['redshift'])
        plot_name = plot_name.replace('.hist', '')
        plot_name = plot_name.replace('mass.vel.circ', 'vel.circ')
        plot_name = plot_name.replace('mass.density', 'density')
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


#===================================================================================================
# star formation analysis
#===================================================================================================
def get_star_form_history(
    part, part_is=None, time_kind='time', time_lim=[0, 3], time_wid=0.01):
    '''
    Get array of times and star-formation rate at each time.

    Parameters
    ----------
    part : dict : dictionary of particles
    part_is : array : star particle indices
    time_kind : string : time kind to use: time, time.lookback, redshift
    time_lim : list : min and max limits of time_kind to get
    time_wid : float : width of time_kind bin

    Returns
    -------
    time_mids : array : times {Gyr}
    dm_dts : array : total star-formation rate at each time {M_sun / yr}
    '''
    species = 'star'

    if part_is is None:
        part_is = np.arange(part[species]['mass'].size, dtype=np.int32)

    part_is_sort = part_is[np.argsort(part[species]['form.time'][part_is])]
    star_form_times = part[species]['form.time'][part_is_sort]
    star_masses = part[species]['mass'][part_is_sort]
    star_masses_cum = np.cumsum(star_masses)

    time_bins = np.arange(min(time_lim), max(time_lim), time_wid)
    if time_kind == 'redshift':
        # input redshift limits and bins, need to convert to time
        redshift_bins = time_bins
        time_bins = np.sort(part.Cosmo.time_from_redshift(time_bins))

    star_mass_cum_bins = np.interp(time_bins, star_form_times, star_masses_cum)
    # convert to {M_sun / yr} and crudely account for stellar mass loss
    dm_dts = np.diff(star_mass_cum_bins) / (np.diff(time_bins) * 1e9) / 0.7

    if 'time' in time_kind:
        # convert to midpoints of bins
        time_bins = time_bins[: time_bins.size - 1] + np.diff(time_bins)
        if 'lookback' in time_kind:
            time_bins = part.Cosmo.time_from_redshift(0) - time_bins  # convert to lookback time
    elif time_kind == 'redshift':
        # convert to midpoints of bins
        time_bins = redshift_bins[: redshift_bins.size - 1] + np.diff(redshift_bins)
        time_bins = np.sort(time_bins)[::-1]

    return time_bins, dm_dts


def plot_star_form_history(
    parts, time_kind='redshift', time_lim=[0, 1], time_wid=0.01,
    distance_lim=[0, 10], center_positions=[],
    write_plot=False, plot_directory='.'):
    '''
    Plot star-formation rate history v time_kind.

    Parameters
    ----------
    parts : dict or list : catalog[s] of particles
    time_kind : string : time kind to use: time, time.lookback, redshift
    time_lim : list : min and max limits of time_kind to get
    time_wid : float : width of time_kind bin
    distance_lim : list : min and max limits of distance to select star particles
    center_positions : list or list of lists : position[s] of galaxy centers {kpc comoving}
    write_plot : boolean : whether to write plot
    plot_directory : string
    '''
    Say = ut.io.SayClass(plot_star_form_history)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = parse_center_positions(parts, center_positions)

    sfrs = []
    times = []
    for part_i, part in enumerate(parts):
        if len(center_positions[part_i]) and len(distance_lim):
            distances = ut.coord.distance(
                'scalar', part['star']['position'], center_positions[part_i],
                part.info['box.length'])
            distances *= part.snapshot['scale-factor']  # {kpc physical}
            part_is = ut.array.elements(distances, distance_lim)
        else:
            part_is = np.arange(part['star']['form.time'].size, dtype=np.int32)

        times_t, sfrs_t = get_star_form_history(part, part_is, time_kind, time_lim, time_wid)

        times.append(times_t)
        sfrs.append(sfrs_t)

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlim(time_lim)
    subplot.set_ylim(plot.get_limits(sfrs, 'log'))

    if time_kind == 'time':
        subplot.set_xlabel('time $[{\\rm Gyr}]$')
    elif time_kind == 'redshift':
        subplot.set_xlabel('redshift')
    subplot.set_ylabel(plot.get_label('sfr', get_symbol=True, get_units=True))

    colors = plot.get_colors(len(parts))

    for part_i, part in enumerate(parts):
        subplot.semilogy(times[part_i], sfrs[part_i], linewidth=2.0, color=colors[part_i],
                         alpha=0.5, label=part.info['simulation.name'])

    # redshift legend
    legend_z = subplot.legend([plt.Line2D((0, 0), (0, 0), linestyle='.')],
                              ['$z=%.1f$' % parts[0].snapshot['redshift']],
                              loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    if len(parts) > 1:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = 'star.form_v_%s_z.%.1f.pdf' % (time_kind, part.info['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


#===================================================================================================
# galaxy disk mass and radius over time, with james and shea
#===================================================================================================
def get_galaxy_mass_v_redshift(
    directory='.',
    redshifts=[3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0]):
    '''
    .
    '''
    from . import gizmo_io

    property_names = ['mass', 'position', 'velocity']

    species = ['star', 'gas', 'dark']
    mass_percents = [50, 90]

    gal = {
        'redshift': [],
        'scale-factor': [],
        'time': [],
        'position': [],
        'star.velocity': [],
        'dark.velocity': [],
    }
    for mass_percent in mass_percents:
        gal['radius.%.0f' % mass_percent] = []
        for spec in species:
            gal[spec + '.mass.%.0f' % mass_percent] = []

    for redshift in redshifts:
        part = gizmo_io.Gizmo.read_snapshot(
            species, 'redshift', redshift, directory, property_names, metal_index_max=1,
            force_float32=True)
        part.center_position = ut.particle.get_center_position(part, 'star')

        gal_radius = ut.particle.get_galaxy_radius(part, [], 50, 30)
        hal_radius = ut.particle.get_halo_radius(part, species, virial_kind='200m')

        for k in ['redshift', 'scale-factor', 'time']:
            gal[k].append(part.snapshot[k])

        gal['position'].append(part.center_position)
        gal['star.velocity'].append(
            ut.particle.get_center_velocity(part, 'star', radius_max=gal_radius))
        gal['dark.velocity'].append(
            ut.particle.get_center_velocity(part, 'dark', radius_max=hal_radius / 2))

        for mass_percent in mass_percents:
            gal_radius = ut.particle.get_galaxy_radius(part, [], mass_percent, 30)
            gal['radius.%.0f' % mass_percent].append(gal_radius)

            for spec in species:
                distances = ut.coord.distance(
                    'scalar', part[spec]['position'], part.center_position, part.info['box.length'])
                distances *= part.snapshot['scale-factor']  # convert to {kpc physical}

                gal[spec + '.mass.%.0f' % mass_percent].append(
                    np.sum(part[spec]['mass'][distances < gal_radius]))

    for prop in gal:
        gal[prop] = np.array(gal[prop])

    return gal


def print_galaxy_mass_v_redshift(gal):
    '''
    Parameters
    ----------
    gal : dict : dictionary of galaxy properties across snapshots
    '''
    print('# redshift scale_factor time[Gyr] ', end='')
    print('star_position(x,y,z)[kpc comov] ', end='')
    print('star_velocity(x,y,z)[km/s phys] dark_velocity(x,y,z)[km/s phys] ', end='')
    print('r_50[kpc phys] star_mass_50[M_sun] gas_mass_50[M_sun] dark_mass_50[M_sun] ', end='')
    print('r_90[kpc phys] star_mass_90[M_sun] gas_mass_90[M_sun] dark_mass_90[M_sun]', end='')
    print()
    for i in xrange(gal['redshift'].size):
        print('%.5f %.5f %.5f ' %
              (gal['redshift'][i], gal['scale-factor'][i], gal['time'][i]), end='')
        print('%.3f %.3f %.3f ' %
              (gal['position'][i][0], gal['position'][i][1], gal['position'][i][2]), end='')
        print('%.3f %.3f %.3f ' %
              (gal['star.velocity'][i][0], gal['star.velocity'][i][1], gal['star.velocity'][i][2]),
              end='')
        print('%.3f %.3f %.3f ' %
              (gal['dark.velocity'][i][0], gal['dark.velocity'][i][1], gal['dark.velocity'][i][2]),
              end='')
        print('%.3e %.3e %.3e %.3e ' %
              (gal['radius.50'][i], gal['star.mass.50'][i], gal['gas.mass.50'][i],
               gal['dark.mass.50'][i]), end='')
        print('%.3e %.3e %.3e %.3e' %
              (gal['radius.90'][i], gal['star.mass.90'][i], gal['gas.mass.90'][i],
               gal['dark.mass.90'][i]), end='')
        print()
