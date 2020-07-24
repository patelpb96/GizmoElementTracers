'''
Plotting analysis of particle data from Gizmo simulations.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

import collections
import numpy as np
from numpy import Inf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors

import utilities as ut
from . import gizmo_default
from . import gizmo_io


# --------------------------------------------------------------------------------------------------
# diagnostic
# --------------------------------------------------------------------------------------------------
def print_properties_statistics(part, species='all'):
    '''
    For each property of each species in particle catalog, print its range and median.

    Parameters
    ----------
    part : dict : catalog of particles (use this instead of reading in)
    species : str or list : name[s] of particle species to print
    '''
    Say = ut.io.SayClass(print_properties_statistics)

    species = ut.array.arrayize(species)
    if 'all' in species:
        species = ['dark2', 'dark', 'star', 'gas']

    species_print = [s for s in species if s in list(part)]

    species_property_dict = collections.OrderedDict()
    species_property_dict['dark2'] = ['id', 'position', 'velocity', 'mass']
    species_property_dict['dark'] = ['id', 'position', 'velocity', 'mass']
    species_property_dict['star'] = [
        'id',
        'id.child',
        'id.generation',
        'position',
        'velocity',
        'mass',
        'form.scalefactor',
        'massfraction.hydrogen',
        'massfraction.helium',
        'massfraction.metals',
    ]
    species_property_dict['gas'] = [
        'id',
        'id.child',
        'id.generation',
        'position',
        'velocity',
        'mass',
        'number.density',
        'size',
        'temperature',
        'hydrogen.neutral.fraction',
        'sfr',
        'massfraction.hydrogen',
        'massfraction.helium',
        'massfraction.metals',
    ]

    # Statistic = ut.statistic.StatisticClass()

    Say.say('printing minimum, median, maximum')
    for spec_name in species_print:
        Say.say(f'\n* {spec_name}')
        for prop_name in species_property_dict[spec_name]:
            try:
                prop_values = part[spec_name].prop(prop_name)
            except KeyError:
                Say.say(f'{prop_name} not in catalog')
                continue

            # Statistic.stat = Statistic.get_statistic_dict(prop_values)
            # Statistic.print_statistics()

            if 'int' in str(prop_values.dtype):
                number_format = '{:.0f}'
            elif np.abs(prop_values).max() < 1e5:
                number_format = '{:.4f}'
            else:
                number_format = '{:.1e}'

            print_string = '{}:  {},  {},  {}'.format(
                prop_name, number_format, number_format, number_format
            )

            Say.say(
                print_string.format(prop_values.min(), np.median(prop_values), prop_values.max())
            )


def plot_metal_v_distance(
    parts,
    species_name='gas',
    metal_name='massfraction.metals',
    axis_y_scaling='log',
    axis_y_limits=[None, None],
    distance_limits=[10, 3000],
    distance_bin_width=0.1,
    distance_scaling='log',
    halo_radius=None,
    scale_to_halo_radius=False,
    center_positions=None,
    host_index=0,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot metallicity (in distance bin or cumulative) of gas or stars v distance from galaxy.

    Parameters
    ----------
    part : dict or list : catalog[s] of particles at snapshot
    species_name : str : name of particle species
    metal_name : str : 'massfraction.X' or 'mass.X'
    axis_y_scaling : str : scaling of y-axis: 'log', 'linear'
    distance_limits : list : min and max limits for distance from galaxy
    distance_bin_width : float : width of each distance bin (in units of distance_scaling)
    distance_scaling : str : scaling of distance: 'log', 'linear'
    halo_radius : float : radius of halo [kpc physical]
    scale_to_halo_radius : bool : whether to scale distance to halo_radius
    center_positions : array : position[s] of galaxy center[s] [kpc comoving]
    host_index : int : index of host halo to get position of (if not input center_positions)
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    virial_kind = '200m'

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)

    distance_limits_use = np.array(distance_limits)
    if halo_radius and scale_to_halo_radius:
        distance_limits_use *= halo_radius

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits_use, distance_bin_width
    )

    metal_values = []
    for part_i, part in enumerate(parts):
        distances = ut.coordinate.get_distances(
            part[species_name]['position'],
            center_positions[part_i],
            part.info['box.length'],
            part.snapshot['scalefactor'],
            total_distance=True,
        )  # [kpc physical]

        metal_mass_kind = metal_name.replace('massfraction.', 'mass.')
        metal_masses = part[species_name].prop(metal_mass_kind)

        pro_metal = DistanceBin.get_sum_profile(distances, metal_masses, get_fraction=True)

        if 'massfraction' in metal_name:
            pro_mass = DistanceBin.get_sum_profile(distances, part[species_name]['mass'])
            if '.cum' in metal_name:
                metal_values.append(pro_metal['sum.cum'] / pro_mass['sum.cum'])
            else:
                metal_values.append(pro_metal['sum'] / pro_mass['sum'])
        elif 'mass' in metal_name:
            if '.cum' in metal_name:
                metal_values.append(pro_metal['sum.cum'])
            else:
                metal_values.append(pro_metal['sum'])

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot,
        distance_scaling,
        distance_limits,
        None,
        axis_y_scaling,
        axis_y_limits,
        metal_values,
    )

    metal_mass_label = f'M_{{\\rm Z,{species_name}}}'
    radius_label = '(r)'
    if '.cum' in metal_name:
        radius_label = '(< r)'
    if 'massfraction' in metal_name:
        axis_y_label = '${}{} \, / \, M_{{\\rm {}}}{}$'.format(
            metal_mass_label, radius_label, species_name, radius_label
        )
    elif 'mass' in metal_name:
        # axis_y_label = '${}(< r) \, / \, M_{{\\rm Z,tot}}$'.format(metal_mass_label)
        axis_y_label = f'${metal_mass_label}{radius_label} \, [M_\odot]$'
    # axis_y_label = '$Z \, / \, Z_\odot$'
    subplot.set_ylabel(axis_y_label)

    if scale_to_halo_radius:
        axis_x_label = f'$d \, / \, R_{{\\rm {virial_kind}}}$'
    else:
        axis_x_label = 'distance $[\\mathrm{kpc}]$'
    subplot.set_xlabel(axis_x_label)

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
            xs,
            metal_values[part_i],
            color=colors[part_i],
            alpha=0.8,
            label=part.info['simulation.name'],
        )

    ut.plot.make_legends(subplot, 'best')

    if plot_file_name is True or plot_file_name == '':
        distance_name = 'dist'
        if halo_radius and scale_to_halo_radius:
            distance_name += '.' + virial_kind
        plot_file_name = ut.plot.get_file_name(
            'mass.ratio', distance_name, species_name, snapshot_dict=part.snapshot
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


def test_metal_variation(parts, species=['star', 'gas'], element_reference='o', log_ratio=True):
    '''
    .
    '''
    Say = ut.io.SayClass(test_metal_variation)

    element_names = ['he', 'c', 'n', 'o', 'ne', 'mg', 'si', 's', 'ca', 'fe']
    element_names = [
        element_name for element_name in element_names if element_name != element_reference
    ]
    if np.isscalar(species):
        species = [species]
    if species[0] in parts:
        parts = [parts]

    Statistic = ut.statistic.StatisticClass()

    for element_name in element_names:
        Say.say(f'\ndex scatter in [{element_name.capitalize()}/{element_reference.capitalize()}]')
        for spec_name in species:
            for part_i, part in enumerate(parts):
                element_ratios_p = part[spec_name].prop(f'massfraction.{element_name}') / part[
                    spec_name
                ].prop(f'massfraction.{element_reference}')
                if part_i == 0:
                    element_ratios = element_ratios_p
                else:
                    element_ratios = np.concatenate((element_ratios, element_ratios_p))

            element_ratios /= np.median(element_ratios)
            if log_ratio:
                element_ratios = np.log10(element_ratios)

            stat = Statistic.get_statistic_dict(element_ratios)
            Say.say(
                '* {:4}: 2-sigma {:.2f}, 3-sigma {:.2f}'.format(
                    spec_name,
                    0.5 * (stat['percent.98'] - stat['percent.2']),
                    0.5 * (stat['percent.99.9'] - stat['percent.0.1']),
                )
            )
            # Say.say(f'* {spec_name}')
            # Say.say(
            #    '  95%: {:.2f} {:.2f} | {:.2f}'.format(
            #        stat['percent.2'],
            #        stat['percent.98'],
            #        0.5 * (stat['percent.98'] - stat['percent.2']),
            #    )
            # )
            # Say.say(
            #    '99.7%: {:.2f} {:.2f} | {:.2f}'.format(
            #        stat['percent.0.1'],
            #        stat['percent.99.9'],
            #        0.5 * (stat['percent.99.9'] - stat['percent.0.1']),
            #    )
            # )


# --------------------------------------------------------------------------------------------------
# visualize
# --------------------------------------------------------------------------------------------------
class ImageClass(ut.io.SayClass):
    '''
    Plot 2-D image[s], save values, write to file.
    '''

    def __init__(self):
        '''
        .
        '''
        self.histogram_valuess = None
        self.histogram_xs = None
        self.histogram_ys = None
        self.plot_file_name = None

    def plot_image(
        self,
        part,
        species_name='dark',
        weight_name='mass',
        image_kind='histogram',
        dimensions_plot=[0, 1, 2],
        dimensions_select=[0, 1, 2],
        distances_max=1000,
        distance_bin_width=1,
        distance_bin_number=None,
        center_position=None,
        rotation=None,
        host_index=0,
        property_select={},
        part_indices=None,
        subsample_factor=None,
        use_column_units=None,
        image_limits=[None, None],
        background_color='black',
        hal=None,
        hal_indices=None,
        hal_position_kind='position',
        hal_radius_kind='radius',
        plot_file_name=False,
        plot_directory='.',
        figure_index=1,
    ):
        '''
        Plot image of the positions of given partcle species, using either a single panel for
        2 dimensions or 3 panels for all axis permutations.

        Parameters
        ----------
        part : dict : catalog of particles
        species_name : str : name of particle species to plot
        weight_name : str : property to weight positions by
        image_kind : str : 'histogram', 'histogram.3d', 'points'
        dimensions_plot : list : which dimensions to plot
            if length 2, plot one v other, if length 3, plot all via 3 panels
        dimensions_select : list : which dimensions to use to select particles
            note : use this to set selection 'depth' of an image
        distances_max : float or array : distance[s] from center to plot and/or cut
        distance_bin_width : float : length of pixel
        distance_bin_number : number of pixels from distance = 0 to max (2x this across image)
        center_position : array-like : position of center
        rotation : bool or array : whether to rotate particles - two options:
          (a) if input array of eigen-vectors, will define rotation axes
          (b) if True, will rotate to align with principal axes defined by input species
        host_index : int : index of host halo to get position and rotation of (if not input them)
        property_select : dict : (other) properties to select on: names as keys and limits as values
        part_indices : array : input selection indices for particles
        subsample_factor : int : factor by which periodically to sub-sample particles
        use_column_units : bool : whether to convert to particle number / cm^2
        image_limits : list : min and max limits to impose on image dynamic range (exposure)
        background_color : str : name of color for background: 'white', 'black'
        hal : dict : catalog of halos at snapshot
        hal_indices : array : indices of halos to plot
        hal_position_kind : str : name of position to use for center of halo
        hal_radius_kind : str : name of radius to use for size of halo
        plot_file_name : str
            whether to write figure to file and its name. True = use default naming convention
        plot_directory : str : path + directory where to write file
            if ends in '.pdf', override default file naming convention and use input name
        add_simulation_name : bool : whether to add name of simulation to figure name
        figure_index : int : index of figure for matplotlib
        '''
        dimen_label = {0: 'x', 1: 'y', 2: 'z'}

        if dimensions_select is None or len(dimensions_select) == 0:
            dimensions_select = dimensions_plot

        if np.isscalar(distances_max):
            distances_max = [
                distances_max for dimen_i in range(part[species_name]['position'].shape[1])
            ]
        distances_max = np.array(distances_max, dtype=np.float64)

        position_limits = []
        for dimen_i in range(distances_max.shape[0]):
            position_limits.append([-distances_max[dimen_i], distances_max[dimen_i]])
        position_limits = np.array(position_limits)

        if part_indices is None or len(part_indices) == 0:
            part_indices = ut.array.get_arange(part[species_name]['position'].shape[0])

        if property_select:
            part_indices = ut.catalog.get_indices_catalog(
                part[species_name], property_select, part_indices
            )

        if subsample_factor is not None and subsample_factor > 1:
            part_indices = part_indices[::subsample_factor]

        positions = np.array(part[species_name]['position'][part_indices])
        weights = None
        if weight_name:
            weights = part[species_name].prop(weight_name, part_indices)

        center_position = ut.particle.parse_property(part, 'position', center_position, host_index)

        if center_position is not None and len(center_position) > 0:
            # re-orient to input center
            positions -= center_position
            positions *= part.snapshot['scalefactor']

            if rotation is not None:
                # rotate image
                if rotation is True:
                    # rotate according to principal axes
                    rotation_tensor = ut.particle.parse_property(part, 'rotation', None, host_index)
                elif len(rotation) > 0:
                    # use input rotation vectors
                    rotation_tensor = np.asarray(rotation)
                    if (
                        np.ndim(rotation_tensor) != 2
                        or rotation_tensor.shape[0] != positions.shape[1]
                        or rotation_tensor.shape[1] != positions.shape[1]
                    ):
                        raise ValueError(f'wrong shape for rotation = {rotation}')
                else:
                    raise ValueError(f'cannot parse rotation = {rotation}')

                positions = ut.coordinate.get_coordinates_rotated(positions, rotation_tensor)

            # keep only particles within distance limits
            masks = positions[:, dimensions_select[0]] <= distances_max[0]
            for dimen_i in dimensions_select:
                masks *= (positions[:, dimen_i] >= -distances_max[dimen_i]) * (
                    positions[:, dimen_i] <= distances_max[dimen_i]
                )

            positions = positions[masks]
            if weights is not None:
                weights = weights[masks]
        else:
            raise ValueError('need to input center_position')

        if distance_bin_width is not None and distance_bin_width > 0:
            position_bin_number = int(
                np.round(2 * np.max(distances_max[dimensions_plot]) / distance_bin_width)
            )
        elif distance_bin_number is not None and distance_bin_number > 0:
            position_bin_number = 2 * distance_bin_number
        else:
            raise ValueError('need to input either distance bin width or bin number')

        if hal is not None:
            # compile halos
            if hal_indices is None or len(hal_indices) == 0:
                hal_indices = ut.array.get_arange(hal['mass'])

            if 0 not in hal_indices:
                hal_indices = np.concatenate([[0], hal_indices])

            hal_positions = np.array(hal[hal_position_kind][hal_indices])
            if center_position is not None and len(center_position) > 0:
                hal_positions -= center_position
            hal_positions *= hal.snapshot['scalefactor']
            hal_radiuss = hal[hal_radius_kind][hal_indices]

            # initialize masks
            masks = hal_positions[:, dimensions_select[0]] <= distances_max[0]
            for dimen_i in dimensions_select:
                masks *= (hal_positions[:, dimen_i] >= -distances_max[dimen_i]) * (
                    hal_positions[:, dimen_i] <= distances_max[dimen_i]
                )

            hal_radiuss = hal_radiuss[masks]
            hal_positions = hal_positions[masks]

        # plot ----------
        BYW = colors.LinearSegmentedColormap('byw', ut.plot.color_map_dict['BlackYellowWhite'])
        plt.register_cmap(cmap=BYW)
        BBW = colors.LinearSegmentedColormap('bbw', ut.plot.color_map_dict['BlackBlueWhite'])
        plt.register_cmap(cmap=BBW)

        # set color map
        if background_color == 'black':
            if 'dark' in species_name:
                color_map = plt.get_cmap('bbw')
            elif species_name == 'gas':
                color_map = plt.cm.afmhot  # pylint: disable=no-member
            elif species_name == 'star':
                # color_map = plt.get_cmap('byw')
                color_map = plt.cm.afmhot  # pylint: disable=no-member
        elif background_color == 'white':
            color_map = plt.cm.YlOrBr  # pylint: disable=no-member

        # set interpolation method
        # interpolation='nearest'
        interpolation = 'bilinear'
        # interpolation='bicubic'
        # interpolation='gaussian'

        if len(dimensions_plot) == 2:
            fig, subplot = ut.plot.make_figure(
                figure_index,
                left=0.22,
                right=0.98,
                bottom=0.15,
                top=0.98,
                background_color=background_color,
            )

            subplot.set_xlim(position_limits[dimensions_plot[0]])
            subplot.set_ylim(position_limits[dimensions_plot[1]])

            subplot.set_xlabel(f'{dimen_label[dimensions_plot[0]]} $\\left[ {{\\rm kpc}} \\right]$')
            subplot.set_ylabel(f'{dimen_label[dimensions_plot[1]]} $\\left[ {{\\rm kpc}} \\right]$')

            if 'histogram' in image_kind:
                hist_valuess, hist_xs, hist_ys, hist_limits = self.get_histogram(
                    image_kind,
                    dimensions_plot,
                    position_bin_number,
                    position_limits,
                    positions,
                    weights,
                    use_column_units,
                )

                image_limits_use = hist_limits
                if image_limits is not None and len(image_limits) > 0:
                    if image_limits[0] is not None:
                        image_limits_use[0] = image_limits[0]
                    if image_limits[1] is not None:
                        image_limits_use[1] = image_limits[1]

                _Image = subplot.imshow(
                    hist_valuess.transpose(),
                    norm=colors.LogNorm(),
                    cmap=color_map,
                    aspect='auto',
                    interpolation=interpolation,
                    extent=np.concatenate(position_limits[dimensions_plot]),
                    vmin=image_limits[0],
                    vmax=image_limits[1],
                )

                # standard method
                # hist_valuess, hist_xs, hist_ys, _Image = subplot.hist2d(
                #    positions[:, dimensions_plot[0]], positions[:, dimensions_plot[1]],
                #    weights=weights, range=position_limits, bins=position_bin_number,
                #    norm=colors.LogNorm(),
                #    cmap=color_map,
                #    vmin=image_limits[0], vmax=image_limits[1],
                # )

                # plot average of property
                # hist_valuess = ut.math.Fraction.get_fraction(hist_valuess, grid_number)
                # subplot.imshow(
                #    hist_valuess.transpose(),
                #    #norm=colors.LogNorm(),
                #    cmap=color_map,
                #    aspect='auto',
                #    interpolation=interpolation,
                #    extent=np.concatenate(position_limits),
                #    vmin=np.min(weights), vmax=np.max(weights),
                # )

                fig.colorbar(_Image)

            elif image_kind == 'points':
                subplot.scatter(
                    positions[:, dimensions_plot[0]],
                    positions[:, dimensions_plot[1]],
                    marker='o',
                    c=weights,
                )
                # , markersize=2.0, markeredgecolor='red', markeredgewidth=0,
                # color='red', alpha=0.02)

            fig.gca().set_aspect('equal')

            # plot halos
            if hal is not None:
                for hal_position, hal_radius in zip(hal_positions, hal_radiuss):
                    print(hal_position, hal_radius)
                    circle = plt.Circle(
                        hal_position[dimensions_plot],
                        hal_radius,
                        color='w',
                        linewidth=1,
                        fill=False,
                    )
                    subplot.add_artist(circle)

        elif len(dimensions_plot) == 3:
            fig, subplots = ut.plot.make_figure(
                figure_index,
                [2, 2],
                left=0.22,
                right=0.97,
                bottom=0.16,
                top=0.97,
                background_color=background_color,
            )

            plot_dimension_iss = [
                [dimensions_plot[0], dimensions_plot[1]],
                [dimensions_plot[0], dimensions_plot[2]],
                [dimensions_plot[1], dimensions_plot[2]],
            ]

            subplot_iss = [[0, 0], [1, 0], [1, 1]]

            histogram_valuesss = []
            for plot_i, plot_dimension_is in enumerate(plot_dimension_iss):
                subplot_is = subplot_iss[plot_i]
                subplot = subplots[subplot_is[0], subplot_is[1]]

                hist_valuess, hist_xs, hist_ys, hist_limits = self.get_histogram(
                    image_kind,
                    plot_dimension_is,
                    position_bin_number,
                    position_limits,
                    positions,
                    weights,
                    use_column_units,
                )

                histogram_valuesss.append(hist_valuess)

                image_limits_use = hist_limits
                if image_limits is not None and len(image_limits) > 0:
                    if image_limits[0] is not None:
                        image_limits_use[0] = image_limits[0]
                    if image_limits[1] is not None:
                        image_limits_use[1] = image_limits[1]

                # ensure that tick labels do not overlap
                subplot.set_xlim(position_limits[plot_dimension_is[0]])
                subplot.set_ylim(position_limits[plot_dimension_is[1]])

                units_label = ' $\\left[ {\\rm kpc} \\right]$'
                if subplot_is == [0, 0]:
                    subplot.set_ylabel(dimen_label[plot_dimension_is[1]] + units_label)
                elif subplot_is == [1, 0]:
                    subplot.set_xlabel(dimen_label[plot_dimension_is[0]] + units_label)
                    subplot.set_ylabel(dimen_label[plot_dimension_is[1]] + units_label)
                elif subplot_is == [1, 1]:
                    subplot.set_xlabel(dimen_label[plot_dimension_is[0]] + units_label)

                _Image = subplot.imshow(
                    hist_valuess.transpose(),
                    norm=colors.LogNorm(),
                    cmap=color_map,
                    # aspect='auto',
                    interpolation=interpolation,
                    extent=np.concatenate(position_limits[plot_dimension_is]),
                    vmin=image_limits[0],
                    vmax=image_limits[1],
                )

                # default method
                # hist_valuess, hist_xs, hist_ys, _Image = subplot.hist2d(
                #    positions[:, plot_dimension_is[0]], positions[:, plot_dimension_is[1]],
                #    norm=colors.LogNorm(),
                #    weights=weights,
                #    range=position_limits, bins=position_bin_number,
                #    cmap=color_map
                # )

                # fig.colorbar(_Image)  # , ax=subplot)

                # plot halos
                if hal is not None:
                    for hal_position, hal_radius in zip(hal_positions, hal_radiuss):
                        circle = plt.Circle(
                            hal_position[plot_dimension_is],
                            hal_radius,
                            color='w',
                            linewidth=1,
                            fill=False,
                        )
                        subplot.add_artist(circle)

                    circle = plt.Circle((0, 0), 10, color='w', fill=False)
                    subplot.add_artist(circle)

                # subplot.axis('equal')
                # fig.gca().set_aspect('equal')

            if part.info['simulation.name']:
                ut.plot.make_label_legend(subplots[0, 1], part.info['simulation.name'])

            hist_valuess = np.array(histogram_valuesss)

        # get name and directory to write plot file
        if '.pdf' in plot_directory:
            # use input file name, write in current directory
            plot_file_name = plot_directory
            plot_directory = '.'
        elif plot_file_name is True or plot_file_name == '':
            # generate default file name
            prefix = part.info['simulation.name']

            prop = 'position'
            for dimen_i in dimensions_plot:
                prop += '.' + dimen_label[dimen_i]
            prop += '_d.{:.0f}'.format(np.max(distances_max[dimensions_plot]))

            plot_file_name = ut.plot.get_file_name(
                weight_name, prop, species_name, 'redshift', part.snapshot, prefix=prefix
            )

            if 'histogram' in image_kind:
                plot_file_name += '_i.{:.1f}-{:.1f}'.format(
                    np.log10(image_limits_use[0]), np.log10(image_limits_use[1])
                )
        ut.plot.parse_output(plot_file_name, plot_directory)

        self.histogram_valuess = hist_valuess
        self.histogram_xs = hist_xs
        self.histogram_ys = hist_ys
        self.plot_file_name = plot_file_name

    def get_histogram(
        self,
        image_kind,
        dimension_list,
        position_bin_number,
        position_limits,
        positions,
        weights,
        use_column_units=False,
    ):
        '''
        Get 2-D histogram, either by summing all partiles along 3rd dimension or computing the
        highest density along 3rd dimension.

        Parameters
        ----------
        image_kind : str : 'histogram', 'histogram.3d'
        dimension_list : list : indices of dimensions to plot
            if length 2, plot one v other, if length 3, plot all via 3 panels
        position_bin_number : number of pixels/bins across image
        position_limits : list or list of lists : min and max values of position to compute
        positions : array : 3-D positions
        weights : array : weight for each position
        use_column_units : bool : whether to convert to [number / cm^2]
        '''
        if '3d' in image_kind:
            # calculate maximum local density along projected dimension
            hist_valuess, (hist_xs, hist_ys, hist_zs) = np.histogramdd(
                positions, position_bin_number, position_limits, weights=weights, normed=False
            )
            # convert to 3-d density
            hist_valuess /= np.diff(hist_xs)[0] * np.diff(hist_ys)[0] * np.diff(hist_zs)[0]

            dimension_project = np.setdiff1d([0, 1, 2], dimension_list)

            # compute maximum density
            hist_valuess = np.max(hist_valuess, dimension_project)

        else:
            # project along single dimension
            hist_valuess, hist_xs, hist_ys = np.histogram2d(
                positions[:, dimension_list[0]],
                positions[:, dimension_list[1]],
                position_bin_number,
                position_limits[dimension_list],
                weights=weights,
                normed=False,
            )

            # convert to surface density
            hist_valuess /= np.diff(hist_xs)[0] * np.diff(hist_ys)[0]

            # convert to number density
            if use_column_units:
                hist_valuess *= ut.constant.hydrogen_per_sun * ut.constant.kpc_per_cm ** 2
                grid_number = hist_valuess.size
                lls_number = np.sum((hist_valuess > 1e17) * (hist_valuess < 2e20))
                dla_number = np.sum(hist_valuess > 2e20)
                self.say(
                    'covering fraction: LLS = {:.2e}, DLA = {:.2e}'.format(
                        lls_number / grid_number, dla_number / grid_number
                    )
                )

        masks = hist_valuess > 0
        self.say(
            'histogram min, med, max = {:.3e}, {:.3e}, {:.3e}'.format(
                hist_valuess[masks].min(), np.median(hist_valuess[masks]), hist_valuess[masks].max()
            )
        )

        hist_limits = np.array([hist_valuess[masks].min(), hist_valuess[masks].max()])

        return hist_valuess, hist_xs, hist_ys, hist_limits

    def print_values(self):
        '''
        Write 2-D histogram values of image to file.
        '''
        file_name = self.plot_file_name + '.txt'

        with open(file_name, 'w') as file_out:
            Write = ut.io.WriteClass(file_out, print_stdout=False)
            Write.write(
                '# pixel (smoothing) scale is {:.2f} kpc'.format(
                    self.histogram_xs[1] - self.histogram_xs[0]
                )
            )
            for ix in range(self.histogram_xs.size - 1):
                x = self.histogram_xs[ix] + 0.5 * (
                    self.histogram_xs[ix + 1] - self.histogram_xs[ix]
                )
                for iy in range(self.histogram_ys.size - 1):
                    y = self.histogram_ys[iy] + 0.5 * (
                        self.histogram_ys[iy + 1] - self.histogram_ys[iy]
                    )
                    Write.write(
                        '{:.3f} {:.3f} {:.3e} {:.3e} {:.3e}'.format(
                            x,
                            y,
                            self.histogram_valuess[0, ix, iy],
                            self.histogram_valuess[1, ix, iy],
                            self.histogram_valuess[2, ix, iy],
                        )
                    )


Image = ImageClass()


# --------------------------------------------------------------------------------------------------
# general property analysis
# --------------------------------------------------------------------------------------------------
def plot_property_distribution(
    parts,
    species_name='gas',
    property_name='density',
    property_limits=[],
    property_bin_width=None,
    property_bin_number=100,
    property_scaling='log',
    property_statistic='probability',
    weight_property_name='',
    distance_limits=[],
    center_positions=None,
    center_velocities=None,
    host_index=0,
    property_select={},
    part_indicess=None,
    axis_y_limits=[],
    axis_y_scaling='log',
    plot_file_name=None,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot distribution of property.

    Parameters
    ----------
    part : dict
        catalog of particles at snapshot
    species_name : str
        name of particle species
    property_name : str
        property name
    property_limits : list
        min and max limits of property
    property_bin_width : float
        width of property bin (use this or property_bin_number)
    property_bin_number : int
        number of bins within limits (use this or property_bin_width)
    property_scaling : str
        scaling of property: 'log', 'linear'
    property_statistic : str :
        statistic to plot: 'probability', 'probability.cum', 'histogram', 'histogram.cum'
    weight_property_name : str
        property to weight each particle by
    distance_limits : list
        min and max limits for distance from galaxy
    center_positions : array or list of arrays
        position[s] of galaxy center[s]
    center_velocities : array or list of arrays
        velocity[s] of galaxy center[s]
    host_index : int
        index of host halo to get position and velocity of (if not input)
    property_select : dict
        (other) properties to select on: names as keys and limits as values
    part_indicess : array or list of arrays
        indices of particles from which to select
    axis_y_limits : list
        min and max limits for y-axis
    axis_y_scaling : str
        'log', 'linear'
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory to write plot file
    figure_index : int
        index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_property_distribution)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)
    if 'velocity' in property_name:
        center_velocities = ut.particle.parse_property(
            parts, 'velocity', center_velocities, host_index
        )

    Stat = ut.statistic.StatisticClass()

    for part_i, part in enumerate(parts):
        if part_indicess[part_i] is not None and len(part_indicess[part_i]) > 0:
            part_indices = part_indicess[part_i]
        else:
            part_indices = ut.array.get_arange(part[species_name]['position'].shape[0])

        if property_select:
            part_indices = ut.catalog.get_indices_catalog(
                part[species_name], property_select, part_indices
            )

        if distance_limits:
            # [kpc physical]
            distances = ut.coordinate.get_distances(
                part[species_name]['position'][part_indices],
                center_positions[part_i],
                part.info['box.length'],
                part.snapshot['scalefactor'],
                total_distance=True,
            )
            part_indices = part_indices[ut.array.get_indices(distances, distance_limits)]

        if 'velocity' in property_name:
            orb = ut.particle.get_orbit_dictionary(
                part,
                species_name,
                part_indices,
                center_positions[part_i],
                center_velocities[part_i],
                host_index,
            )
            prop_values = orb[property_name]
        else:
            prop_values = part[species_name].prop(property_name, part_indices)

        if weight_property_name:
            weights = part[species_name].prop(weight_property_name, part_indices)
        else:
            weights = None

        Say.say(f'keeping {prop_values.size} {species_name} particles')

        Stat.append_to_dictionary(
            prop_values,
            property_limits,
            property_bin_width,
            property_bin_number,
            property_scaling,
            weights,
        )

        Stat.print_statistics(-1)
        print()

    colors = ut.plot.get_colors(len(parts))

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    y_values = np.array([Stat.distr[property_statistic][part_i] for part_i in range(len(parts))])

    ut.plot.set_axes_scaling_limits(subplot, property_scaling, property_limits, prop_values)
    ut.plot.set_axes_scaling_limits(
        subplot, None, None, None, axis_y_scaling, axis_y_limits, y_values
    )

    axis_x_label = ut.plot.Label.get_label(property_name, species_name=species_name, get_words=True)
    subplot.set_xlabel(axis_x_label)
    axis_y_label = ut.plot.Label.get_label(
        property_name, property_statistic, species_name, property_scaling, get_units=False
    )
    subplot.set_ylabel(axis_y_label)

    for part_i, part in enumerate(parts):
        subplot.plot(
            Stat.distr['bin.mid'][part_i],
            Stat.distr[property_statistic][part_i],
            color=colors[part_i],
            alpha=0.8,
            label=part.info['simulation.name'],
        )

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = ut.plot.get_file_name(
            property_name, 'distribution', species_name, snapshot_dict=part.snapshot
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_velocity_v_age(
    part,
    species_name='star',
    x_property_name='age',
    x_property_limits=[0, 13.5],
    x_property_bin_width=0.25,
    x_property_scaling='linear',
    y_property_limits=[0, 70],
    y_property_scaling='linear',
    center_position=None,
    host_index=0,
    part_indices=None,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    .
    '''
    center_position = ut.particle.parse_property(part, 'position', center_position, host_index)

    if part_indices is None or len(part_indices) == 0:
        part_indices = ut.array.get_arange(part[species_name].prop(x_property_name))

    distances = part[species_name].prop('host.distance.principal.cylindridal', part_indices)
    part_indices = ut.array.get_indices(distances[:, 0], [7, 9], part_indices)
    part_indices = ut.array.get_indices(distances[:, 2], [-1, 1], part_indices)

    y_prop_values = part[species_name].prop('host.velocity.principal.cylindrical', part_indices)[
        :, 2
    ]

    Bin = ut.binning.BinClass(x_property_limits, x_property_bin_width, scaling=x_property_scaling)

    ages = part[species_name].prop(x_property_name, part_indices)
    ages_future = ages * 10 ** np.random.normal(0, 0.04, ages.size)
    ages_now = ages * 10 ** np.random.normal(0, 0.08, ages.size)
    ages_past = ages * 10 ** np.random.normal(0, 0.18, ages.size)

    stat = Bin.get_statistics_of_array(ages, y_prop_values)
    stat_future = Bin.get_statistics_of_array(ages_future, y_prop_values)
    stat_now = Bin.get_statistics_of_array(ages_now, y_prop_values)
    stat_past = Bin.get_statistics_of_array(ages_past, y_prop_values)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    _axis_x_limits, _axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot,
        x_property_scaling,
        x_property_limits,
        ages,
        y_property_scaling,
        y_property_limits,
        y_prop_values,
    )

    # if 'log' in x_property_scaling:
    #    _x_prop_values = ut.math.get_log(ages)
    if 'log' in y_property_scaling:
        y_prop_values = ut.math.get_log(y_prop_values)

    axis_x_label = ut.plot.Label.get_label(
        x_property_name, species_name=species_name, get_words=True
    )
    subplot.set_xlabel(axis_x_label)
    subplot.set_ylabel('$\sigma_v$ [km/s]')

    colors = ut.plot.get_colors(4)

    stat_name = 'std'
    subplot.plot(
        stat['bin.mid'], 0.5 * stat[stat_name], color='black', alpha=0.7, label='no age uncertainty'
    )
    subplot.plot(
        stat['bin.mid'],
        0.5 * stat_future[stat_name],
        color=colors[1],
        alpha=0.7,
        label='10% age uncertainty',
    )
    subplot.plot(
        stat['bin.mid'],
        0.5 * stat_now[stat_name],
        color=colors[2],
        alpha=0.7,
        label='20% age uncertainty',
    )
    subplot.plot(
        stat['bin.mid'],
        0.5 * stat_past[stat_name],
        color=colors[3],
        alpha=0.7,
        label='50% age uncertainty',
    )

    ut.plot.make_legends(subplot)

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = 'test'
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_property_v_property(
    part,
    species_name='gas',
    x_property_name='log number.density',
    x_property_limits=[],
    x_property_scaling='linear',
    y_property_name='log temperature',
    y_property_limits=[],
    y_property_scaling='linear',
    property_bin_number=150,
    weight_by_mass=True,
    cut_percent=0,
    host_distance_limits=[0, 300],
    center_position=None,
    host_index=0,
    property_select={},
    part_indices=None,
    draw_statistics=False,
    plot_file_name=False,
    plot_directory='.',
    add_simulation_name=False,
    figure_index=1,
):
    '''
    Plot property v property.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species_name : str : name of particle species
    x_property_name : str : property name for x-axis
    x_property_limits : list : min and max limits to impose on x_property_name
    x_property_scaling : str : 'log', 'linear'
    y_property_name : str : property name for y-axis
    y_property_limits : list : min and max limits to impose on y_property_name
    y_property_scaling : str : 'log', 'linear'
    property_bin_number : int : number of bins for histogram along each axis
    weight_by_mass : bool : whether to weight property by particle mass
    host_distance_limits : list : min and max limits for distance from galaxy
    center_position : array : position of galaxy center
    host_index : int : index of host galaxy/halo to get position of (if not input)
    property_select : dict : (other) properties to select on: names as keys and limits as values
    part_indices : array : indices of particles from which to select
    draw_statistics : bool : whether to draw statistics (such as median) on figure
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    add_simulation_name : bool : whether to add name of simulation to figure name
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_property_v_property)

    center_position = ut.particle.parse_property(part, 'position', center_position, host_index)

    if part_indices is None or len(part_indices) == 0:
        part_indices = ut.array.get_arange(part[species_name].prop(x_property_name))

    if property_select:
        part_indices = ut.catalog.get_indices_catalog(
            part[species_name], property_select, part_indices
        )

    if (
        len(center_position) > 0
        and host_distance_limits is not None
        and len(host_distance_limits) > 0
    ):
        distances = ut.coordinate.get_distances(
            part[species_name]['position'][part_indices],
            center_position,
            part.info['box.length'],
            part.snapshot['scalefactor'],
            total_distance=True,
        )  # [kpc phy]
        part_indices = part_indices[ut.array.get_indices(distances, host_distance_limits)]

    x_prop_values = part[species_name].prop(x_property_name, part_indices)
    y_prop_values = part[species_name].prop(y_property_name, part_indices)
    masses = None
    if weight_by_mass:
        masses = part[species_name].prop('mass', part_indices)

    part_indices = ut.array.get_arange(part_indices)

    if x_property_limits:
        part_indices = ut.array.get_indices(x_prop_values, x_property_limits, part_indices)

    if y_property_limits:
        part_indices = ut.array.get_indices(y_prop_values, y_property_limits, part_indices)

    if cut_percent > 0:
        x_limits = ut.array.get_limits(x_prop_values[part_indices], cut_percent=cut_percent)
        y_limits = ut.array.get_limits(y_prop_values[part_indices], cut_percent=cut_percent)
        part_indices = ut.array.get_indices(x_prop_values, x_limits, part_indices)
        part_indices = ut.array.get_indices(y_prop_values, y_limits, part_indices)

    x_prop_values = x_prop_values[part_indices]
    y_prop_values = y_prop_values[part_indices]
    if weight_by_mass:
        masses = masses[part_indices]

    Say.say(f'keeping {x_prop_values.size} particles')

    if draw_statistics:
        stat_bin_number = int(np.round(property_bin_number / 10))
        Bin = ut.binning.BinClass(
            x_property_limits, None, stat_bin_number, False, x_property_scaling
        )
        stat = Bin.get_statistics_of_array(x_prop_values, y_prop_values)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot,
        x_property_scaling,
        x_property_limits,
        x_prop_values,
        y_property_scaling,
        y_property_limits,
        y_prop_values,
    )

    if 'log' in x_property_scaling:
        x_prop_values = ut.math.get_log(x_prop_values)
    if 'log' in y_property_scaling:
        y_prop_values = ut.math.get_log(y_prop_values)

    axis_x_label = ut.plot.Label.get_label(
        x_property_name, species_name=species_name, get_words=True
    )
    subplot.set_xlabel(axis_x_label)

    axis_y_label = ut.plot.Label.get_label(
        y_property_name, species_name=species_name, get_words=True
    )
    subplot.set_ylabel(axis_y_label)

    color_map = plt.cm.inferno_r  # pylint: disable=no-member
    # color_map = plt.cm.gist_heat_r  # pylint: disable=no-member
    # color_map = plt.cm.afmhot_r  # pylint: disable=no-member

    # """
    _valuess, _xs, _ys, _Image = plt.hist2d(
        x_prop_values,
        y_prop_values,
        property_bin_number,
        [axis_x_limits, axis_y_limits],
        norm=colors.LogNorm(),
        weights=masses,
        cmin=None,
        cmax=None,
        cmap=color_map,
    )

    # valuess, _xs, _ys = np.histogram2d(
    #    x_prop_values, y_prop_values, property_bin_number,
    #    [axis_x_limits, axis_y_limits],
    #    normed=False, weights=masses)

    # subplot.imshow(
    #    valuess.transpose(), norm=colors.LogNorm(), cmap=color_map,
    #    aspect='auto',
    #    interpolation='nearest',
    #    #interpolation='none',
    #    extent=(axis_x_limits[0], axis_x_limits[1], axis_y_limits[0], axis_y_limits[1]),
    #    #vmin=valuess.min(), vmax=valuess.max(),
    #    #label=label,
    # )
    # plt.colorbar()

    if draw_statistics:
        print(stat['bin.mid'])
        subplot.plot(stat['bin.mid'], stat['median'], color='black', linestyle='-', alpha=0.4)
        subplot.plot(stat['bin.mid'], stat['percent.16'], color='black', linestyle='--', alpha=0.3)
        subplot.plot(stat['bin.mid'], stat['percent.84'], color='black', linestyle='--', alpha=0.3)

    # distance legend
    if host_distance_limits is not None and len(host_distance_limits) > 0:
        label = ut.plot.Label.get_label('radius', property_limits=host_distance_limits)
        ut.plot.make_label_legend(subplot, label, 'best')

    if plot_file_name is True or plot_file_name == '':
        if add_simulation_name:
            prefix = part.info['simulation.name']
        else:
            prefix = ''

        plot_file_name = ut.plot.get_file_name(
            y_property_name,
            x_property_name,
            species_name,
            snapshot_dict=part.snapshot,
            host_distance_limits=host_distance_limits,
            prefix=prefix,
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_property_v_distance(
    parts,
    species_name='dark',
    property_name='mass',
    property_statistic='sum',
    property_scaling='log',
    weight_by_mass=False,
    property_limits=[],
    distance_limits=[0.1, 300],
    distance_bin_width=0.02,
    distance_scaling='log',
    dimension_number=3,
    rotation=None,
    other_axis_distance_limits=None,
    center_positions=None,
    center_velocities=None,
    host_index=0,
    property_select={},
    part_indicess=None,
    distance_reference=None,
    plot_nfw=False,
    plot_fit=False,
    fit_distance_limits=[],
    print_values=False,
    get_values=False,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshots)
    species_name : str : name of particle species to compute mass from
        options: 'dark', 'star', 'gas', 'baryon', 'total'
    property_name : str : property to get profile of
    property_statistic : str : statistic/type to plot:
        'sum, sum.cum, density, density.cum, vel.circ, sum.fraction, sum.cum.fraction,
        median, average'
    property_scaling : str : scaling for property (y-axis): 'log', 'linear'
    weight_by_mass : bool : whether to weight property by particle mass
    property_limits : list : limits to impose on y-axis
    distance_limits : list : min and max distance for binning
    distance_bin_width : float : width of distance bin
    distance_scaling : str : 'log', 'linear'
    dimension_number : int : number of spatial dimensions for profile
        note : if 1, get profile along minor axis, if 2, get profile along 2 major axes
    rotation : bool or array : whether to rotate particles - two options:
      (a) if input array of eigen-vectors, will define rotation axes
      (b) if True, will rotate to align with principal axes stored in species dictionary
    other_axis_distance_limits : float :
        min and max distances along other axis[s] to keep particles [kpc physical]
    center_positions : array or list of arrays : position of center for each particle catalog
    center_velocities : array or list of arrays : velocity of center for each particle catalog
    host_index : int : index of host halo to get position and/or velocity of (if not input them)
    property_select : dict : (other) properties to select on: names as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    distance_reference : float : reference distance at which to draw vertical line
    plot_nfw : bool : whether to overplot NFW profile: density ~ 1 / r
    plot_fit : bool : whether to overplot linear fit
    fit_distance_limits : list : min and max distance for fit
    print_values : bool : whether to print values plotted
    get_values : bool : whether to return values plotted
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)
    if 'velocity' in property_name:
        center_velocities = ut.particle.parse_property(
            parts, 'velocity', center_velocities, host_index
        )
    else:
        center_velocities = [center_velocities for _ in center_positions]
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)

    SpeciesProfile = ut.particle.SpeciesProfileClass(
        distance_scaling,
        distance_limits,
        width=distance_bin_width,
        dimension_number=dimension_number,
    )

    pros = []

    for part_i, part in enumerate(parts):
        pros_part = SpeciesProfile.get_profiles(
            part,
            species_name,
            property_name,
            property_statistic,
            weight_by_mass,
            host_index,
            center_positions[part_i],
            center_velocities[part_i],
            rotation,
            other_axis_distance_limits,
            property_select,
            part_indicess[part_i],
        )

        pros.append(pros_part)

    if print_values:
        # print results
        print(pros[0][species_name]['distance'])
        for part_i, pro in enumerate(pros):
            print(pro[species_name][property_statistic])

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    y_values = [pro[species_name][property_statistic] for pro in pros]
    _axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot,
        distance_scaling,
        distance_limits,
        None,
        property_scaling,
        property_limits,
        y_values,
    )

    if dimension_number in [2, 3]:
        axis_x_label = 'radius'
    elif dimension_number == 1:
        axis_x_label = 'height'
    axis_x_label = ut.plot.Label.get_label(axis_x_label, get_words=True)
    subplot.set_xlabel(axis_x_label)

    if property_statistic == 'vel.circ':
        label_property_name = 'vel.circ'
    else:
        label_property_name = property_name
    axis_y_label = ut.plot.Label.get_label(
        label_property_name, property_statistic, species_name, dimension_number=dimension_number
    )
    subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(len(parts))

    if (
        'fraction' in property_statistic
        or 'beta' in property_name
        or 'velocity.rad' in property_name
    ):
        if 'fraction' in property_statistic:
            y_values = [1, 1]
        elif 'beta' in property_name:
            y_values = [0, 0]
        elif 'velocity.rad' in property_name:
            y_values = [0, 0]
        subplot.plot(distance_limits, y_values, color='black', linestyle=':', alpha=0.3)

    if distance_reference is not None:
        subplot.plot(
            [distance_reference, distance_reference],
            axis_y_limits,
            color='black',
            linestyle=':',
            alpha=0.6,
        )

    if plot_nfw:
        pro = pros[0]
        distances_nfw = pro[species_name]['distance']
        # normalize to outermost distance bin
        densities_nfw = (
            np.ones(pro[species_name]['distance'].size) * pro[species_name][property_statistic][-1]
        )
        densities_nfw *= pro[species_name]['distance'][-1] / pro[species_name]['distance']
        subplot.plot(distances_nfw, densities_nfw, color='black', linestyle=':', alpha=0.6)

    # plot profiles
    if len(pros) == 1:
        alpha = None
        linewidth = 3.5
    else:
        alpha = 0.7
        linewidth = None

    for part_i, pro in enumerate(pros):
        color = colors[part_i]

        label = parts[part_i].info['simulation.name']
        if len(pros) > 1 and parts[0].info['simulation.name'] == parts[1].info['simulation.name']:
            label = '$z={:.1f}$'.format(parts[part_i].snapshot['redshift'])

        masks = pro[species_name][property_statistic] != 0  # plot only non-zero values
        subplot.plot(
            pro[species_name]['distance'][masks],
            pro[species_name][property_statistic][masks],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_fit:
        xs = pro[species_name]['distance']
        ys = pro[species_name][property_statistic]

        masks = np.isfinite(xs)
        if fit_distance_limits is not None and len(fit_distance_limits) > 0:
            masks = (xs >= min(fit_distance_limits)) * (xs < max(fit_distance_limits))

        fit_kind = 'exponential'
        # fit_kind = 'sech2.single'
        # fit_kind = 'sech2.double'

        from scipy.optimize import curve_fit
        from scipy import stats

        if fit_kind == 'exponential':
            if 'log' in distance_scaling:
                xs = np.log10(xs)
            if 'log' in property_scaling:
                ys = np.log10(ys)

            slope, intercept, _r_value, _p_value, _std_err = stats.linregress(xs[masks], ys[masks])

            print('# raw fit: slope = {:.3f}, intercept = {:.3f}'.format(slope, intercept))
            if 'log' in property_scaling and 'log' not in distance_scaling:
                print('# exponential fit:')
                print('  scale length = {:.3f} kpc'.format(-1 * np.log10(np.e) / slope))
                print('  normalization = 10^{:.2f} Msun / kpc^2'.format(intercept))

            ys_fit = intercept + slope * xs

            if 'log' in distance_scaling:
                xs = 10 ** xs
            if 'log' in property_scaling:
                ys_fit = 10 ** ys_fit

        elif fit_kind == 'sech2.single':

            def disk_height_single(xs, a, b):
                return a / np.cosh(xs / (2 * b)) ** 2

            params, _ = curve_fit(
                disk_height_single, xs[masks], ys[masks], [1e7, 0.5], bounds=[[0, 0], [1e14, 10]]
            )
            print('# single sech^2 fit:')
            print('  scale height = {:.2f} kpc'.format(params[1]))
            print('  normalization = {:.2e} Msun / kpc'.format(params[0] / 2))

            ys_fit = disk_height_single(xs, *params)

        elif fit_kind == 'sech2.double':

            def disk_height_double(xs, a, b, c, d):
                return a / np.cosh(xs / (2 * b)) ** 2 + c / np.cosh(xs / (2 * d)) ** 2

            params, _ = curve_fit(
                disk_height_double,
                xs[masks],
                ys[masks],
                [1e8, 0.1, 1e8, 2],
                bounds=[[10, 0.01, 10, 0.2], [1e14, 3, 1e14, 5]],
            )

            print('# double sech^2 fit:')
            print('* thin scale height = {:.3f} kpc'.format(params[1]))
            print('  normalization = {:.2e} Msun / kpc'.format(params[0] / 2))
            print('* thick scale height = {:.3f} kpc'.format(params[3]))
            print('  normalization = {:.2e} Msun / kpc'.format(params[2] / 2))

            ys_fit = disk_height_double(xs, *params)

        subplot.plot(xs, ys_fit, color='black', alpha=0.5, linewidth=3.5)

    distance_name = 'dist'
    if dimension_number == 2:
        distance_name += '.2d'
    elif dimension_number == 1:
        distance_name = 'height'

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = ut.plot.get_file_name(
            property_name + '.' + property_statistic,
            distance_name,
            species_name,
            snapshot_dict=parts[0].snapshot,
        )
        plot_file_name = plot_file_name.replace('.sum', '')
        plot_file_name = plot_file_name.replace('mass.vel.circ', 'vel.circ')
        plot_file_name = plot_file_name.replace('mass.density', 'density')
    ut.plot.parse_output(plot_file_name, plot_directory)

    if get_values:
        if len(parts) == 1:
            pros = pros[0]
        return pros


def print_densities(
    parts,
    species_names=['star', 'dark', 'gas'],
    distance_limitss=[[8.0, 8.4], [0, 2 * np.pi], [-1.1, 1.1]],
    coordinate_system='cylindrical',
    center_positions=None,
    center_velocities=None,
    rotation=True,
    host_index=0,
):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshots)
    species_names : str or list thereof: name of particle species to compute densities of
        options: 'dark', 'star', 'gas'
    distance_limitss : list of lists : min and max distances/positions
    coordinate_system : str : which coordinates to get positions in:
        'cartesian' (default), 'cylindrical', 'spherical'
    center_positions : array or list of arrays : position of center for each particle catalog
    center_velocities : array or list of arrays : velocity of center for each particle catalog
    rotation : bool or array : whether to rotate particles - two options:
      (a) if input array of eigen-vectors, will define rotation axes
      (b) if True, will rotate to align with principal axes stored in species dictionary
    host_index : int : index of host galaxy/halo to get position, velocity, and/or rotation
        tensor of (if not input them)
    property_select : dict : (other) properties to select on: names as keys and limits as values
    '''
    Say = ut.io.SayClass(print_densities)

    assert coordinate_system in ('cartesian', 'cylindrical', 'spherical')

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)
    center_velocities = ut.particle.parse_property(parts, 'velocity', center_velocities, host_index)

    for part_i, part in enumerate(parts):
        densities_2d = []
        densities_3d = []

        for spec_name in species_names:
            distances = ut.particle.get_distances_wrt_center(
                part,
                spec_name,
                None,
                center_positions[part_i],
                rotation,
                host_index,
                coordinate_system,
            )

            pis = None
            for dimen_i, distance_limits in enumerate(distance_limitss):
                pis = ut.array.get_indices(distances[:, dimen_i], distance_limits, pis)

            mass = np.sum(part[spec_name]['mass'][pis])

            # compute densities
            # compute surface area [pc^2]
            area = (
                np.pi
                * (max(distance_limitss[0]) ** 2 - min(distance_limitss[0]) ** 2)
                * ut.constant.kilo ** 2
            )
            area *= (max(distance_limitss[2]) - min(distance_limitss[2])) / (2 * np.pi)
            # compute voluem [pc^3]
            volume = area * (max(distance_limitss[1]) - min(distance_limitss[1])) * ut.constant.kilo
            density_2d = mass / area
            density_3d = mass / volume

            Say.say('{}:'.format(spec_name))
            Say.say('  density_2d = {:.5f} Msun / pc^2'.format(density_2d))
            Say.say('  density_3d = {:.5f} Msun / pc^3'.format(density_3d))

            densities_2d.append(density_2d)
            densities_3d.append(density_3d)

        Say.say('total:')
        Say.say('  density_2d = {:.5f} Msun / pc^2'.format(np.sum(densities_2d)))
        Say.say('  density_3d = {:.5f} Msun / pc^3'.format(np.sum(densities_3d)))


def plot_disk_orientation_v_property(
    parts,
    species_names=['star', 'star.25', 'star.young'],
    property_name='distance',
    property_limits=[1, 15],
    property_bin_width=1,
    property_scaling='linear',
    reference_distance_max=8.2,
    center_positions=None,
    host_index=0,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot orientation angle of the disk versus property_name.

    Parameters
    ----------
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshots)
    species_names : str or list : name[s] of particle species to compute
        options: 'star', 'gas', 'dark'
    property_name : str : which property to vary (along x-axis): 'distance', 'age'
    property_limits : list : min and max property for binning
    property_bin_width : float : width of property bin
    property_scaling : str : 'log', 'linear'
    reference_distance_max : float : reference distance to compute principal axes
    center_positions : array or list of arrays : position of center for each particle catalog
    host_index : int : index of host galaxy/halo to get stored position of (if not input it)
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    axis_index = 2  # which principal axis to measure orientation angle of
    gas_temperature_limits = [0, 5e4]  # [K]
    young_star_age_limits = [0, 1]  # [Gyr]

    Say = ut.io.SayClass(plot_disk_orientation_v_property)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)

    PropertyBin = ut.binning.BinClass(
        property_limits, property_bin_width, scaling=property_scaling, include_max=True
    )

    angles = np.zeros((len(parts), len(species_names), PropertyBin.number)) * np.nan

    for part_i, part in enumerate(parts):
        Say.say('{}'.format(part.info['simulation.name']))

        # compute reference principal axes using all stars out to reference_distance
        principal_axes = ut.particle.get_principal_axes(
            part,
            'star',
            reference_distance_max,
            age_limits=[0, 1],
            center_positions=center_positions[part_i],
            host_index=host_index,
            verbose=False,
        )
        reference_rotation = principal_axes['rotation'][axis_index]

        for spec_i, spec_name in enumerate(species_names):
            Say.say(f'  {spec_name}')

            if spec_name == 'gas':
                part_indices = ut.array.get_indices(
                    part[spec_name]['temperature'], gas_temperature_limits
                )
            elif 'star' in spec_name and 'young' in spec_name:
                part_indices = ut.array.get_indices(part['star'].prop('age'), young_star_age_limits)
                spec_name = 'star'
            elif 'star' in spec_name and '.25' in spec_name:
                star_age_limits = [0, np.percentile(part['star'].prop('age'), 25)]
                part_indices = ut.array.get_indices(part['star'].prop('age'), star_age_limits)
                spec_name = 'star'
            else:
                part_indices = None

            for prop_i, property_max in enumerate(PropertyBin.mins):
                if property_name == 'distance':
                    distance_max = property_max
                elif property_name == 'age':
                    part_indices = ut.array.get_indices(part['star'].prop('age'), [0, property_max])
                    distance_max = reference_distance_max

                principal_axes = ut.particle.get_principal_axes(
                    part,
                    spec_name,
                    distance_max,
                    center_positions=center_positions[part_i],
                    host_index=host_index,
                    part_indicess=part_indices,
                    verbose=False,
                )

                # get orientation of axis of interest
                axis_rotation = principal_axes['rotation'][axis_index]
                angle = np.arccos(np.dot(axis_rotation, reference_rotation))
                if angle is np.nan:
                    angle = 0  # sanity check, for exact alignment
                angle *= 180 / np.pi  # [degree]
                # if angle > 90:
                #    Say.say(
                #        '!   {:4.1f} kpc: {:.1f} deg (raw), min/maj = {:.2f}'.format(
                #            property_max, angle, principal_axes['axis.ratios'][0]
                #        )
                #    )
                #    angle = min(angle, 180 - angle)  # deal with possible flip

                angles[part_i, spec_i, prop_i] = angle

                if property_name == 'distance':
                    Say.say(
                        '  {:4.1f} kpc: {:.1f} deg, min/maj = {:.2f}'.format(
                            property_max, angle, principal_axes['axis.ratios'][0]
                        )
                    )
                elif property_name == 'age':
                    Say.say('  {:4.1f} Gyr: {:.1f} deg'.format(property_max, angle))

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    _axis_x_limits, _axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot, property_scaling, property_limits, None, 'linear', [0, None], angles
    )

    if property_name == 'distance':
        subplot.set_xlabel('maximum radius $\\left[ {{\\rm kpc}} \\right]$')
    else:
        subplot.set_xlabel('star maximum age $\\left[ {{\\rm Gyr}} \\right]$')
    subplot.set_ylabel('disk offset angle $\\left[ {{\\rm deg}} \\right]$')

    if len(parts) > len(species_names):
        colors = ut.plot.get_colors(len(parts))
    else:
        colors = ut.plot.get_colors(len(species_names))

    for part_i, part in enumerate(parts):
        for spec_i, spec_name in enumerate(species_names):
            if len(parts) > len(species_names):
                label = part.info['simulation.name']
                color = colors[part_i]
            else:
                label = spec_name
                color = colors[spec_i]

            subplot.plot(
                PropertyBin.mins, angles[part_i, spec_i], color=color, alpha=0.8, label=label
            )

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_file_name is True or plot_file_name == '':
        property_y = 'disk.orientation'
        if len(parts) == 1:
            property_y = parts[0].info['simulation.name'] + '_' + property_y
        plot_file_name = ut.plot.get_file_name(
            property_y, property_name, snapshot_dict=part.snapshot
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_disk_orientation_v_time(
    parts,
    time_kind='time.lookback',
    time_limits=[0, 13],
    time_scaling='linear',
    refrence_snapshot_index=gizmo_default.snapshot_index,
    axis_indices=[0, 1, 2],
    angle_limits=[0, 90],
    host_index=0,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot orientation angle of the disk versus time_kind, wrt
    Requires that you have read pre-compiled host rotation tensors in star_form_coordinates_*.hdf5.

    Parameters
    ----------
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshots)
    time_kind : str : time kind to use:s
        'time', 'time.lookback', 'age', 'redshift', 'scalefactor'
    time_limits : list : min and max limits of time_kind to impose
    time_width : float : width of time_kind bin
    time_scaling : str : scaling of time_kind: 'log', 'linear'
    refrence_snapshot_index : int : index of reference snapshot, that defines angle zero point
    axis_indices : list : which principal axes to plot the orientation angles of
    host_index : int : index of host galaxy/halo to get stored position of (if not input it)
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    # Say = ut.io.SayClass(plot_disk_orientation_v_time)

    if isinstance(parts, dict):
        parts = [parts]

    angles = np.zeros((len(parts), len(axis_indices), parts[0].hostz['rotation'].shape[0])) * np.nan

    for part_i, part in enumerate(parts):
        rotation_tensors = part.hostz['rotation'][:, host_index]
        reference_rotation_tensor = rotation_tensors[refrence_snapshot_index]
        for axis_ii, axis_i in enumerate(axis_indices):
            angles[part_i, axis_ii] = np.dot(
                rotation_tensors[:, axis_i], reference_rotation_tensor[axis_i]
            )

    masks = np.isfinite(angles)
    angles[masks] = np.arccos(angles[masks]) * 180 / np.pi  # [degree]

    if time_kind in ['time.lookback', 'age']:
        times = parts[0].Snapshot['time'][-1] - parts[0].Snapshot['time']
    else:
        times = parts[0].Snapshot[time_kind]

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    _axis_x_limits, _axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot, time_scaling, time_limits, times, 'linear', angle_limits, angles
    )

    subplot.set_xlabel(time_kind)
    subplot.set_ylabel('disk offset angle $\\left[ {{\\rm deg}} \\right]$')

    if len(parts) > len(axis_indices):
        colors = ut.plot.get_colors(len(parts))
    else:
        colors = ut.plot.get_colors(len(axis_indices))

    for part_i, part in enumerate(parts):
        for axis_ii, axis_i in enumerate(axis_indices):
            if len(parts) > len(axis_indices):
                label = part.info['simulation.name']
                color = colors[part_i]
            else:
                label = f'principal axis {axis_i}'
                color = colors[axis_ii]

            subplot.plot(times, angles[part_i, axis_ii], color=color, alpha=0.8, label=label)

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = f'disk.orientation_v_{time_kind}'
        if len(parts) == 1:
            property_y = parts[0].info['simulation.name'] + '_' + plot_file_name
        plot_file_name = ut.plot.get_file_name(
            property_y, plot_file_name, snapshot_dict=part.snapshot
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_velocity_distribution_of_halo(
    parts,
    species_name='star',
    property_name='velocity.tan',
    property_limits=[],
    property_bin_width=None,
    property_bin_number=100,
    property_scaling='linear',
    property_statistic='probability',
    distance_limits=[70, 90],
    center_positions=None,
    center_velocities=None,
    host_index=0,
    property_select={},
    part_indicess=None,
    axis_y_limits=[],
    axis_y_scaling='linear',
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot distribution of velocities.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species_name : str : name of particle species
    property_name : str : property name
    property_limits : list : min and max limits of property
    property_bin_width : float : width of property bin (use this or property_bin_number)
    property_bin_number : int : number of bins within limits (use this or property_bin_width)
    property_scaling : str : scaling of property: 'log', 'linear'
    property_statistic : str : statistic to plot:
        'probability', 'probability.cum', 'histogram', 'histogram.cum'
    distance_limits : list : min and max limits for distance from galaxy
    center_positions : array or list of arrays : position[s] of galaxy center[s]
    center_velocities : array or list of arrays : velocity[s] of galaxy center[s]
    host_index : int : index of host galaxy/halo to get position and/or velocity of
        (if not input them)
    property_select : dict : (other) properties to select on: names as keys and limits as values
    part_indicess : array or list of arrays : indices of particles from which to select
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : str : 'log', 'linear'
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_velocity_distribution_of_halo)

    if isinstance(parts, dict):
        parts = [parts]

    center_positions = ut.particle.parse_property(parts, 'position', center_positions, host_index)
    part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)
    if 'velocity' in property_name:
        center_velocities = ut.particle.parse_property(
            parts, 'velocity', center_velocities, host_index
        )

    Stat = ut.statistic.StatisticClass()

    for part_i, part in enumerate(parts):
        if part_indicess[part_i] is not None and len(part_indicess[part_i]) > 0:
            part_indices = part_indicess[part_i]
        else:
            part_indices = ut.array.get_arange(part[species_name]['position'].shape[0])

        if property_select:
            part_indices = ut.catalog.get_indices_catalog(
                part[species_name], property_select, part_indices
            )

        if distance_limits:
            # [kpc physical]
            distances = ut.coordinate.get_distances(
                part[species_name]['position'][part_indices],
                center_positions[part_i],
                part.info['box.length'],
                part.snapshot['scalefactor'],
                total_distance=True,
            )
            part_indices = part_indices[ut.array.get_indices(distances, distance_limits)]

        if 'velocity' in property_name:
            orb = ut.particle.get_orbit_dictionary(
                part,
                species_name,
                part_indices,
                center_positions[part_i],
                center_velocities[part_i],
                host_index,
            )
            prop_values = orb[property_name]
        else:
            prop_values = part[species_name].prop(property_name, part_indices)

        Say.say(f'keeping {prop_values.size} {species_name} particles')

        Stat.append_to_dictionary(
            prop_values, property_limits, property_bin_width, property_bin_number, property_scaling
        )

        # Stat.print_statistics(-1)
        # print()

    colors = ut.plot.get_colors(len(parts))

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    y_values = np.array([Stat.distr[property_statistic][part_i] for part_i in range(len(parts))])

    ut.plot.set_axes_scaling_limits(subplot, property_scaling, property_limits, prop_values)
    ut.plot.set_axes_scaling_limits(
        subplot, None, None, None, axis_y_scaling, axis_y_limits, y_values
    )

    axis_x_label = ut.plot.Label.get_label(property_name, species_name=species_name, get_words=True)
    subplot.set_xlabel(axis_x_label)
    axis_y_label = ut.plot.Label.get_label(
        property_name, property_statistic, species_name, get_units=False
    )
    subplot.set_ylabel(axis_y_label)

    for part_i, part in enumerate(parts):
        subplot.plot(
            Stat.distr['bin.mid'][part_i],
            Stat.distr[property_statistic][part_i],
            color=colors[part_i],
            alpha=0.8,
            label=part.info['simulation.name'],
        )

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = ut.plot.get_file_name(
            property_name, 'distribution', species_name, snapshot_dict=part.snapshot
        )
    ut.plot.parse_output(plot_file_name, plot_directory)


# --------------------------------------------------------------------------------------------------
# properties of halos
# --------------------------------------------------------------------------------------------------
def assign_vel_circ_at_radius(
    part,
    hal,
    radius=0.6,
    sort_property_name='vel.circ.max',
    sort_property_value_min=20,
    halo_number_max=100,
    host_distance_limits=[1, 310],
):
    '''
    .
    '''
    Say = ut.io.SayClass(assign_vel_circ_at_radius)

    his = ut.array.get_indices(hal.prop('mass.bound/mass'), [0.1, Inf])
    his = ut.array.get_indices(hal['host.distance'], host_distance_limits, his)
    his = ut.array.get_indices(hal[sort_property_name], [sort_property_value_min, Inf], his)
    Say.say(f'{his.size} halos within limits')

    his = his[np.argsort(hal[sort_property_name][his])]
    his = his[::-1][:halo_number_max]

    mass_key = 'vel.circ.rad.{:.1f}'.format(radius)
    hal[mass_key] = np.zeros(hal['mass'].size)
    dark_mass = np.median(part['dark']['mass'])

    for hii, hi in enumerate(his):
        if hii > 0 and hii % 10 == 0:
            ut.io.print_flush(hii)
        pis = ut.particle.get_indices_within_coordinates(
            part, 'dark', [0, radius], hal['position'][hi]
        )
        hal[mass_key][hi] = ut.halo_property.get_circular_velocity(pis.size * dark_mass, radius)


def plot_vel_circ_v_radius_halos(
    parts=None,
    hals=None,
    part_indicesss=None,
    hal_indicess=None,
    pros=None,
    gal=None,
    total_mass_limits=None,
    star_mass_limits=[1e5, Inf],
    host_distance_limits=[1, 310],
    sort_property_name='vel.circ.max',
    sort_property_value_min=15,
    halo_number_max=20,
    vel_circ_limits=[0, 50],
    vel_circ_scaling='linear',
    radius_limits=[0.1, 3],
    radius_bin_width=0.1,
    radius_scaling='log',
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
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
            his = ut.array.get_indices(hal.prop('mass.bound/mass'), [0.1, Inf], his)
            his = ut.array.get_indices(hal['mass'], total_mass_limits, his)
            his = ut.array.get_indices(hal['host.distance'], host_distance_limits, his)

            if 'star.indices' in hal:
                his = ut.array.get_indices(hal['star.mass'], star_mass_limits, his)
            else:
                his = ut.array.get_indices(
                    hal[sort_property_name], [sort_property_value_min, Inf], his
                )
                his = his[np.argsort(hal[sort_property_name][his])[::-1]]
                his = his[:halo_number_max]

                Say.say(
                    '{} halos with {} [min, max] = [{:.3f}, {:.3f}]'.format(
                        his.size,
                        sort_property_name,
                        hal[sort_property_name][his[0]],
                        hal[sort_property_name][his[-1]],
                    )
                )

            hiss.append(his)

    gal_indices = None
    if gal is not None:
        gal_indices = ut.array.get_indices(gal['star.mass'], star_mass_limits)
        gal_indices = ut.array.get_indices(gal['host.distance'], host_distance_limits, gal_indices)
        gal_indices = gal_indices[gal['host.name'][gal_indices] == 'MW'.encode()]

    pros = plot_property_v_distance_halos(
        parts,
        hals,
        part_indicesss,
        hiss,
        pros,
        gal,
        gal_indices,
        'total',
        'mass',
        'vel.circ',
        vel_circ_scaling,
        False,
        vel_circ_limits,
        radius_limits,
        radius_bin_width,
        radius_scaling,
        3,
        None,
        plot_file_name,
        plot_directory,
        figure_index,
    )

    # plot_property_v_distance_halos(
    #    parts, hals, part_indicesss, hiss,
    #    None,
    #    gal, gal_indices,
    #    'star', 'velocity.total', 'std.cum', vel_circ_scaling, True, vel_circ_limits,
    #    radius_limits, radius_bin_width, radius_scaling, 3, False,
    #    plot_file_name, plot_directory, figure_index)

    return pros


def plot_property_v_distance_halos(
    parts=None,
    hals=None,
    part_indicesss=None,
    hal_indicess=None,
    pros=None,
    gal=None,
    gal_indices=None,
    species_name='total',
    property_name='mass',
    property_statistic='vel.circ',
    property_scaling='linear',
    weight_by_mass=False,
    property_limits=[],
    distance_limits=[0.1, 3],
    distance_bin_width=0.1,
    distance_scaling='log',
    dimension_number=3,
    distance_reference=None,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    parts : dict or list : catalog[s] of particles at snapshot
    hals : dict or list : catalog[s] of halos at snapshot
    part_indicesss : array (halo catalog number x halo number x particle number) :
    hal_indicess : array (halo catalog number x halo number) : indices of halos to plot
    gal : dict : catalog of observed galaxies
    gal_indices : array : indices of galaxies to plot
    species_name : str : name of particle species to compute mass from
        options: 'dark', 'star', 'gas', 'baryon', 'total'
    property_name : str : property to get profile of
    property_statistic : str : statistic/type to plot:
        'sum', sum.cum, density, density.cum, vel.circ, sum.fraction, sum.cum.fraction, median, ave'
    property_scaling : str : scaling for property (y-axis): 'log', 'linear'
    weight_by_mass : bool : whether to weight property by particle mass
    property_limits : list : limits to impose on y-axis
    distance_limits : list : min and max distance for binning
    distance_bin_width : float : width of distance bin
    distance_scaling : str : 'log', 'linear'
    dimension_number : int : number of spatial dimensions for profile
    distance_reference : float : reference distance at which to draw vertical line
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
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
    distance_limits_bin = [
        distance_limits[0] - distance_bin_width,
        distance_limits[1] + distance_bin_width,
    ]

    SpeciesProfile = ut.particle.SpeciesProfileClass(
        distance_scaling,
        distance_limits_bin,
        width=distance_bin_width,
        dimension_number=dimension_number,
    )

    if pros is None:
        pros = []
        if hals is not None:
            for cat_i, hal in enumerate(hals):
                part = parts[cat_i]
                hal_indices = hal_indicess[cat_i]

                if species_name == 'star' and hal['star.position'].max() > 0:
                    position_kind = 'star.position'
                    velocity_kind = 'star.velocity'
                elif species_name == 'dark' and hal['dark.position'].max() > 0:
                    position_kind = 'dark.position'
                    velocity_kind = 'dark.velocity'
                else:
                    # position_kind = 'position'
                    # velocity_kind = 'velocity'
                    position_kind = 'dark.position'
                    velocity_kind = 'dark.velocity'

                pros_cat = []

                for hal_i in hal_indices:
                    if part_indicesss is not None:
                        part_indices = part_indicesss[cat_i][hal_i]
                    elif species_name == 'star' and 'star.indices' in hal:
                        part_indices = hal['star.indices'][hal_i]
                    # elif species == 'dark' and 'dark.indices' in hal:
                    #    part_indices = hal['dark.indices'][hal_i]
                    else:
                        part_indices = None

                    pro_hal = SpeciesProfile.get_profiles(
                        part,
                        species_name,
                        property_name,
                        property_statistic,
                        weight_by_mass,
                        center_position=hal[position_kind][hal_i],
                        center_velocity=hal[velocity_kind][hal_i],
                        part_indicess=part_indices,
                    )

                    pros_cat.append(pro_hal)
                pros.append(pros_cat)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    y_values = []
    for pro_cat in pros:
        for pro_hal in pro_cat:
            y_values.append(pro_hal[species_name][property_statistic])

    ut.plot.set_axes_scaling_limits(
        subplot,
        distance_scaling,
        distance_limits,
        None,
        property_scaling,
        property_limits,
        y_values,
    )

    if 'log' in distance_scaling:
        subplot.xaxis.set_ticks([0.1, 0.2, 0.3, 0.5, 1, 2])
        subplot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    else:
        subplot.xaxis.set_minor_locator(AutoMinorLocator(2))

    # subplot.yaxis.set_minor_locator(AutoMinorLocator(5))

    axis_x_label = ut.plot.Label.get_label('radius', get_words=True)
    subplot.set_xlabel(axis_x_label)

    if property_statistic in ['vel.circ']:
        label_property_name = property_statistic
    else:
        label_property_name = property_name
    axis_y_label = ut.plot.Label.get_label(
        label_property_name, property_statistic, species_name, dimension_number=dimension_number
    )
    subplot.set_ylabel(axis_y_label)

    # draw reference values
    if (
        'fraction' in property_statistic
        or 'beta' in property_name
        or 'velocity.rad' in property_name
    ):
        if 'fraction' in property_statistic:
            y_values = [1, 1]
        elif 'beta' in property_name:
            y_values = [0, 0]
        elif 'velocity.rad' in property_name:
            y_values = [0, 0]
        subplot.plot(
            distance_limits, y_values, color='black', linestyle=':', alpha=0.5, linewidth=2
        )

    if distance_reference is not None:
        subplot.plot(
            [distance_reference, distance_reference],
            property_limits,
            color='black',
            linestyle=':',
            alpha=0.6,
        )

    # draw halos
    if hals is not None:
        colors = ut.plot.get_colors(len(hals))
        for cat_i, hal in enumerate(hals):
            hal_indices = hal_indicess[cat_i]
            for hal_ii, hal_i in enumerate(hal_indices):
                color = colors[cat_i]
                linewidth = 1.9
                alpha = 0.5

                if pros[cat_i][hal_ii][species_name][property_statistic][0] > 12.5:  # dark vel.circ
                    color = ut.plot.get_color('blue.lite')
                    linewidth = 3.0
                    alpha = 0.8

                if species_name == 'star':
                    linewidth = 2.0
                    alpha = 0.6
                    color = ut.plot.get_color('orange.mid')
                    if pros[cat_i][hal_ii][species_name][property_statistic][-1] > 27:
                        color = ut.plot.get_color('orange.lite')
                        linewidth = 3.5
                        alpha = 0.9

                subplot.plot(
                    pros[cat_i][hal_ii][species_name]['distance'],
                    pros[cat_i][hal_ii][species_name][property_statistic],
                    color=color,
                    linestyle='-',
                    alpha=alpha,
                    linewidth=linewidth,
                    # label=parts[part_i].info['simulation.name'],
                )

    # draw observed galaxies
    if gal is not None:
        gis = ut.array.get_indices(gal['star.radius.50'], distance_limits, gal_indices)
        gis = gis[gal['host.name'][gis] == 'MW'.encode()]
        print(gal['vel.circ.50'][gis] / gal['star.vel.std'][gis])
        for gal_i in gis:
            subplot.errorbar(
                gal['star.radius.50'][gal_i],
                gal['vel.circ.50'][gal_i],
                [[gal['vel.circ.50.err.lo'][gal_i]], [gal['vel.circ.50.err.hi'][gal_i]]],
                color='black',
                marker='s',
                markersize=10,
                alpha=0.7,
                capthick=2.5,
            )

    ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

    if plot_file_name is True or plot_file_name == '':
        snapshot_dict = None
        if parts is not None:
            snapshot_dict = parts[0].snapshot
        plot_file_name = ut.plot.get_file_name(
            property_name + '.' + property_statistic,
            'dist',
            species_name,
            snapshot_dict=snapshot_dict,
        )
        plot_file_name = plot_file_name.replace('.sum', '')
        plot_file_name = plot_file_name.replace('mass.vel.circ', 'vel.circ')
        plot_file_name = plot_file_name.replace('mass.density', 'density')
    ut.plot.parse_output(plot_file_name, plot_directory)

    return pros


# --------------------------------------------------------------------------------------------------
# mass and star-formation history
# --------------------------------------------------------------------------------------------------
class StarFormHistoryClass(ut.io.SayClass):
    '''
    .
    '''

    def plot_star_form_history(
        self,
        parts=None,
        sfh_kind='form.rate',
        time_kind='time.lookback',
        time_limits=[0, 13],
        time_width=0.2,
        time_scaling='linear',
        distance_limits=[0, 15],
        center_positions=None,
        host_index=0,
        property_select={},
        part_indicess=None,
        sfh_limits=[],
        sfh_scaling='log',
        verbose=False,
        plot_file_name=False,
        plot_directory='.',
        figure_index=1,
    ):
        '''
        Plot star-formation history v time_kind.

        Parameters
        ----------
        parts : dict or list : catalog[s] of particles
        sfh_kind : str : star formation kind to plot:
            'form.rate', 'form.rate.specific', 'mass', 'mass.normalized'
        time_kind : str : time kind to use:
            'time', 'time.lookback', 'age', 'redshift', 'scalefactor'
        time_limits : list : min and max limits of time_kind to impose
        time_width : float : width of time_kind bin
        time_scaling : str : scaling of time_kind: 'log', 'linear'
        distance_limits : list : min and max limits of distance to select star particles
        center_positions : list or list of lists : position[s] of galaxy centers [kpc comoving]
        host_index : int : index of host galaxy/halo to get position of
            (if not input center_position)
        property_select : dict : properties to select on: names as keys and limits as values
        part_indicess : array : part_indices of particles from which to select
        sfh_limits : list : min and max limits for y-axis
        sfh_scaling : str : scaling of y-axis: 'log', 'linear'
        plot_file_name : str
            whether to write figure to file and its name. True = use default naming convention
        plot_directory : str : directory to write figure file
        figure_index : int : index of figure for matplotlib
        '''
        if isinstance(parts, dict):
            parts = [parts]

        center_positions = ut.particle.parse_property(
            parts, 'position', center_positions, host_index
        )
        part_indicess = ut.particle.parse_property(parts, 'indices', part_indicess)

        time_limits = np.array(time_limits)
        if None in time_limits:
            if time_kind == 'redshift':
                if time_limits[0] is None:
                    time_limits[0] = np.floor(parts[0].snapshot[time_kind])
                if time_limits[1] is None:
                    time_limits[1] = 7
            elif time_kind == 'time':
                if time_limits[0] is None:
                    time_limits[0] = 0
                elif time_limits[1] is None:
                    time_limits[1] = parts[0].snapshot[time_kind]
            elif time_kind == 'time.lookback':
                if time_limits[0] is None:
                    time_limits[0] = 0
                elif time_limits[1] is None:
                    # time_limits[1] = 13.6  # [Gyr]
                    time_limits[1] = parts[0]['star'].prop('age').max()  # [Gyr]

        sfh = {}

        for part_i, part in enumerate(parts):
            sfh_p = self._get_star_form_history(
                part,
                time_kind,
                time_limits,
                time_width,
                time_scaling,
                distance_limits,
                center_positions[part_i],
                property_select,
                part_indicess[part_i],
            )

            if part_i == 0:
                for k in sfh_p:
                    sfh[k] = []  # initialize

            for k in sfh_p:
                sfh[k].append(sfh_p[k])

            if verbose:
                self.say(
                    'M_star max = {}'.format(
                        ut.io.get_string_from_numbers(sfh_p['mass'].max(), 2, exponential=True)
                    )
                )

        if time_kind == 'redshift' and 'log' in time_scaling:
            time_limits += 1  # convert to z + 1 so log is well-defined

        # plot ----------
        left = None
        if 'specific' in sfh_kind:
            left = 0.215
        _fig, subplot = ut.plot.make_figure(figure_index, left=left, axis_secondary='x')

        y_values = None
        if sfh is not None:
            y_values = sfh[sfh_kind]

        ut.plot.set_axes_scaling_limits(
            subplot, time_scaling, time_limits, None, sfh_scaling, sfh_limits, y_values
        )

        axis_x_label = ut.plot.Label.get_label(time_kind, get_words=True)
        subplot.set_xlabel(axis_x_label)

        if sfh_kind == 'mass.normalized':
            axis_y_label = '$M_{\\rm star}(z) \, / \, M_{\\rm star}(z=0)$'
        else:
            axis_y_label = ut.plot.Label.get_label('star.' + sfh_kind)
        subplot.set_ylabel(axis_y_label)

        ut.plot.make_axis_secondary_time(subplot, time_kind, time_limits, parts[0].Cosmology)

        colors = ut.plot.get_colors(len(parts))

        for part_i, part in enumerate(parts):
            tis = sfh[sfh_kind][part_i] > 0
            if time_kind in ['redshift', 'time.lookback', 'age']:
                tis *= sfh[time_kind][part_i] >= parts[0].snapshot[time_kind] * 0.99
            else:
                tis *= sfh[time_kind][part_i] <= parts[0].snapshot[time_kind] * 1.01
            subplot.plot(
                sfh[time_kind][part_i][tis],
                sfh[sfh_kind][part_i][tis],
                color=colors[part_i],
                alpha=0.8,
                label=part.info['simulation.name'],
            )

        ut.plot.make_legends(subplot, time_value=parts[0].snapshot['redshift'])

        if plot_file_name is True or plot_file_name == '':
            time_value = None
            if time_kind == 'redshift' and min(time_limits) > 1.1 * parts[0].snapshot['redshift']:
                time_value = min(time_limits)
            plot_file_name = ut.plot.get_file_name(
                sfh_kind + '.history', time_kind, 'star', 'redshift', parts[0].snapshot, time_value,
            )
        ut.plot.parse_output(plot_file_name, plot_directory)

    def plot_star_form_history_galaxies(
        self,
        part=None,
        hal=None,
        gal=None,
        mass_kind='star.mass',
        mass_limits=[1e5, 1e9],
        property_select={},
        hal_indices=None,
        sfh_kind='mass.normalized',
        sfh_limits=[],
        sfh_scaling='linear',
        time_kind='time.lookback',
        time_limits=[13.7, 0],
        time_width=0.2,
        time_scaling='linear',
        plot_file_name=False,
        plot_directory='.',
        figure_index=1,
    ):
        '''
        Plot star-formation history v time_kind for multiple galaxies in a halo catalog.

        Parameters
        ----------
        part : dict : catalog of particles
        hal : dict : catalog of halos at snapshot
        gal : dict : catalog of galaxies in the Local Group with SFHs
        mass_kind : str : mass kind by which to select halos
        mass_limits : list : min and max limits to impose on mass_kind
        property_select : dict : properties to select on: names as keys and limits as values
        hal_indices : index or array : index[s] of halo[s] whose particles to plot
        sfh_kind : str : star form kind to plot:
            'rate', 'rate.specific', 'mass', 'mass.normalized'
        sfh_limits : list : min and max limits for y-axis
        sfh_scaling : str : scailng of y-axis: 'log', 'linear'
        time_kind : str : time kind to plot: 'time', 'time.lookback', 'age', 'redshift'
        time_limits : list : min and max limits of time_kind to plot
        time_width : float : width of time_kind bin
        time_scaling : str : scaling of time_kind: 'log', 'linear'
        plot_file_name : str
            whether to write figure to file and its name. True = use default naming convention
        plot_directory : str : directory to write figure file
        figure_index : int : index of figure for matplotlib
        '''
        time_limits = np.array(time_limits)
        if part is not None:
            if time_limits[0] is None:
                time_limits[0] = part.snapshot[time_kind]
            if time_limits[1] is None:
                time_limits[1] = part.snapshot[time_kind]

        sfh = None
        if hal is not None:
            if hal_indices is None or not len(hal_indices) > 0:
                hal_indices = ut.array.get_indices(hal.prop('star.number'), [2, Inf])

            if mass_limits is not None and len(mass_limits) > 0:
                hal_indices = ut.array.get_indices(hal.prop(mass_kind), mass_limits, hal_indices)

            if property_select:
                hal_indices = ut.catalog.get_indices_catalog(hal, property_select, hal_indices)

            hal_indices = hal_indices[np.argsort(hal.prop(mass_kind, hal_indices))]

            print(f'halo number = {hal_indices.size}')

            sfh = {}

            for hal_ii, hal_i in enumerate(hal_indices):
                part_indices = hal.prop('star.indices', hal_i)
                sfh_h = self._get_star_form_history(
                    part,
                    time_kind,
                    time_limits,
                    time_width,
                    time_scaling,
                    part_indices=part_indices,
                )

                if hal_ii == 0:
                    for k in sfh_h:
                        sfh[k] = []  # initialize

                for k in sfh_h:
                    sfh[k].append(sfh_h[k])

                string = 'id = {:8d}, star.mass = {:.3e}, particle.number = {}, distance = {:.0f}'
                self.say(
                    string.format(
                        hal_i,
                        sfh_h['mass'].max(),
                        part_indices.size,
                        hal.prop('host.distance', hal_i),
                    )
                )
                # print(hal.prop('position', hal_i))

            for k in sfh:
                sfh[k] = np.array(sfh[k])

            sfh['mass.normalized.median'] = np.median(sfh['mass.normalized'], 0)

        if time_kind == 'redshift' and 'log' in time_scaling:
            time_limits += 1  # convert to z + 1 so log is well-defined

        # plot ----------
        _fig, subplot = ut.plot.make_figure(figure_index, axis_secondary='x')

        y_values = None
        if sfh is not None:
            y_values = sfh[sfh_kind]

        ut.plot.set_axes_scaling_limits(
            subplot, time_scaling, time_limits, None, sfh_scaling, sfh_limits, y_values
        )

        subplot.xaxis.set_minor_locator(AutoMinorLocator(2))

        axis_x_label = ut.plot.Label.get_label(time_kind, get_words=True)
        subplot.set_xlabel(axis_x_label)

        if sfh_kind == 'mass.normalized':
            axis_y_label = '$M_{\\rm star}(z)\, / \, M_{\\rm star}(z=0)$'
        else:
            axis_y_label = ut.plot.Label.get_label('star.' + sfh_kind)
        subplot.set_ylabel(axis_y_label)

        ut.plot.make_axis_secondary_time(subplot, time_kind, time_limits, part.Cosmology)

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
                subplot.plot(
                    gal.sfh[gal_name][time_kind],
                    gal.sfh[gal_name][sfh_kind],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                    color=color,
                    label=label,
                )

        # draw simulated galaxies
        if hal is not None:
            label = '$M_{{\\rm star}}=$'
            subplot.plot(-1, -1, label=label)
            for hal_ii, hal_i in enumerate(hal_indices):
                linewidth = 2.5 + 0.1 * hal_ii
                # linewidth = 3.0
                mass = ut.io.get_string_from_numbers(sfh['mass'][hal_ii][-1], 1, exponential=True)
                label = f'${mass}\,{{\\rm M}}_\odot$'
                subplot.plot(
                    sfh[time_kind][hal_ii],
                    sfh[sfh_kind][hal_ii],
                    linewidth=linewidth,
                    color=colors[hal_ii],
                    alpha=0.55,
                    label=label,
                )

        # subplot.plot(sfh['time'][0], sfh['mass.normalized.median'],
        #             linewidth=4.0, color='black', alpha=0.5)

        ut.plot.make_legends(subplot, time_value=part.snapshot['redshift'])

        if plot_file_name is True or plot_file_name == '':
            snapshot_dict = None
            if part is not None:
                snapshot_dict = part.snapshot
            time_kind_file_name = 'redshift'
            if hal is None:
                time_kind_file_name = None
            host_distance_limits = None
            if 'host.distance' in property_select:
                host_distance_limits = property_select['host.distance']

            plot_file_name = ut.plot.get_file_name(
                sfh_kind,
                time_kind,
                'star',
                time_kind_file_name,
                snapshot_dict,
                host_distance_limits=host_distance_limits,
            )
            if gal is not None:
                plot_file_name += '_lg'
        ut.plot.parse_output(plot_file_name, plot_directory)

    def _get_star_form_history(
        self,
        part,
        time_kind='redshift',
        time_limits=[0, 8],
        time_width=0.1,
        time_scaling='linear',
        distance_limits=None,
        center_position=None,
        host_index=0,
        property_select={},
        part_indices=None,
    ):
        '''
        Get array of times and star-formation rate at each time.

        Parameters
        ----------
        part : dict : catalog of particles
        time_kind : str : time metric to use:
            'time', 'time.lookback', 'age', 'redshift', 'scalefactor'
        time_limits : list : min and max limits of time_kind to impose
        time_width : float : width of time_kind bin (in units set by time_scaling)
        time_scaling : str : scaling of time_kind: 'log', 'linear'
        distance_limits : list : min and max limits of galaxy distance to select star particles
        center_position : list : position of galaxy centers [kpc comoving]
        host_index : int : index of host galaxy/halo to get position of (if not input it)
        property_select : dict : dictionary with property names as keys and limits as values
        part_indices : array : indices of star particles to select

        Returns
        -------
        sfh : dictionary : arrays of SFH properties
        '''
        species = 'star'

        if part_indices is None:
            part_indices = ut.array.get_arange(part[species]['mass'])

        if property_select:
            part_indices = ut.catalog.get_indices_catalog(
                part['star'], property_select, part_indices
            )

        center_position = ut.particle.parse_property(part, 'position', center_position, host_index)

        if (
            center_position is not None
            and len(center_position) > 0
            and distance_limits is not None
            and len(distance_limits) > 0
            and (min(distance_limits) > 0 or max(distance_limits) < np.inf)
        ):
            distances = ut.coordinate.get_distances(
                part['star']['position'][part_indices],
                center_position,
                part.info['box.length'],
                part.snapshot['scalefactor'],
                total_distance=True,
            )  # [kpc physical]
            part_indices = part_indices[ut.array.get_indices(distances, distance_limits)]

        # get formation times of star particles, sorted from earliest
        part_indices_sort = part_indices[np.argsort(part[species].prop('form.time', part_indices))]
        form_times = part[species].prop('form.time', part_indices_sort)
        # need to store as 64-bit float to avoid precision error with cumsum below
        form_masses = part[species].prop('form.mass', part_indices_sort).astype(np.float64)
        current_masses = part[species]['mass'][part_indices_sort].astype(np.float64)

        # get time bins, ensure are ordered from earliest
        time_dict = part.Cosmology.get_time_bins(
            time_kind, time_limits, time_width, time_scaling, part.Cosmology
        )
        time_bins = np.sort(time_dict['time'])
        time_difs = np.diff(time_bins)

        form_mass_cum_bins = np.interp(time_bins, form_times, np.cumsum(form_masses))
        form_mass_difs = np.diff(form_mass_cum_bins)
        form_rate_bins = form_mass_difs / time_difs / ut.constant.giga  # convert to [M_sun / yr]

        current_mass_cum_bins = np.interp(time_bins, form_times, np.cumsum(current_masses))
        current_mass_difs = np.diff(current_mass_cum_bins)

        print(np.cumsum(current_masses))

        # convert to midpoints of bins
        current_mass_cum_bins = (
            current_mass_cum_bins[: current_mass_cum_bins.size - 1] + 0.5 * current_mass_difs
        )
        form_mass_cum_bins = (
            form_mass_cum_bins[: form_mass_cum_bins.size - 1] + 0.5 * form_mass_difs
        )

        for k in time_dict:
            time_dict[k] = time_dict[k][: time_dict[k].size - 1] + 0.5 * np.diff(time_dict[k])

        # ensure that ordering jives with ordering of input limits
        if time_dict['time'][0] > time_dict['time'][1]:
            form_rate_bins = form_rate_bins[::-1]
            current_mass_cum_bins = current_mass_cum_bins[::-1]
            form_mass_cum_bins = form_mass_cum_bins[::-1]

        sfh = {}
        for k in time_dict:
            sfh[k] = time_dict[k]
        sfh['form.rate'] = form_rate_bins
        sfh['form.rate.specific'] = form_rate_bins / form_mass_cum_bins
        sfh['mass'] = current_mass_cum_bins
        sfh['mass.normalized'] = current_mass_cum_bins / current_mass_cum_bins.max()
        sfh['particle.number'] = form_times.size

        return sfh


StarFormHistory = StarFormHistoryClass()


# --------------------------------------------------------------------------------------------------
# analysis across time
# --------------------------------------------------------------------------------------------------
def plot_gas_neutral_fraction_v_redshift(
    parts=None,
    redshift_limits=[6, 8.4],
    simulation_directory=gizmo_default.simulation_directory,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    .
    '''
    if parts is None:
        Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
        snapshot_indices = ut.array.get_indices(
            Snapshot['redshift'], [min(redshift_limits) * 0.99, max(redshift_limits) * 1.01]
        )
        redshifts = Snapshot['redshift'][snapshot_indices]

        Read = gizmo_io.ReadClass()

        parts = Read.read_snapshots(
            'gas',
            'index',
            snapshot_indices,
            simulation_directory,
            properties=['mass', 'density', 'hydrogen.neutral.fraction'],
            assign_hosts=False,
        )
    else:
        snapshot_indices = np.array([part.snapshot['index'] for part in parts], np.int32)
        redshifts = parts[0].Snapshot['redshift'][snapshot_indices]

    # Statistic = ut.statistic.StatisticClass()

    neutral_fraction_by_mass = {}
    neutral_fraction_by_volume = {}
    for part in parts:
        values = part['gas']['hydrogen.neutral.fraction']

        # weights = None
        weights = part['gas']['mass']
        # Stat = Statistic.get_statistic_dict(values, weights=weights)
        stat = {}
        # stat['median'] = ut.math.percentile_weighted(values, 50, weights)
        # stat['percent.16'] = ut.math.percentile_weighted(values, 16, weights)
        # stat['percent.84'] = ut.math.percentile_weighted(values, 84, weights)
        stat['average'] = np.sum(values * weights) / np.sum(weights)
        stat['std'] = np.sqrt(np.sum(weights / np.sum(weights) * (values - stat['average']) ** 2))
        stat['std.lo'] = stat['average'] - stat['std']
        stat['std.hi'] = stat['average'] + stat['std']
        ut.array.append_dictionary(neutral_fraction_by_mass, stat)

        weights = part['gas']['mass'] / part['gas']['density']
        # Stat = Statistic.get_statistic_dict(values, weights=weights)
        stat = {}
        # stat['median'] = ut.math.percentile_weighted(values, 50, weights)
        # stat['percent.16'] = ut.math.percentile_weighted(values, 16, weights)
        # stat['percent.84'] = ut.math.percentile_weighted(values, 84, weights)
        stat['average'] = np.sum(values * weights) / np.sum(weights)
        stat['std'] = np.sqrt(np.sum(weights / np.sum(weights) * (values - stat['average']) ** 2))
        stat['std.lo'] = stat['average'] - stat['std']
        stat['std.hi'] = stat['average'] + stat['std']

        ut.array.append_dictionary(neutral_fraction_by_volume, stat)

    ut.array.arrayize_dictionary(neutral_fraction_by_mass)
    ut.array.arrayize_dictionary(neutral_fraction_by_volume)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(subplot, 'linear', None, redshifts, 'linear', [0, 1])

    subplot.set_xlabel('redshift')
    subplot.set_ylabel('neutral fraction')

    colors = ut.plot.get_colors(2)

    # name_hi = 'percent.84'
    # name_mid = 'median'
    # name_lo = 'percent.16'

    name_hi = 'std.hi'
    name_mid = 'average'
    name_lo = 'std.lo'

    subplot.fill_between(
        redshifts,
        neutral_fraction_by_mass[name_lo],
        neutral_fraction_by_mass[name_hi],
        alpha=0.4,
        color=colors[0],
    )
    subplot.plot(redshifts, neutral_fraction_by_mass[name_mid], label='mass-weighted')

    subplot.fill_between(
        redshifts,
        neutral_fraction_by_volume[name_lo],
        neutral_fraction_by_volume[name_hi],
        alpha=0.4,
        color=colors[1],
    )
    subplot.plot(redshifts, neutral_fraction_by_volume[name_mid], label='volume-weighted')

    ut.plot.make_legends(subplot)

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = 'gas.neutral.frac_v_redshift'
    ut.plot.parse_output(plot_file_name, plot_directory)

    return parts


# --------------------------------------------------------------------------------------------------
# analysis with halo catalog
# --------------------------------------------------------------------------------------------------
def explore_galaxy(
    hal,
    hal_index=None,
    part=None,
    species_plot=['star'],
    distance_max=None,
    distance_bin_width=0.2,
    distance_bin_number=None,
    plot_only_members=True,
    plot_file_name=False,
    plot_directory='.',
):
    '''
    Print and plot several properties of galaxies in list.

    Parameters
    ----------
    hal : dict : catalog of halos at snapshot
    hal_index : int : index within halo catalog
    part : dict : catalog of particles at snapshot
    species_plot : str or dict : which particle species to plot
    distance_max : float : max distance (radius) for galaxy image
    distance_bin_width : float : length of pixel for galaxy image
    distance_bin_number : int : number of pixels for galaxy image
    plot_only_members : bool : whether to plat only particles that are members of halo
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    '''
    from halo_analysis import halo_plot

    halo_plot.print_properties(hal, hal_index)

    hi = hal_index

    if part is not None:
        if not distance_max and 'star.radius.90' in hal:
            distance_max = 2 * hal.prop('star.radius.90', hi)

        if 'star' in species_plot and 'star' in part and 'star.indices' in hal:
            part_indices = None
            if plot_only_members:
                part_indices = hal.prop('star.indices', hi)

            # image of member particles
            Image.plot_image(
                part,
                'star',
                'mass',
                'histogram',
                [0, 1, 2],
                [0, 1, 2],
                distance_max,
                distance_bin_width,
                distance_bin_number,
                hal.prop('star.position', hi),
                part_indices=part_indices,
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=10,
            )

            # image of all nearby particles
            Image.plot_image(
                part,
                'star',
                'mass',
                'histogram',
                [0, 1, 2],
                [0, 1, 2],
                distance_max * 4,
                distance_bin_width,
                distance_bin_number,
                hal.prop('star.position', hi),
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=11,
            )

            plot_property_distribution(
                part,
                'star',
                'velocity.total',
                [0, None],
                2,
                None,
                'linear',
                'histogram',
                [],
                hal.prop('star.position', hi),
                hal.prop('star.velocity', hi),
                0,
                {},
                part_indices,
                [0, None],
                'linear',
                plot_file_name,
                plot_directory,
                figure_index=12,
            )

            try:
                element_name = 'metallicity.iron'
                hal.prop('star.' + element_name)
            except KeyError:
                element_name = 'metallicity.total'

            plot_property_distribution(
                part,
                'star',
                element_name,
                [-4, 1],
                0.1,
                None,
                'linear',
                'histogram',
                part_indicess=part_indices,
                axis_y_limits=[0, None],
                axis_y_scaling='linear',
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=13,
            )

            plot_property_v_distance(
                part,
                'star',
                'mass',
                'density',
                'log',
                False,
                None,
                [0.1, distance_max],
                0.1,
                'log',
                3,
                center_positions=hal.prop('star.position', hi),
                part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=14,
            )

            # plot_property_v_distance(
            #    part, 'star', 'mass', 'sum.cum', 'log', False, None,
            #    [0.1, distance_max], 0.1, 'log', 3,
            #    center_positions=hal.prop('star.position', hi), part_indicess=part_indices,
            #    distance_reference=hal.prop('star.radius.50', hi),
            #    plot_file_name=plot_file_name, plot_directory=plot_directory, figure_index=15,
            # )

            plot_property_v_distance(
                part,
                'star',
                'velocity.total',
                'std.cum',
                'linear',
                True,
                None,
                [0.1, distance_max],
                0.1,
                'log',
                3,
                center_positions=hal.prop('star.position', hi),
                center_velocities=hal.prop('star.velocity', hi),
                part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=16,
            )

            plot_property_v_distance(
                part,
                'star',
                element_name,
                'median',
                'linear',
                True,
                None,
                [0.1, distance_max],
                0.2,
                'log',
                3,
                center_positions=hal.prop('star.position', hi),
                part_indicess=part_indices,
                distance_reference=hal.prop('star.radius.50', hi),
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=17,
            )

            StarFormHistory.plot_star_form_history(
                part,
                'mass.normalized',
                'time.lookback',
                [13.6, 0],
                0.2,
                'linear',
                part_indicess=part_indices,
                sfh_limits=[0, 1],
                sfh_scaling='linear',
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=18,
            )

        if 'dark' in species_plot and 'dark' in part:
            part_indices = None

            center_position = hal.prop('position', hi)
            # if 'star.position' in hal:
            #    center_position = hal.prop('star.position', hi)

            if 'star.radius.50' in hal:
                distance_reference = hal.prop('star.radius.50', hi)
            else:
                distance_reference = None

            # DM image centered on stars
            Image.plot_image(
                part,
                'dark',
                'mass',
                'histogram',
                [0, 1, 2],
                [0, 1, 2],
                distance_max,
                distance_bin_width,
                distance_bin_number,
                hal.prop('star.position', hi),
                background_color='black',
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=20,
            )

            # DM image centered on DM halo
            Image.plot_image(
                part,
                'dark',
                'mass',
                'histogram',
                [0, 1, 2],
                [0, 1, 2],
                distance_max,
                distance_bin_width,
                distance_bin_number,
                hal.prop('position', hi),
                background_color='black',
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=21,
            )

            plot_property_v_distance(
                part,
                'dark',
                'mass',
                'density',
                'log',
                False,
                None,
                [0.1, distance_max],
                0.1,
                'log',
                3,
                center_positions=center_position,
                part_indicess=part_indices,
                distance_reference=distance_reference,
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=22,
            )

            # plot_property_v_distance(
            #    part, 'dark', 'velocity.total', 'std.cum', 'linear', True, None,
            #    [0.1, distance_max], 0.1, 'log', 3,
            #    center_positions=center_position, center_velocities=center_velocity,
            #    part_indicess=part_indices,
            #    distance_reference=distance_reference,
            #    plot_file_name=plot_file_name, plot_directory=plot_directory, figure_index=23)

            plot_property_v_distance(
                part,
                'dark',
                'mass',
                'vel.circ',
                'linear',
                True,
                None,
                [0.1, distance_max],
                0.1,
                'log',
                3,
                center_positions=center_position,
                part_indicess=part_indices,
                distance_reference=distance_reference,
                plot_file_name=plot_file_name,
                plot_directory=plot_directory,
                figure_index=24,
            )

        if 'gas' in species_plot and 'gas' in part and 'gas.indices' in hal:
            part_indices = None
            if plot_only_members:
                part_indices = hal.prop('gas.indices', hi)

            if part_indices is None or len(part_indices) >= 3:
                Image.plot_image(
                    part,
                    'gas',
                    'mass',
                    'histogram',
                    [0, 1, 2],
                    [0, 1, 2],
                    distance_max,
                    distance_bin_width,
                    distance_bin_number,
                    hal.prop('star.position', hi),
                    part_indices=part_indices,
                    plot_file_name=plot_file_name,
                    plot_directory=plot_directory,
                    figure_index=30,
                )

                Image.plot_image(
                    part,
                    'gas',
                    'mass.neutral',
                    'histogram',
                    [0, 1, 2],
                    [0, 1, 2],
                    distance_max,
                    distance_bin_width,
                    distance_bin_number,
                    hal.prop('star.position', hi),
                    part_indices=part_indices,
                    plot_file_name=plot_file_name,
                    plot_directory=plot_directory,
                    figure_index=31,
                )
            else:
                fig = plt.figure(10)
                fig.clf()
                fig = plt.figure(11)
                fig.clf()


def plot_density_profile_halo(
    part,
    species_name='star',
    hal=None,
    hal_index=None,
    center_position=None,
    distance_limits=[0.1, 2],
    distance_bin_width=0.1,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot density profile for single halo/center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species_name : str : name of particle species to plot
    hal : dict : catalog of halos at snapshot
    hal_index : int : index of halo in catalog
    center_position : array : position to center profile (to use instead of halo position)
    distance_max : float : max distance (radius) for galaxy image
    distance_bin_width : float : length of pixel for galaxy image
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    distance_scaling = 'log'
    dimension_number = 3

    if center_position is None:
        center_positions = []
        # center_positions.append(hal['position'][hal_index])
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
        parts,
        species_name,
        'mass',
        'density',
        'log',
        False,
        None,
        distance_limits,
        distance_bin_width,
        distance_scaling,
        dimension_number,
        center_positions=center_positions,
        part_indicess=None,
        distance_reference=distance_reference,
        plot_file_name=plot_file_name,
        plot_directory=plot_directory,
        figure_index=figure_index,
    )


def plot_density_profiles_halos(
    part,
    hal,
    hal_indices,
    species_name='dark',
    density_limits=None,
    distance_limits=[0.05, 1],
    distance_bin_width=0.2,
    plot_only_members=False,
    plot_file_name=False,
    plot_directory='.',
    figure_index=0,
):
    '''
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
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
                part_indicess.append(hal.prop(species_name + '.indices', hal_i))
        else:
            center_positions.append(hal.prop('position', hal_i))
            if plot_only_members:
                part_indicess.append(hal.prop(species_name + '.indices', hal_i))

    plot_property_v_distance(
        parts,
        species_name,
        'mass',
        'density',
        'log',
        False,
        density_limits,
        distance_limits,
        distance_bin_width,
        'log',
        3,
        center_positions=center_positions,
        part_indicess=part_indicess,
        plot_file_name=plot_file_name,
        plot_directory=plot_directory,
        figure_index=figure_index,
    )


# --------------------------------------------------------------------------------------------------
# galaxy mass and radius at snapshots
# --------------------------------------------------------------------------------------------------
def write_galaxy_properties_v_time(
    simulation_directory=gizmo_default.simulation_directory, redshifts=[], species=['star']
):
    '''
    Read snapshots and store dictionary of host galaxy properties (such as mass and radius)
    at snapshots.

    Parameters
    ----------
    simulation_directory : str : root directory of simulation
    redshifts : array-like : redshifts at which to get properties
        'all' = read and store all snapshots
    species : str or list : name[s] of species to read and get properties of

    Returns
    -------
    gal : dict : dictionary of host galaxy properties at input redshifts
    '''
    Read = gizmo_io.ReadClass()

    star_distance_max = 15

    properties_read = ['mass', 'position']

    mass_percents = [50, 90]

    simulation_directory = ut.io.get_path(simulation_directory)

    gal = {'index': [], 'redshift': [], 'scalefactor': [], 'time': [], 'time.lookback': []}

    for spec_name in species:
        gal[f'{spec_name}.position'] = []
        for mass_percent in mass_percents:
            gal['{}.radius.{:.0f}'.format(spec_name, mass_percent)] = []
            gal['{}.mass.{:.0f}'.format(spec_name, mass_percent)] = []

    if redshifts == 'all' or redshifts is None or redshifts == []:
        Snapshot = ut.simulation.SnapshotClass()
        Snapshot.read_snapshots(gizmo_default.snapshot_time_file_name, simulation_directory)
        redshifts = Snapshot['redshift']
    else:
        if np.isscalar(redshifts):
            redshifts = [redshifts]

    redshifts = np.sort(redshifts)

    for _zi, redshift in enumerate(redshifts):
        part = Read.read_snapshots(
            species, 'redshift', redshift, simulation_directory, properties=properties_read
        )

        for k in ['index', 'redshift', 'scalefactor', 'time', 'time.lookback']:
            gal[k].append(part.snapshot[k])

        # get position and velocity
        gal['star.position'].append(part.host['position'][0])

        for spec_name in species:
            for mass_percent in mass_percents:
                gal_prop = ut.particle.get_galaxy_properties(
                    part, spec_name, 'mass.percent', mass_percent, distance_max=star_distance_max
                )
                k = '{}.radius.{:.0f}'.format(spec_name, mass_percent)
                gal[k].append(gal_prop['radius'])
                k = '{}.mass.{:.0f}'.format(spec_name, mass_percent)
                gal[k].append(gal_prop['radius'])

    for prop_name in gal:
        gal[prop_name] = np.array(gal[prop_name])

    ut.io.file_pickle(simulation_directory + 'host_properties_v_time', gal)

    return gal


def plot_galaxy_property_v_time(
    gals=None,
    sfhs=None,
    Cosmology=None,
    property_name='star.mass',
    time_kind='redshift',
    time_limits=[0, 8],
    time_scaling='linear',
    snapshot_subsample_factor=1,
    axis_y_limits=[],
    axis_y_scaling='log',
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot host galaxy property v time_kind, using tabulated dictionary of properties of progenitor
    across snapshots.

    Parameters
    ----------
    gals : dict : tabulated dictionary of host galaxy properties
    sfhs : dict : tabulated dictinnary of star-formation histories (computed at single snapshot)
    property_name : str : name of star formation history property to plot:
        'rate', 'rate.specific', 'mass', 'mass.normalized'
    time_kind : str : time kind to use: 'time', 'time.lookback', 'redshift'
    time_limits : list : min and max limits of time_kind to get
    time_scaling : str : scaling of time_kind: 'log', 'linear'
    snapshot_subsample_factor : int : factor by which to sub-sample snapshots from gals
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : str : scaling of y-axis: 'log', 'linear'
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    # Say = ut.io.SayClass(plot_galaxy_property_v_time)

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
    _fig, subplot = ut.plot.make_figure(figure_index)

    y_values = []
    if gals is not None:
        y_values.append(gals[0][property_name])
    if sfhs is not None:
        y_values.append(sfhs[0][time_kind])
    subplot.set_ylim(ut.plot.get_axis_limits(y_values, axis_y_scaling, axis_y_limits))

    axis_y_label = ut.plot.Label.get_label('star.mass')
    subplot.set_ylabel(axis_y_label)

    ut.plot.make_axis_secondary_time(subplot, time_kind, time_limits, Cosmology)

    # colors = ut.plot.get_colors(len(gals))

    if gals is not None:
        for _gal_i, gal in enumerate(gals):
            subplot.plot(
                gal[time_kind][::snapshot_subsample_factor],
                gal[property_name][::snapshot_subsample_factor],
                linewidth=3.0,
                alpha=0.9,
                # color=colors[gal_i],
                color=ut.plot.get_color('blue.mid'),
                label='main progenitor',
            )

    if sfhs is not None:
        for _sfh_i, sfh in enumerate(sfhs):
            subplot.plot(
                sfh[time_kind],
                sfh['mass'],
                '--',
                linewidth=3.0,
                alpha=0.9,
                # color=colors[sfh_i],
                color=ut.plot.get_color('orange.mid'),
                label='SFH computed at $z=0$',
            )

    ut.plot.make_legends(subplot)

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = f'galaxy_{property_name}_v_{time_kind}'
    ut.plot.parse_output(plot_file_name, plot_directory)


# --------------------------------------------------------------------------------------------------
# disk mass and radius over time
# --------------------------------------------------------------------------------------------------
def get_galaxy_mass_profiles_v_redshift(
    directory='.',
    redshifts=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0],
    parts=None,
):
    '''
    Read snapshots and store dictionary of galaxy/halo position, velocity, size, mass at input
    scale-factors, for Shea.

    Parameters
    ----------
    directory : str : directory of snapshot files
    redshifts : array-like : redshifts at which to get properties
    parts : list : list of particle dictionaries

    Returns
    -------
    dictionary of galaxy/halo properties at each redshift
    '''
    Read = gizmo_io.ReadClass()

    species_read = ['star', 'dark']
    properties_read = ['mass', 'position', 'velocity', 'potential']

    star_distance_max = 20
    dark_distance_max = 50

    profile_species_name = 'star'
    profile_mass_percents = [50, 90]

    gal = {
        'index': [],  # snapshot index
        'redshift': [],  # snapshot redshift
        'scalefactor': [],  # snapshot scale-factor
        'time': [],  # snapshot time [Gyr]
        'time.lookback': [],  # snapshot lookback time [Gyr]
        'star.position': [],  # position of galaxy (star) center [kpc comoving]
        'star.velocity': [],  # center-of-mass velocity of stars within R_50 [km / s]
        'dark.position': [],  # position of DM center [kpc comoving]
        'dark.velocity': [],  # center-of-mass velocity of DM within 0.5 * R_200m [km / s]
        'rotation': [],  # rotation tensor of disk
        'axis.ratios': [],  # axis ratios of disk
        'profile.3d.distance': [],  # distance bins in 3-D [kpc physical]
        'profile.3d.density': [],  # density, in 3-D [M_sun / kpc ^ 3]
        'profile.major.distance': [],  # distance bins along major (R) axis [kpc physical]
        'profile.major.density': [],  # surface density, in 2-D [M_sun / kpc ^ 2]
        'profile.minor.bulge.distance': [],  # distance bins along minor (Z) axis [kpc physical]
        'profile.minor.bulge.density': [],  # density, in 1-D [M_sun / kpc]
        'profile.minor.disk.distance': [],  # distance bins along minor (Z) axis [kpc physical]
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

    for z_i, redshift in enumerate(redshifts):
        if parts is not None and len(parts) > 0:
            part = parts[z_i]
        else:
            part = Read.read_snapshots(
                species_read, 'redshift', redshift, directory, properties=properties_read
            )

        for k in ['index', 'redshift', 'scalefactor', 'time', 'time.lookback']:
            gal[k].append(part.snapshot[k])

        # get position and velocity
        gal['star.position'].append(part.host['position'][0])
        gal['star.velocity'].append(part.host['velocity'][0])

        gal['dark.position'].append(
            ut.particle.get_center_positions(part, 'dark', weight_property='potential')
        )
        gal['dark.velocity'].append(
            ut.particle.get_center_velocities(part, 'dark', distance_max=dark_distance_max)
        )

        # get radius_90 as fiducial
        gal_90 = ut.particle.get_galaxy_properties(
            part, profile_species_name, 'mass.percent', mass_percent, distance_max=star_distance_max
        )

        principal_axes = ut.particle.get_principal_axes(
            part, profile_species_name, gal_90['radius']
        )

        gal['rotation'].append(principal_axes['rotation'])
        gal['axis.ratios'].append(principal_axes['axis.ratios'])

        for mass_percent in profile_mass_percents:
            mass_percent_name = '{:.0f}'.format(mass_percent)

            gal = ut.particle.get_galaxy_properties(
                part,
                profile_species_name,
                'mass.percent',
                mass_percent,
                distance_max=star_distance_max,
            )
            gal['radius.3d.' + mass_percent_name].append(gal['radius'])
            gal['mass.3d.' + mass_percent_name].append(gal['mass'])

            gal_minor = ut.particle.get_galaxy_properties(
                part,
                profile_species_name,
                'mass.percent',
                mass_percent,
                'minor',
                star_distance_max,
                rotation_tensor=principal_axes['rotation'],
                other_axis_distance_limits=[0, gal_90['radius']],
            )
            gal['radius.minor.' + mass_percent_name].append(gal_minor['radius'])
            gal['mass.minor.' + mass_percent_name].append(gal_minor['mass'])

            gal_major = ut.particle.get_galaxy_properties(
                part,
                profile_species_name,
                'mass.percent',
                mass_percent,
                'major',
                star_distance_max,
                rotation_tensor=principal_axes['rotation'],
                other_axis_distance_limits=[0, gal_minor['radius']],
            )
            gal['radius.major.' + mass_percent_name].append(gal_major['radius'])
            gal['mass.major.' + mass_percent_name].append(gal_major['radius'])

        pro = plot_property_v_distance(
            part,
            profile_species_name,
            'mass',
            'density',
            'log',
            False,
            None,
            [0.05, 20],
            0.1,
            'log',
            3,
            get_values=True,
        )
        for k in ['distance', 'density']:
            gal['profile.3d.' + k].append(pro[profile_species_name][k])

        pro = plot_property_v_distance(
            part,
            profile_species_name,
            'mass',
            'density',
            'log',
            False,
            None,
            [0.05, 20],
            0.1,
            'log',
            2,
            rotation=principal_axes['rotation'],
            other_axis_distance_limits=[0, 1],
            get_values=True,
        )
        for k in ['distance', 'density']:
            gal['profile.major.' + k].append(pro[profile_species_name][k])

        pro = plot_property_v_distance(
            part,
            profile_species_name,
            'mass',
            'density',
            'log',
            False,
            None,
            [0.05, 20],
            0.1,
            'log',
            1,
            rotation=principal_axes['rotation'],
            other_axis_distance_limits=[0, 0.05],
            get_values=True,
        )
        for k in ['distance', 'density']:
            gal['profile.minor.bulge.' + k].append(pro[profile_species_name][k])

        pro = plot_property_v_distance(
            part,
            profile_species_name,
            'mass',
            'density',
            'log',
            False,
            None,
            [0.05, 20],
            0.1,
            'log',
            1,
            rotation=principal_axes['rotation'],
            other_axis_distance_limits=[1, 10],
            get_values=True,
        )
        for k in ['distance', 'density']:
            gal['profile.minor.disk.' + k].append(pro[profile_species_name][k])

    for prop_name in gal:
        gal[prop_name] = np.array(gal[prop_name])

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
    print('star_velocity(x,y,z)[km/s] dark_velocity(x,y,z)[km/s] ', end='')
    print('R_50[kpc] M_star_50[Msun] M_gas_50[Msun] M_dark_50[Msun] ', end='')
    print('R_90[kpc] M_star_90[Msun] M_gas_90[Msun] M_dark_90[Msun]', end='\n')

    for z_i in range(gal['redshift'].size):
        print(
            '{:.5f} {:.5f} {:.5f} '.format(
                gal['redshift'][z_i], gal['scalefactor'][z_i], gal['time'][z_i]
            ),
            end='',
        )
        print(
            '{:.3f} {:.3f} {:.3f} '.format(
                gal['star.position'][z_i][0],
                gal['star.position'][z_i][1],
                gal['star.position'][z_i][2],
            ),
            end='',
        )
        print(
            '{:.3f} {:.3f} {:.3f} '.format(
                gal['star.velocity'][z_i][0],
                gal['star.velocity'][z_i][1],
                gal['star.velocity'][z_i][2],
            ),
            end='',
        )
        print(
            '{:.3f} {:.3f} {:.3f} '.format(
                gal['dark.velocity'][z_i][0],
                gal['dark.velocity'][z_i][1],
                gal['dark.velocity'][z_i][2],
            ),
            end='',
        )
        print(
            '{:.3e} {:.3e} {:.3e} {:.3e} '.format(
                gal['radius.50'][z_i],
                gal['star.mass.50'][z_i],
                gal['gas.mass.50'][z_i],
                gal['dark.mass.50'][z_i],
            ),
            end='',
        )
        print(
            '{:.3e} {:.3e} {:.3e} {:.3e}'.format(
                gal['radius.90'][z_i],
                gal['star.mass.90'][z_i],
                gal['gas.mass.90'][z_i],
                gal['dark.mass.90'][z_i],
            ),
            end='\n',
        )


# --------------------------------------------------------------------------------------------------
# compare simulations
# --------------------------------------------------------------------------------------------------
class CompareSimulationsClass(ut.io.SayClass):
    '''
    Analyze and plot different simulations for comparison.
    '''

    def __init__(
        self,
        galaxy_radius_limits=[0, 15],
        galaxy_profile_radius_limits=[0.1, 30],
        halo_profile_radius_limits=[0.5, 300],
        plot_directory='plot',
    ):
        '''
        Set directories and names of simulations to read.
        '''
        self.Read = gizmo_io.ReadClass()

        self.properties = ['mass', 'position', 'form.scalefactor', 'massfraction']

        self.galaxy_radius_limits = galaxy_radius_limits
        self.galaxy_profile_radius_limits = galaxy_profile_radius_limits
        self.halo_profile_radius_limits = halo_profile_radius_limits

        self.plot_directory = ut.io.get_path(plot_directory)

        self.simulation_names = []

    def _parse_inputs(self, parts=None, species=None, redshifts=None):
        '''
        parts : list : dictionaries of particles at snapshot
        species : str or list : name[s] of particle species to read and analyze
        redshifts : float or list
        '''
        if parts is not None and isinstance(parts, dict):
            parts = [parts]

        if species is not None and np.isscalar(species):
            species = [species]

        if redshifts is None:
            redshifts = parts[0].snapshot['redshift']
        if np.isscalar(redshifts):
            redshifts = [redshifts]

        if parts is not None and redshifts is not None and len(redshifts) > 1:
            self.say('! input particles at single snapshot but also input more than one redshift')
            self.say(
                '  analyzing just snapshot redshift = {:.3f}'.format(parts[0].snapshot['redshift'])
            )
            redshifts = [parts[0].snapshot['redshift']]

        return parts, species, redshifts

    def plot(
        self,
        parts=None,
        species=['star', 'gas', 'dark'],
        simulation_directories=None,
        redshifts=None,
        galaxy_radius_limits=None,
        plot_properties_v_distance=True,
        plot_abundances=True,
        plot_properties_v_properties=True,
        plot_histories=True,
        plot_images=True,
    ):
        '''
        Analyze and plot all quantities for all simulations at each redshift.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        species : str or list : name[s] of particle species to read and analyze
        simulation_directories : list : simulation directories and names/labels for figure
        redshifts : float or list
        '''
        parts, species, redshifts = self._parse_inputs(parts, species, redshifts)

        for redshift in redshifts:
            if len(redshifts) > 1 or parts is None:
                parts = self.Read.read_snapshots_simulations(
                    species,
                    'redshift',
                    redshift,
                    simulation_directories,
                    properties=self.properties,
                )

            if 'star' in species:
                self.print_masses_sizes(parts, ['star'])
            if plot_properties_v_distance:
                self.plot_properties_v_distance(parts, plot_abundances=plot_abundances)
            if plot_abundances and plot_properties_v_properties:
                self.plot_properties_v_properties(parts)
            if plot_histories:
                self.plot_histories(parts, galaxy_radius_limits)
            if plot_images:
                self.plot_images(parts)

    def print_masses_sizes(self, parts, species=['star'], distance_max=20, mass_fraction=90):
        '''
        Print masses and sizes of simulations / galaxies.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        species : str or list : name[s] of particle species to read and analyze
        distance_max : float : maximum distance from center to plot
        mass_fraction : float : mass fraction (within distance_max) to determine edge of galaxy
        '''
        if species is not None and np.isscalar(species):
            species = [species]

        gals = []
        for spec_name in ut.array.get_list_combined(species, parts[0], 'intersect'):
            for part in parts:
                gal = ut.particle.get_galaxy_properties(
                    part, spec_name, 'mass.percent', mass_fraction, 'both', distance_max
                )
                gals.append(gal)

            self.say(f'\n# species = {spec_name}')

            for part_i, part in enumerate(parts):
                gal = gals[part_i]
                self.say('\n{}'.format(part.info['simulation.name']))

                self.say(
                    '* M_{},sim = {} Msun, log = {:.2f}'.format(
                        spec_name,
                        ut.io.get_string_from_numbers(part[spec_name]['mass'].sum(), 2, True),
                        np.log10(part[spec_name]['mass'].sum()),
                    )
                )

                pindices = ut.array.get_indices(
                    part[spec_name].prop('host.distance.total'), [0, distance_max]
                )
                self.say(
                    '* M_{}(< {:.0f} kpc) = {} Msun, log = {:.2f}'.format(
                        spec_name,
                        distance_max,
                        ut.io.get_string_from_numbers(
                            part[spec_name]['mass'][pindices].sum(), 2, True
                        ),
                        np.log10(part[spec_name]['mass'][pindices].sum()),
                    )
                )

                distance_min = 50
                pindices = ut.array.get_indices(
                    part[spec_name].prop('host.distance.total'), [distance_min, Inf]
                )
                self.say(
                    '* M_{}(> {:.0f} kpc) = {} Msun, log = {:.2f}'.format(
                        spec_name,
                        distance_min,
                        ut.io.get_string_from_numbers(
                            part[spec_name]['mass'][pindices].sum(), 2, True
                        ),
                        np.log10(part[spec_name]['mass'][pindices].sum()),
                    )
                )

                self.say(
                    '* M_{},{} = {} Msun, log = {:.2f}'.format(
                        spec_name,
                        mass_fraction,
                        ut.io.get_string_from_numbers(gal['mass'], 2, True),
                        np.log10(gal['mass']),
                    )
                )
                self.say(
                    '* R_{},{} major, minor = {:.1f}, {:.1f} kpc'.format(
                        spec_name, mass_fraction, gal['radius.major'], gal['radius.minor']
                    )
                )
        print()

    def plot_properties_v_distance(self, parts, distance_bin_width=0.1, plot_abundances=True):
        '''
        Plot profiles of various properties, comparing all simulations at each redshift.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        distance_bin_width : float : width of distance bin
        plot_abundances : bool : whether to plot elemental abundances
        '''
        if 'dark' in parts[0] and 'gas' in parts[0] and 'star' in parts[0]:
            plot_property_v_distance(
                parts,
                'total',
                'mass',
                'vel.circ',
                'linear',
                False,
                [0, None],
                [0.1, self.halo_profile_radius_limits[1]],
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            plot_property_v_distance(
                parts,
                'total',
                'mass',
                'sum.cum',
                'log',
                False,
                [None, None],
                self.halo_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            plot_property_v_distance(
                parts,
                'baryon',
                'mass',
                'sum.cum.fraction',
                'linear',
                False,
                [0, 2],
                [10, 2000],
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

        spec_name = 'dark'
        if spec_name in parts[0]:
            plot_property_v_distance(
                parts,
                spec_name,
                'mass',
                'sum.cum',
                'log',
                False,
                [None, None],
                self.halo_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            plot_property_v_distance(
                parts,
                spec_name,
                'mass',
                'density',
                'log',
                False,
                [None, None],
                self.halo_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

        spec_name = 'gas'
        if spec_name in parts[0]:
            plot_property_v_distance(
                parts,
                spec_name,
                'mass',
                'sum.cum',
                'log',
                False,
                [None, None],
                self.halo_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            if 'massfraction' in parts[0][spec_name] and plot_abundances:
                try:
                    plot_property_v_distance(
                        parts,
                        spec_name,
                        'metallicity.total',
                        'median',
                        'linear',
                        True,
                        [None, None],
                        self.halo_profile_radius_limits,
                        distance_bin_width,
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )

                    plot_property_distribution(
                        parts,
                        spec_name,
                        'metallicity.total',
                        [-5, 1.3],
                        0.1,
                        None,
                        'linear',
                        'probability',
                        self.halo_profile_radius_limits,
                        axis_y_limits=[1e-4, None],
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )
                except Exception:
                    pass

            # if 'velocity' in parts[0][prop]:
            #    plot_property_v_distance(
            #        parts, spec, 'host.velocity.rad', 'average', 'linear', True,
            #        [None, None], self.halo_profile_radius_limits, 0.25,
            #        plot_file_name=True, plot_directory=self.plot_directory,
            #    )

        spec_name = 'star'
        if spec_name in parts[0]:
            plot_property_v_distance(
                parts,
                spec_name,
                'mass',
                'sum.cum',
                'log',
                False,
                [None, None],
                self.halo_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            plot_property_v_distance(
                parts,
                spec_name,
                'mass',
                'density',
                'log',
                False,
                [None, None],
                self.galaxy_profile_radius_limits,
                distance_bin_width,
                plot_file_name=True,
                plot_directory=self.plot_directory,
            )

            if 'massfraction' in parts[0][spec_name] and plot_abundances:
                try:
                    plot_property_v_distance(
                        parts,
                        spec_name,
                        'metallicity.fe',
                        'median',
                        'linear',
                        True,
                        [None, None],
                        self.galaxy_profile_radius_limits,
                        distance_bin_width,
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )

                    plot_property_distribution(
                        parts,
                        spec_name,
                        'metallicity.fe',
                        [-5, 1.3],
                        0.1,
                        None,
                        'linear',
                        'probability',
                        self.galaxy_radius_limits,
                        axis_y_limits=[1e-4, None],
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )
                except Exception:
                    pass

                try:
                    plot_property_v_distance(
                        parts,
                        spec_name,
                        'metallicity.mg - metallicity.fe',
                        'median',
                        'linear',
                        True,
                        [None, None],
                        self.galaxy_profile_radius_limits,
                        distance_bin_width,
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )

                    plot_property_distribution(
                        parts,
                        spec_name,
                        'metallicity.mg - metallicity.fe',
                        [-1.7, 0.6],
                        0.1,
                        None,
                        'linear',
                        'probability',
                        self.galaxy_radius_limits,
                        axis_y_limits=[1e-4, None],
                        plot_file_name=True,
                        plot_directory=self.plot_directory,
                    )
                except Exception:
                    pass

            if 'form.scalefactor' in parts[0][spec_name] and parts[0].snapshot['redshift'] <= 5:
                plot_property_v_distance(
                    parts,
                    spec_name,
                    'age',
                    'average',
                    'linear',
                    True,
                    [None, None],
                    self.galaxy_radius_limits,
                    distance_bin_width * 2,
                    'linear',
                    plot_file_name=True,
                    plot_directory=self.plot_directory,
                )

    def plot_histories(self, parts, galaxy_radius_limits=[0, 15], plot_directory=None):
        '''
        Plot histories of star formation and mass.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        '''
        if galaxy_radius_limits is None or len(galaxy_radius_limits) == 0:
            galaxy_radius_limits = self.galaxy_radius_limits

        if plot_directory is None or len(plot_directory) == 0:
            plot_directory = self.plot_directory

        StarFormHistory = StarFormHistoryClass()

        if 'star' in parts[0]:
            StarFormHistory.plot_star_form_history(
                parts,
                'mass',
                'redshift',
                [None, 6],
                0.1,
                'linear',
                galaxy_radius_limits,
                sfh_limits=[None, None],
                plot_file_name=True,
                plot_directory=plot_directory,
            )

            StarFormHistory.plot_star_form_history(
                parts,
                'mass',
                'redshift',
                [3, 10],
                0.1,
                'linear',
                galaxy_radius_limits,
                sfh_limits=[None, None],
                plot_file_name=True,
                plot_directory=plot_directory,
            )

            StarFormHistory.plot_star_form_history(
                parts,
                'form.rate',
                'time.lookback',
                [None, 13],
                0.5,
                'linear',
                galaxy_radius_limits,
                sfh_limits=[0, None],
                sfh_scaling='linear',
                plot_file_name=True,
                plot_directory=plot_directory,
            )

            # StarFormHistory.plot_star_form_history(
            #    parts,
            #    'form.rate',
            #    'redshift',
            #    [3, 10],
            #    0.5,
            #    'linear',
            #    galaxy_radius_limits,
            #    sfh_limits=[0, None],
            #    sfh_scaling='linear',
            #    plot_file_name=True,
            #    plot_directory=plot_directory,
            # )

            # StarFormHistory.plot_star_form_history(
            #    parts,
            #    'form.rate.specific',
            #    'time.lookback',
            #    [None, 13],
            #    0.4,
            #    'linear',
            #    galaxy_radius_limits,
            #    sfh_limits=[None, None],
            #    plot_file_name=True,
            #    plot_directory=plot_directory,
            # )

    def plot_properties_v_properties(self, parts, property_bin_number=100):
        '''
        Plot property v property for each simulation.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        property_bin_number : int : number of bins along each dimension for histogram
        '''
        plot_directory = self.plot_directory + 'property_2d'

        for part in parts:
            species_name = 'star'
            if species_name in part:
                if 'massfraction' in parts[0][species_name]:
                    try:
                        plot_property_v_property(
                            part,
                            species_name,
                            'metallicity.fe',
                            [-3, 1],
                            'linear',
                            'metallicity.mg - metallicity.fe',
                            [-0.5, 0.55],
                            'linear',
                            property_bin_number,
                            host_distance_limits=self.galaxy_radius_limits,
                            draw_statistics=False,
                            plot_file_name=True,
                            plot_directory=plot_directory,
                            add_simulation_name=True,
                        )

                        plot_property_v_property(
                            part,
                            species_name,
                            'age',
                            [0, 13.5],
                            'linear',
                            'metallicity.fe',
                            [-3, 1],
                            'linear',
                            property_bin_number,
                            host_distance_limits=self.galaxy_radius_limits,
                            draw_statistics=True,
                            plot_file_name=True,
                            plot_directory=plot_directory,
                            add_simulation_name=True,
                        )

                        plot_property_v_property(
                            part,
                            species_name,
                            'age',
                            [0, 13.5],
                            'linear',
                            'metallicity.mg - metallicity.fe',
                            [-0.5, 0.55],
                            'linear',
                            property_bin_number,
                            host_distance_limits=self.galaxy_radius_limits,
                            draw_statistics=True,
                            plot_file_name=True,
                            plot_directory=plot_directory,
                            add_simulation_name=True,
                        )
                    except Exception:
                        pass

            species_name = 'gas'
            # if species_name in part:
            #    plot_property_v_property(
            #        part, species_name,
            #        'number.density', [-4, 4], 'log',
            #        'temperature', [10, 1e7], 'log',
            #        property_bin_number, host_distance_limits=self.galaxy_radius_limits,
            #        draw_statistics=False,
            #        plot_file_name=True, plot_directory=plot_directory, add_simulation_name=True,
            #    )

    def plot_images(
        self, parts, distance_max=21, distance_bin_width=0.05, align_principal_axes=True
    ):
        '''
        Plot images of each simulation.

        Parameters
        ----------
        parts : list : dictionaries of particles at snapshot
        distance_max : float : maximum distance from center to plot
        distance_bin_width : float : distance bin width (pixel size)
        align_principal_axes : bool : whether to align plot axes with principal axes
        '''
        plot_directory = self.plot_directory + 'image'

        for part in parts:
            species_name = 'star'
            if species_name in part:
                Image.plot_image(
                    part,
                    species_name,
                    'mass',
                    'histogram',
                    [0, 1, 2],
                    [0, 1, 2],
                    distance_max,
                    distance_bin_width,
                    rotation=align_principal_axes,
                    image_limits=[10 ** 6, 10 ** 10.5],
                    background_color='black',
                    plot_file_name=True,
                    plot_directory=plot_directory,
                )

            species_name = 'gas'
            if species_name in part:
                Image.plot_image(
                    part,
                    species_name,
                    'mass',
                    'histogram',
                    [0, 1, 2],
                    [0, 1, 2],
                    distance_max,
                    distance_bin_width,
                    rotation=align_principal_axes,
                    image_limits=[10 ** 4, 10 ** 9],
                    background_color='black',
                    plot_file_name=True,
                    plot_directory=plot_directory,
                )

            species_name = 'dark'
            if species_name in part:
                Image.plot_image(
                    part,
                    species_name,
                    'mass',
                    'histogram',
                    [0, 1, 2],
                    [0, 1, 2],
                    distance_max,
                    distance_bin_width,
                    rotation=align_principal_axes,
                    image_limits=[10 ** 5.5, 10 ** 9],
                    background_color='black',
                    plot_file_name=True,
                    plot_directory=plot_directory,
                )


CompareSimulations = CompareSimulationsClass()


def compare_star_formation_models(
    parts, density_limits=[1, 1e6], density_bin_width=0.2, distance_limits=[0, 12]):
    '''
    .
    '''
    # get star-forming gas
    part_indicess = []
    for part in parts:
        part_indicess.append(ut.array.get_indices(part['gas']['sfr'], [1e-10, 1e10]))

    # plot density of star-forming gas
    plot_file_name = 'gas.sf.density_distribution_z.{:.1f}'.format(parts[0].snapshot['redshift'])
    plot_property_distribution(
        parts,
        'gas',
        'number.density',
        density_limits,
        density_bin_width,
        None,
        weight_property_name='',
        part_indicess=part_indicess,
        distance_limits=None,
        plot_file_name=plot_file_name,
    )

    # plot density of star-forming gas weighted by SFR
    plot_file_name = 'gas.sf.density*sfr_distribution_z.{:.1f}'.format(
        parts[0].snapshot['redshift']
    )
    plot_property_distribution(
        parts,
        'gas',
        'number.density',
        density_limits,
        density_bin_width,
        None,
        weight_property_name='sfr',
        part_indicess=part_indicess,
        distance_limits=None,
        plot_file_name=plot_file_name,
    )

    # plot density of all gas
    plot_file_name = 'gas.density_distribution_z.{:.1f}'.format(parts[0].snapshot['redshift'])
    plot_property_distribution(
        parts,
        'gas',
        'number.density',
        density_limits,
        density_bin_width,
        None,
        distance_limits=distance_limits,
        plot_file_name=plot_file_name,
    )


def compare_resolution(
    parts=None,
    simulation_names=[],
    redshifts=[0],
    distance_limits=[0.01, 20],
    distance_bin_width=0.1,
):
    '''
    .
    '''
    if not simulation_names:
        simulation_names = []

    if np.isscalar(redshifts):
        redshifts = [redshifts]

    if parts is None:
        parts = []
        for simulation_directory, simulation_name in simulation_names:
            for redshift in redshifts:
                assign_hosts = True
                if 'res880' in simulation_directory:
                    assign_hosts = False
                Read = gizmo_io.ReadClass()
                part = Read.read_snapshots(
                    'dark',
                    'redshift',
                    redshift,
                    simulation_directory,
                    simulation_name=simulation_name,
                    properties=['position', 'mass'],
                    assign_hosts=assign_hosts,
                )
                if len(redshifts) > 1:
                    part.info['simulation.name'] += ' z={:.1f}'.format(redshift)

                parts.append(part)

    plot_property_v_distance(
        parts,
        'dark',
        'mass',
        'vel.circ',
        'log',
        False,
        [None, None],
        distance_limits,
        distance_bin_width,
        plot_file_name=True,
    )

    plot_property_v_distance(
        parts,
        'dark',
        'mass',
        'density',
        'log',
        False,
        [None, None],
        distance_limits,
        distance_bin_width,
        plot_file_name=True,
    )

    plot_property_v_distance(
        parts,
        'dark',
        'mass',
        'density*r',
        'log',
        False,
        [None, None],
        distance_limits,
        distance_bin_width,
        plot_file_name=True,
    )

    return parts
