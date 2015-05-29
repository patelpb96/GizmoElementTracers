'''
Analysis of Gizmo simulations.

Masses in {M_sun}, positions in {kpc comoving}, distances & radii in {kpc physical}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# local ----
from utilities import utility as ut
from utilities import constants as const
from utilities import halo_property
from utilities import plot


#===================================================================================================
# utility
#===================================================================================================
def get_species_positions_masses(part, species):
    '''
    Parameters
    ----------
    catalog of particles: dict
    list of species: string or list
    '''
    Say = ut.io.SayClass(get_species_positions_masses)

    if np.isscalar(species):
        species = [species]

    if len(species) == 1:
        positions = part[species[0]]['position']
        if np.unique(part[species[0]]['mass']).size == 1:
            masses = np.unique(part[species[0]]['mass'])[0]
        else:
            masses = part[species[0]]['mass']
    else:
        particle_num = 0
        for spec in species:
            if spec in part:
                particle_num += part[spec]['position'].shape[0]

        if particle_num > 1e8:
            Say.say('! warning: allocating positions & masses for %d particles!' % particle_num)

        positions = np.zeros([particle_num, part[spec]['position'].shape[1]],
                             dtype=part[spec]['position'].dtype)
        masses = np.zeros(particle_num, dtype=part[spec]['mass'].dtype)

        particle_index = 0
        for spec in species:
            if spec in part:
                spec_part_num = part[spec]['position'].shape[0]
                positions[particle_index: spec_part_num + particle_index] = part[spec]['position']
                masses[particle_index: spec_part_num + particle_index] = part[spec]['mass']
                particle_index += spec_part_num

    return positions, masses


def get_center_position(
    part, species=['star', 'dark', 'gas'], center_position=[0, 0, 0], radius_max=1e10):
    '''
    Get position of center of mass, using iterative zoom-in on species.

    Parameters
    ----------
    part : dict : dictionary of particles
    species : string or list: names of species to use
        note: 'all' = use all in particle dictionary
    center_pos : list/array : initial center position
    radius_max : float : maximum initial radius to consider during iteration {kpc comoving}
    '''
    if np.isscalar(species):
        species = [species]  # ensure is list

    if species == ['all']:
        species = ['star', 'dark', 'gas']

    positions, masses = get_species_positions_masses(part, species)

    #periodic_len = part.info['box.length']
    periodic_len = None  # assume zoom-in run far from box edge, for speed

    return ut.coord.position_center_of_mass_zoom(
        positions, masses, periodic_len, center_position, radius_max)


def get_halo_radius(
    part, species=['dark', 'star', 'gas'], center_position=[], virial_kind='200m',
    radius_scaling='log', radius_lim=[10, 500], radius_bin_num=100):
    '''
    Parameters
    ----------
    part : dict : catalog of particles
    species : string or list : name[s] of species to use
        note: 'all' = use all in particle dictionary
    center_position : list/array : center position to use
        note: if input none, will find it using the input species list
    virial_kind : string : virial overdensity definition
      '180m' -> average density is 180 x matter
      '200c' -> average density is 200 x critical
      'vir' -> average density is Bryan & Norman
      'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
      'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
    radius_scaling : string : radius bin scaling
        options: log, lin
    radius_lim : list/array : limits for radius bins {kpc comoving}
    radius_bin_num : int : number of radius bins

    Returns
    -------
    halo virial radius {kpc comoving}: float
    '''
    Say = ut.io.SayClass(get_halo_radius)

    HaloProperty = halo_property.HaloPropertyClass(part.Cosmo, part.snap['redshift'])

    DistanceBin = ut.bin.DistanceBinClass(
        radius_scaling, radius_lim, radius_bin_num, dimension_num=3)

    overdensity, reference_density = HaloProperty.overdensity(virial_kind)
    virial_density = overdensity * reference_density

    if np.isscalar(species):
        species = [species]  # ensure is list

    if species == ['all']:
        species = part.keys()
        # species = ['star', 'dark', 'gas']

    positions, masses = get_species_positions_masses(part, species)

    #periodic_len = part.info['box.length']
    periodic_len = None  # assume zoom-in run far from box edge, for speed

    rads = ut.coord.distance('scalar', positions, center_position, periodic_len)  # {kpc comoving}
    #rads *= part.snap['scale.factor']  # {kpc physical}

    # get masses in bins
    if 'log' in radius_scaling:
        rads = log10(rads)
        radius_lim = log10(radius_lim)

    if np.isscalar(masses):
        mass_in_bins = np.histogram(rads, radius_bin_num, radius_lim, False, None)[0]
    else:
        mass_in_bins = np.histogram(rads, radius_bin_num, radius_lim, False, masses)[0]

    # get mass within distance minimum, for computing cumulative values
    rad_indices = np.where(rads < np.min(radius_lim))[0]
    if np.isscalar(masses):
        masses_cum = masses * (rad_indices.size + np.cumsum(mass_in_bins))
    else:
        masses_cum = np.sum(masses[rad_indices]) + np.cumsum(mass_in_bins)

    if part.info['has.baryons'] and species == ['dark']:
        # correct for baryonic mass if analyzing only dark matter in baryonic simulation
        mass_factor = 1 + part.Cosmo['omega_baryon'] / part.Cosmo['omega_matter']
        masses_cum *= mass_factor

    # cumulative densities in bins
    density_cum_in_bins = masses_cum / DistanceBin.volumes_cum

    # import ipdb; ipdb.set_trace()
    # mass_in_bins[1e99]

    for dist_bin_i in xrange(DistanceBin.num - 1):
        if (density_cum_in_bins[dist_bin_i] >= virial_density and
                density_cum_in_bins[dist_bin_i + 1] < virial_density):
            log_den_inner = log10(density_cum_in_bins[dist_bin_i])
            log_den_outer = log10(density_cum_in_bins[dist_bin_i + 1])
            # interpolate in log space
            log_rad_inner = DistanceBin.log_maxs[dist_bin_i]
            log_rad_outer = DistanceBin.log_maxs[dist_bin_i + 1]
            log_slope = (log_rad_outer - log_rad_inner) / (log_den_inner - log_den_outer)

            halo_radius = 10 ** (log_rad_inner + log_slope *
                                 (log_den_inner - log10(virial_density)))

            if 'log' in radius_scaling:
                rad_use = log10(halo_radius)
            else:
                rad_use = halo_radius

            if np.isscalar(masses):
                halo_mass = masses * np.sum(rads < rad_use)
            else:
                halo_mass = np.sum(masses[rads < rad_use])

            Say.say('M_%s = %.3e M_sun, log = %.3f\n  R_%s = %.3f kpc comoving' %
                    (virial_kind, halo_mass, log10(halo_mass), virial_kind, halo_radius))

            return halo_radius
    else:
        Say.say('! could not find virial radius - might need to increase radius limits')


#===================================================================================================
# initial conditions
#===================================================================================================
def write_initial_condition_points(
    part_fin, part_ini, center_pos=None, distance_select=None, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m',
    use_onorbe_method=False, refinement_num=1, method='particles'):
    '''
    Print positions of initial conditions of dark-matter particles selected at z = 0.
    Use rules of thumb from Onorbe et al.

    Parameters
    ----------
    catalog of particles at final time: dict
    catalog of particles at initial time: dict
    center position at final time: list
    distance from center to select particles at final time {kpc comoving, or units of R_halo}: float
        if None, use halo radius
    whether to scale distance to halo radius: boolean
    virial kind for halo radius: string
    whether to use method of Onorbe et al to make selection region uncontaminated: boolean
    if above is true, number of refinement levels beyond current level for region: int
    method to identify initial zoom-in regon: string
        options: particles, convex.hull, cube
    '''
    file_name = 'ic_agora_m12i_points.txt'

    Say = ut.io.SayClass(write_initial_condition_points)

    assert method in ['particles', 'convex.hull', 'cube']

    # sanity check
    spec_names = []
    for spec_name in ['dark', 'dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6']:
        if spec_name in part_fin:
            spec_names.append(spec_name)
            if np.min(part_fin[spec_name]['id'] == part_ini[spec_name]['id']) == False:
                Say.say('! species = %s: ids in final and initial catalogs not match' % spec_name)
                if spec_name in ['dark', 'dark.2']:
                    return
    Say.say('using species: %s' % spec_names)

    if center_pos is None or not len(center_pos):
        center_pos = get_center_position(part_fin, ['all'])

    distance_select_input = distance_select
    if not distance_select or scale_to_halo_radius:
        if not halo_radius:
            halo_radius = get_halo_radius(part_fin, 'all', center_pos, virial_kind, 'log')
        if not distance_select:
            distance_select = halo_radius
        elif scale_to_halo_radius:
            distance_select *= halo_radius

    if use_onorbe_method:
        # convert distance_max according to Onorbe et al
        distance_pure = distance_select
        if method == 'cube':
            distance_select = (1.5 * refinement_num + 1) * distance_pure
        elif method in ['particles', 'convex.hull']:
            distance_select = (1.5 * refinement_num + 7) * distance_pure
    else:
        distance_pure = None

    mass_pure = 0
    mass_select = 0
    poss_ini = []
    spec_select_num = []
    for spec_name in spec_names:
        poss_fin = part_fin[spec_name]['position']
        dists = ut.coord.distance('scalar', poss_fin, center_pos, part_fin.info['box.length'])
        pure_indices = ut.array.elements(dists, [0, distance_pure])
        select_indices = ut.array.elements(dists, [0, distance_select])

        poss_ini.extend(part_ini[spec_name]['position'][select_indices])

        mass_pure += part_ini[spec_name]['mass'][pure_indices].sum()
        mass_select += part_ini[spec_name]['mass'][select_indices].sum()
        spec_select_num.append(select_indices.size)

    poss_ini = np.array(poss_ini)
    poss_ini_limits = [[poss_ini[:, dimen_i].min(), poss_ini[:, dimen_i].max()] for dimen_i in
                       xrange(poss_ini.shape[1])]

    volume_ini = ut.coord.volume_convex_hull(poss_ini)
    density_ini = part_ini.Cosmo.density_matter(part_ini.snap['redshift'])
    if part_ini.info['has.baryons']:
        # subtract baryonic mass
        density_ini *= part_ini.Cosmo['omega_dark'] / part_ini.Cosmo['omega_matter']
    mass_ini = volume_ini * density_ini  # assume cosmic density within volume

    Say.say('final redshift = %.3f, initial redshift = %.3f' %
            (part_fin.snap['redshift'], part_ini.snap['redshift']))
    Say.say('centering on volume at final time = [%.3f, %.3f, %.3f] kpc comoving' %
            (center_pos[0], center_pos[1], center_pos[2]))
    if scale_to_halo_radius:
        Say.say('selecting radius as %.2f x R_%s, R_%s = %.2f kpc comoving' %
                (distance_select_input, virial_kind, virial_kind, halo_radius))
    Say.say('radius of selection volume at final time = %.3f kpc comoving' % distance_select)
    if use_onorbe_method:
        Say.say('radius of uncontaminated volume (Onorbe et al) at final time = %.3f kpc comoving' %
                distance_pure)
    Say.say('number of particles in selection volume at final time = %d' % np.sum(spec_select_num))
    for spec_i in xrange(len(spec_names)):
        spec_name = spec_names[spec_i]
        Say.say('  species %s: number = %d' % (spec_name, spec_select_num[spec_i]))
    Say.say('mass of all dark-matter particles:')
    Say.say('  at highest-resolution in input catalog = %.2e M_sun' %
            part_ini['dark']['mass'].sum())
    Say.say('  in selection volume at final time = %.2e M_sun' % mass_select)
    if use_onorbe_method:
        Say.say('  in uncontaminated volume (Onorbe et al) at final time = %.2e M_sun' % mass_pure)
    Say.say('  in convex hull at initial time = %.2e M_sun' % mass_ini)
    Say.say('volume of convex hull at initial time = %.1f Mpc ^ 3 comoving' %
            (volume_ini * const.mega_per_kilo ** 3))

    # MUSIC does not support header information in points file, so put in separate log file
    log_file_name = file_name.replace('.txt', '_log.txt')
    file_io = open(log_file_name, 'w')
    file_io.write('# final redshift = %.3f, initial redshift = %.3f\n' %
                  (part_fin.snap['redshift'], part_ini.snap['redshift']))
    file_io.write('# centering on volume at final time = [%.3f, %.3f, %.3f] kpc comoving\n' %
                  (center_pos[0], center_pos[1], center_pos[2]))
    if scale_to_halo_radius:
        file_io.write('# selecting radius as %.2f x R_%s, R_%s = %.2f kpc comoving\n' %
                      (distance_select_input, virial_kind, virial_kind, halo_radius))
    file_io.write('# radius of selection volume at final time = %.3f kpc comoving\n' %
                  distance_select)
    if use_onorbe_method:
        file_io.write(
            '# radius of uncontaminated volume (Onorbe et al) at final time = %.3f kpc comoving\n' %
            distance_pure)
    file_io.write('# number of particles in selection volume at final time = %d\n' %
                  poss_ini.shape[0])
    for spec_i in xrange(len(spec_names)):
        file_io.write('#   species %s: number = %d\n' %
                      (spec_names[spec_i], spec_select_num[spec_i]))
    file_io.write('# mass of all dark-matter particles:\n')
    file_io.write('#   at highest-resolution in input catalog = %.2e M_sun\n' %
                  part_ini['dark']['mass'].sum())
    file_io.write('#   in selection volume at final time = %.2e M_sun\n' % mass_select)
    if use_onorbe_method:
        file_io.write('#   in uncontaminated volume (Onorbe et al) at final time = %.2e M_sun\n' %
                      mass_pure)
    file_io.write('#   in convex hull at initial time = %.2e M_sun\n' % mass_ini)
    file_io.write('# volume of convex hull at initial time = %.1f Mpc ^ 3 comoving\n' %
                  (volume_ini * const.mega_per_kilo ** 3))
    for dimen_i in xrange(poss_ini.shape[1]):
        file_io.write('# initial position-%s [min, max] = %s kpc comoving, %s box units\n' %
                      (dimen_i, ut.array.get_limits(poss_ini_limits[dimen_i], digit_num=3),
                       ut.array.get_limits(poss_ini_limits[dimen_i] / part_ini.info['box.length'],
                                           digit_num=8)))

    poss_ini /= part_ini.info['box.length']  # renormalize to box units

    if method == 'convex.hull':
        # use convex hull to define initial region to reduce memory
        ConvexHull = spatial.ConvexHull(poss_ini)
        poss_ini = poss_ini[ConvexHull.vertices]
        file_io.write('# using convex hull with %d vertices to define initial volume\n' %
                      poss_ini.shape[0])

    file_io.close()

    file_io = open(file_name, 'w')
    for pi in xrange(poss_ini.shape[0]):
        file_io.write('%.8f %.8f %.8f\n' % (poss_ini[pi, 0], poss_ini[pi, 1], poss_ini[pi, 2]))
    file_io.close()


#===================================================================================================
# tests
#===================================================================================================
def plot_mass_contamination(
    part, center_pos=[], distance_lim=[1, 2000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', halo_radius=None, scale_to_halo_radius=False,
    write_plot=False, plot_directory='.'):
    '''
    Plot lower resolution particle contamination v distance from input center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
    distance limits: list or array
    distance scaling: float
        options: log, lin
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

    x_lim = np.array(distance_lim)
    if halo_radius and scale_to_halo_radius:
        x_lim *= halo_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    pros = {species_ref: {}}
    for spec in species_test:
        pros[spec] = {}

    ratios = {}

    #periodic_len = part.info['box.length']
    periodic_len = None

    for spec in pros:
        dists = ut.coord.distance('scalar', part[spec]['position'], center_pos, periodic_len)
        pros[spec] = DistanceBin.get_mass_profile(dists, part[spec]['mass'], get_spline=False)

    for spec in species_test:
        mass_ratio_bin = pros[spec]['mass'] / pros[species_ref]['mass']
        mass_ratio_cum = pros[spec]['mass.cum'] / pros[species_ref]['mass.cum']
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
    dists = pros[spec]['distance.cum']
    print_string = '  d < %.3f kpc: cumulative contamination mass = %.2e, number = %d'
    if scale_to_halo_radius:
        dists /= halo_radius
        print_string = '  d/R_halo < %.3f: mass = %.2e, number = %d'
    for dist_i in xrange(pros[spec]['mass.cum'].size):
        if pros[spec]['mass.cum'][dist_i] > 0:
            Say.say(print_string % (dists[dist_i], pros[spec]['mass.cum'][dist_i],
                                    pros[spec]['mass.cum'][dist_i] / part[spec]['mass'][0]))

    # plot ----------
    colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim([0.0001, 3])
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    subplot.set_ylabel('$M_{\\rm spec} / M_{\\rm %s}$' % species_ref, fontsize=20)
    if scale_to_halo_radius:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    # import ipdb; ipdb.set_trace()

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
        plot_name = 'mass.ratio_v_%s_z.%.1f.pdf' % (dist_name, part.snap['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def plot_metal_v_distance(
    part, center_pos=[], distance_lim=[10, 3000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', halo_radius=None, scale_to_halo_radius=False,
    plot_kind='gas.metallicity', write_plot=False, plot_directory='.'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
        note: if not input, generate
    distance limits: list or array
    distance bin number: int
    distance scaling: string
        options: log, lin
    y-axis scaling: string
        options: log, lin
    halo radius: float
    whether to scale distance to halo radius: boolean
    plot king: string
        options: gas.metallicity, gas.metal.mass.cum
    whether to write plot to file: boolean
    directory to place plot: string
    '''
    metal_index = 0  # overall metallicity

    Say = ut.io.SayClass(plot_metal_v_distance)

    # if center_pos is None or not len(center_pos):
    #    center_pos = get_center_position(part, center_species)

    x_lim = np.array(distance_lim)
    if halo_radius and scale_to_halo_radius:
        x_lim *= halo_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    dists = ut.coord.distance('scalar', part['gas']['position'], center_pos,
                              part.info['box.length'])
    metal_masses = part['gas']['metallicity'][:, metal_index] * part['gas']['mass'] / 0.02  # solar

    pro_metal = DistanceBin.get_mass_profile(dists, metal_masses, get_mass_fraction=True)
    if 'metallicity' in plot_kind:
        pro_mass = DistanceBin.get_mass_profile(dists, part['gas']['mass'])
        ys = pro_metal['mass'] / pro_mass['mass']
        y_lim = np.clip(plot.get_limits(ys), 0.0001, 10)
    elif 'metal.mass.cum' in plot_kind:
        ys = pro_metal['fraction.cum']
        y_lim = [0.001, 1]

    # plot ----------
    # colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if halo_radius and scale_to_halo_radius:
        xs /= halo_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    # subplot.set_ylim([0, 0.1])
    subplot.set_ylim(y_lim)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    if 'metallicity' in plot_kind:
        subplot.set_ylabel('$Z \, / \, Z_\odot$', fontsize=20)
    elif 'metal.mass.cum' in plot_kind:
        subplot.set_ylabel('$M_{\\rm Z}(< r) \, / \, M_{\\rm Z,tot}$', fontsize=20)

    if scale_to_halo_radius:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'

    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if halo_radius:
        if scale_to_halo_radius:
            x_ref = 1
        else:
            x_ref = halo_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    # import ipdb; ipdb.set_trace()

    plot_func(xs, ys, color='blue', alpha=0.6)

    # legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    # legend.get_frame().set_alpha(0.7)

    # plt.tight_layout(pad=0.02)

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
# analysis
#===================================================================================================
def plot_property_v_property(
    part, species='gas', prop_x='density', prop_y='energy.internal', distance_lim=[20, 300],
    write_plot=False, plot_directory='.'):
    '''
    .
    '''
    from matplotlib.colors import LogNorm

    try:
        center_pos = part['dark']['position'][np.argmin(part['dark']['potential'])]
    except:
        center_pos = get_center_position(part, 'dark')

    dists = ut.coord.distance('scalar', part[species]['position'], center_pos,
                              part.info['box.length'])

    masks = ut.array.elements(dists, distance_lim)

    props_x = log10(part[species][prop_x][masks])
    props_y = log10(part[species][prop_y][masks])

    # plot ----------
    plt.close()
    plt.minorticks_on()

    plt.hist2d(props_x, props_y, bins=200, norm=LogNorm())
    plt.colorbar()
    plt.show()

    # fig, subplot = plt.subplots(1, 1, sharex=True)
    # fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.14, hspace=0.03)

    # subplot.xscale('linear')
    # subplot.yscale('log')
    # subplot.set_xlabel(prop_x)
    # subplot.set_ylabel(prop_y)

    # subplot.loglog(props_x, props_y, '.', color='green', alpha=0.1, linewidth=0.1)
    # plot_func = plot.get_plot_function(subplot, 'log', 'log')
    # plot_func(props_x, props_y, 'o', color='green', alpha=0.5)

    # plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        if len(species) == 1:
            spec_name = species[0]
        else:
            spec_name = 'total'
        plot_name = (spec_name + '.' + prop_y + '_v_' + prop_x + '_z.%.1f.pdf' %
                     part.info['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
    else:
        plt.show(block=False)


def plot_property_distr(
    parts, species='gas', prop_name='density', prop_scaling='log', prop_lim=[], prop_bin_wid=None,
    prop_bin_num=100, prop_stat='probability',
    center_poss=[], distance_lim=[], part_labels=[],
    axis_y_scaling='log', axis_y_lim=[], write_plot=False, plot_directory='.'):
    '''
    .
    '''
    Say = ut.io.SayClass(plot_property_distr)

    if isinstance(parts, dict):
        parts = [parts]
    if np.ndim(center_poss) == 1:
        center_poss = [center_poss]

    prop_name_dict = prop_name
    if prop_name == 'density.num':
        prop_name_dict = 'density'

    Stat = ut.math.StatisticClass()

    for part_i, part in enumerate(parts):
        prop_vals = part[species][prop_name_dict]

        if distance_lim:
            dists = ut.coord.distance(
                'scalar', part[species]['position'], center_poss[part_i],
                part.info['box.length'])
            dists *= part.snap['scale.factor']  # {kpc physical}
            prop_is = ut.array.elements(dists, distance_lim)
            prop_vals = prop_vals[prop_is]

        Say.say('keeping %s %s particles' % (prop_vals.size, species))

        if prop_scaling == 'log':
            prop_vals = log10(prop_vals)
            if prop_name == 'density.num':
                # convert to {cm ^ -3 physical}
                prop_vals += log10(const.proton_per_sun) + 3 * log10(const.kpc_per_cm)

        #Say.say('%s %s range = %s' %
        #        (prop_name, prop_scaling, ut.array.get_limits(prop_vals, digit_num=2)))

        Stat.append_to_dictionary(prop_vals, prop_lim, prop_bin_wid, prop_bin_num)

        Stat.print_statistics(-1)

    #import ipdb; ipdb.set_trace()

    colors = plot.get_colors(len(parts))
    if not part_labels:
        part_labels = [None for _ in xrange(len(parts))]

    # plot ----------
    plt.close()
    plt.minorticks_on()

    fig = plt.figure()
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlabel(plot.get_label(prop_name, species, get_units=True))
    draw_log = False
    if prop_scaling == 'log':
        draw_log = True
    subplot.set_ylabel('${\\rm d}f/{\\rm d}$%s' %
                       plot.get_label(prop_name, species, get_symbol=True, get_units=False,
                                      draw_log=draw_log))
    if prop_scaling == 'log':
        prop_lim = 10 ** np.array(prop_lim)
    subplot.set_xlim(prop_lim)

    if not axis_y_lim:
        axis_y_lim = plot.get_limits(Stat.distr[prop_stat][0], axis_y_scaling, exclude_zero=True)
    subplot.set_ylim(axis_y_lim)

    plot_func = plot.get_plot_function(subplot, prop_scaling, axis_y_scaling)
    for part_i in xrange(len(parts)):
        if prop_scaling == 'log':
            vals_x = 10 ** Stat.distr['bin.mid'][part_i]
        else:
            vals_x = Stat.distr['bin.mid'][part_i]
        plot_func(vals_x, Stat.distr[prop_stat][part_i],
                  color=colors[part_i], alpha=0.5, linewidth=2, label=part_labels[part_i])

    # redshift legend
    legend_z = subplot.legend([plt.Line2D((0, 0), (0, 0), linestyle='.')],
                              ['$z=%.1f$' % parts[0].snap['redshift']],
                              loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    if part_labels[0]:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        subplot.add_artist(legend_z)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        if not np.isscalar(species):
            if len(species) == 1:
                species = species[0]
            else:
                species = 'total'
        plot_name = species + '.' + prop_name + '_distr_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def plot_mass_v_distance(
    parts, species='dark', prop='density', center_positions=[], distance_scaling='log',
    distance_lim=[1, 1000], distance_bin_num=100, axis_y_scaling='log', axis_y_lim=[],
    part_labels=[], write_plot=False, plot_directory='.'):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshot)
    species : string or list : species to compute total mass of
    prop : string : mass-related property to compute
    center_positions : list : center position for each particle catalog
    distance_scaling : string : lin or log
    distance_lim : list : min and max distance for binning
    distance_bin_num : int : number of bins between limits
    axis_y_scaling : string : scaling for y-axis, lin or log
    axis_y_lim : list : limits to impose on y-axis
    '''
    Say = ut.io.SayClass(plot_mass_v_distance)

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, distance_lim, distance_bin_num, dimension_num=3)

    if isinstance(parts, dict):
        parts = [parts]

    if np.ndim(center_positions) == 1:
        center_positions = [center_positions]

    # ensure is list even if just one species
    if np.isscalar(species):
        species = [species]

    if species == ['all']:
        species = parts[0].keys()
        # species = ['star', 'dark', 'gas']

    pros = []
    for part_i, part in enumerate(parts):
        for spec_i, spec in enumerate(species):
            # split up species to make more memory efficient
            positions, masses = get_species_positions_masses(part, spec)

            if np.isscalar(masses):
                masses = np.zeros(positions.shape[0], dtype=masses.dtype) + masses

            # {kpc comoving}
            dists = ut.coord.distance(
                'scalar', positions, center_positions[part_i], part.info['box.length'])

            dists *= part.snap['scale.factor']  # {kpc physical}

            pro = DistanceBin.get_mass_profile(dists, masses)

            if spec_i == 0:
                pros.append(pro)
            else:
                ks = [k for k in pro if 'distance' not in k]
                for k in ks:
                    if 'log' in k:
                        pros[-1][k] = log10(10 ** pros[-1][k] + 10 ** pro[k])
                    else:
                        pros[-1][k] += pro[k]

        if prop == 'vel.circ':
            pros[-1]['vel.circ'] = (pros[-1]['mass.cum'] / pros[-1]['distance.cum'] *
                                    const.grav_kpc_msun_yr)
            pros[-1]['vel.circ'] = np.sqrt(pros[-1]['vel.circ'])
            pros[-1]['vel.circ'] *= const.km_per_kpc * const.yr_per_sec

        if part_i > 0:
            print(pros[part_i][prop] / pros[0][prop])

    #import ipdb; ipdb.set_trace()

    colors = plot.get_colors(len(parts))
    if not part_labels:
        part_labels = [None for _ in xrange(len(parts))]

    # plot ----------
    plt.close()
    plt.minorticks_on()

    fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlim(distance_lim)

    if axis_y_lim:
        subplot.set_ylim(axis_y_lim)
    else:
        print(plot.get_limits(pros[0][prop]))
        subplot.set_ylim(plot.get_limits(pros[0][prop]))

    subplot.set_xlabel('radius $r$ $[\\rm kpc\,physical]$')
    subplot.set_ylabel(plot.get_label(prop, species, get_symbol=True, get_units=True))

    for pro_i, pro in enumerate(pros):
        plot_func = plot.get_plot_function(subplot, distance_scaling, axis_y_scaling)
        plot_func(pro['distance'], pro[prop], color=colors[pro_i],
                  alpha=0.5, linewidth=2, linestyle='-', label=part_labels[pro_i])

    if part_labels[0]:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        if not np.isscalar(species):
            if len(species) == 1:
                species = species[0]
            else:
                species = 'total'
        plot_name = species + '.' + prop + '_v_dist_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def get_sfr_history(part, pis=None, redshift_lim=[0, 1], scalefactor_wid=0.001, time_kind='time'):
    '''
    Get array of times and star-formation rate at each time.

    Parameters
    ----------
    part : dict : dictionary of particle species
    pis : array : list of star particle indices
    redshift_lim : list : redshift limits of times to get
    scalefactor_wid : float : width of scale factor for time binning

    Returns
    -------
    time_mids : array : times {Gyr}
    dm_dts : array : total star-formation rate at each time {M_sun / yr}
    '''
    species = 'star'

    if pis is None:
        pis = np.arange(part[species]['mass'].size, dtype=np.int32)

    pis_sort = np.argsort(part[species]['form.time'])
    star_form_aexps = part[species]['form.time'][pis_sort]  # form.time = scale factor
    star_masses = part[species]['mass'][pis_sort]
    star_masses_cum = np.cumsum(star_masses)

    if redshift_lim:
        redshift_lim = np.array(redshift_lim)
        scalefactor_lim = np.sort(1 / (1 + redshift_lim))
    else:
        scalefactor_lim = [np.min(star_form_aexps), np.max(star_form_aexps)]

    scalefactor_bins = np.arange(scalefactor_lim.min(), scalefactor_lim.max(), scalefactor_wid)
    redshift_bins = 1 / scalefactor_bins - 1
    time_bins = part.Cosmo.age(redshift_bins)
    # time_bins = part.Cosmo.age(0) - time_bins    # lookback time
    time_bins *= 1e9  # convert to {yr}

    star_mass_cum_bins = np.interp(scalefactor_bins, star_form_aexps, star_masses_cum)
    dm_dts = np.diff(star_mass_cum_bins) / np.diff(time_bins) / 0.7  # account for mass loss

    time_mids = time_bins[: time_bins.size - 1] + np.diff(time_bins)  # midpoints of bins
    time_mids /= 1e9  # convert to {Gyr}

    redshift_mids = redshift_bins[: redshift_bins.size - 1] + np.diff(redshift_bins)

    if time_kind == 'time':
        return time_mids, dm_dts
    elif time_kind == 'redshift':
        return redshift_mids, dm_dts


def plot_sfr_history(
    parts, redshift_lim=[0, 1], scalefactor_wid=0.001, time_kind='time',
    center_positions=[], distance_lim=[0, 10], part_labels=['ref12', 'ref13'],
    write_plot=False, plot_directory='.'):
    '''
    Plot star-formation rate v cosmic time.

    Parameters
    ----------
    part : dict : dictionary of particle species
    pis : array : list of star particle indices
    redshift_lim : list : redshift limits of times to get
    scalefactor_wid : float : width of scale factor for time binning
    write_plot : boolean : whether to write plot
    '''
    Say = ut.io.SayClass(plot_sfr_history)

    if isinstance(parts, dict):
        parts = [parts]

    if np.ndim(center_positions) == 1:
        center_positions = [center_positions]

    sfrs = []
    times = []
    for part_i, part in enumerate(parts):
        if center_positions:
            periodic_len = part.info['box.length']
            periodic_len = None
            # {kpc comoving}
            dists = ut.coord.distance(
                'scalar', part['star']['position'], center_positions[part_i], periodic_len)

            dists *= part.snap['scale.factor']  # {kpc physical}

            pis = ut.array.elements(dists, distance_lim)
        else:
            pis = np.arange(part['star']['form.time'].size, dtype=np.int32)

        times_t, sfrs_t = get_sfr_history(part, pis, redshift_lim, scalefactor_wid, time_kind)
        times.append(times_t)
        sfrs.append(sfrs_t)

    colors = plot.get_colors(len(parts))
    if not part_labels:
        part_labels = [None for _ in xrange(len(parts))]

    # plot ----------
    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    # subplot.xscale('linear')
    # subplot.yscale('log')
    if time_kind == 'time':
        subplot.set_xlabel('time $[{\\rm Gyr}]$')
    elif time_kind == 'redshift':
        subplot.set_xlabel('redshift')
    # pylab.ylabel(r'${\rm SFR}\ \ \dot{M}_{\ast}\ \  [{\rm M_{\odot}\,yr^{-1}}]$')
    subplot.set_ylabel('${\\rm SFR}\,[{\\rm M_{\odot}\,yr^{-1}}]$')

    for part_i in xrange(len(parts)):
        subplot.semilogy(times[part_i], sfrs[part_i], linewidth=2.0, color=colors[part_i],
                         alpha=0.5, label=part_labels[part_i])

    # redshift legend
    legend_z = subplot.legend([plt.Line2D((0, 0), (0, 0), linestyle='.')],
                              ['$z=%.1f$' % parts[0].snap['redshift']],
                              loc='lower left', prop=FontProperties(size=16))
    legend_z.get_frame().set_alpha(0.5)

    if part_labels[0]:
        # property legend
        legend_prop = subplot.legend(loc='best', prop=FontProperties(size=16))
        legend_prop.get_frame().set_alpha(0.5)
        subplot.add_artist(legend_z)

    # plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_path(plot_directory)
        plot_name = 'star.fr_v_time_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)
