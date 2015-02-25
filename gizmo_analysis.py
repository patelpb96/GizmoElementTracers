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
def get_center_position(part, species=['star', 'dark', 'gas'], center_position=[0, 0, 0],
                        radius_max=1e10):
    '''
    Get position of center of mass, using iterative zoom-in on species.

    Parameters
    ----------
    catalog of particles: dict
    names of species to use: string or list
        note: 'all' = use all in particle dictionary
    initial center position: list or array
    maximum radius to consider {kpc comoving}: float
    '''
    # ensure is list even if just one species
    if species == 'all' or species == ['all']:
        species = ['star', 'dark', 'gas']
    if np.isscalar(species):
        species = [species]

    positions = []
    masses = []
    for spec in species:
        if spec in part:
            positions.extend(part[spec]['position'])
            masses.extend(part[spec]['mass'])
    positions = np.array(positions)
    masses = np.array(masses)

    return ut.coord.position_center_of_mass_zoom(
        positions, masses, part.info['box.length'], center_position, radius_max)


def get_virial_radius(
    part, species=['dark', 'star', 'gas'], center_position=[], virial_kind='200m',
    radius_scaling='log', radius_lim=[10, 3000], radius_bin_num=100):
    '''
    Parameters
    ----------
    catalog of particles: dict
    names of species to use: string or list
        note: 'all' = use all in particle dictionary
    center position to use: list or array
        note: if input none, will find it using the input species list
    virial overdensity definition: string
      '180m' -> average density is 180 x matter
      '200c' -> average density is 200 x critical
      'vir' -> average density is Bryan & Norman
      'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
      'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
    radius bin scaling: string
        options: log, lin
    radius bin limits: list or array
    radius bin number: int
    whether simulation includes baryons: boolean
        note: use this to scale dark matter mass if using only dark matter
    '''
    Say = ut.io.SayClass(get_virial_radius)

    HaloProperty = halo_property.HaloPropertyClass(part.Cosmo, part.snap['redshift'])

    DistanceBin = ut.bin.DistanceBinClass(radius_scaling, radius_lim, radius_bin_num,
                                          dimension_num=3)

    overdensity, reference_density = HaloProperty.overdensity(virial_kind)
    virial_density = overdensity * reference_density

    # ensure is list even if just one species
    if species == 'all' or species == ['all']:
        #species = part.keys()
        species = ['star', 'dark', 'gas']
    if np.isscalar(species):
        species = [species]

    if len(species) == 1:
        positions = part[species[0]]['position']
        masses = part[species[0]]['mass']
    else:
        positions = []
        masses = []
        for spec in species:
            if spec in part:
                positions.extend(part[spec]['position'])
                masses.extend(part[spec]['mass'])
        positions = np.array(positions)
        masses = np.array(masses)

    # correct for baryonic mass
    if not part.info['has.baryons'] and species == ['dark']:
        masses *= 1 + part.Cosmo['omega_baryon'] / part.Cosmo['omega_matter']

    if center_position is None or not len(center_position):
        center_position = get_center_position(part, positions, masses)

    rads = ut.coord.distance('scalar', positions, center_position, part.info['box.length'])

    pro = DistanceBin.get_mass_profile(rads, masses)

    for dist_bin_i in reversed(xrange(DistanceBin.num)):
        if (pro['density.cum'][dist_bin_i] < virial_density and
                pro['density.cum'][dist_bin_i - 1] > virial_density):
            # use linear interpolation in log space to get virial radius
            log_den_inner = log10(pro['density.cum'][dist_bin_i - 1])
            log_den_outer = log10(pro['density.cum'][dist_bin_i])
            log_rad_inner = DistanceBin.log_mins[dist_bin_i]
            log_rad_outer = DistanceBin.log_maxs[dist_bin_i]
            log_slope = (log_rad_outer - log_rad_inner) / (log_den_inner - log_den_outer)
            virial_radius = 10 ** (log_rad_inner + log_slope *
                                   (log_den_inner - log10(virial_density)))
            virial_mass = np.sum(masses[rads < virial_radius])
            Say.say('virial mass = %.3e' % virial_mass)
            return virial_radius
    else:
        Say.say('could not find virial radius')


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
                Say.say('! ids in final and initial particle catalogs not match')
                return
    Say.say('using species: %s' % spec_names)

    if center_pos is None or not len(center_pos):
        center_pos = get_center_position(part_fin, ['all'])

    distance_select_input = distance_select
    if not distance_select or scale_to_halo_radius:
        if not halo_radius:
            halo_radius = get_virial_radius(part_fin, ['all'], center_pos, virial_kind, 'log')
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
    mass_ini = volume_ini * density_ini    # assume cosmic density within volume

    Say.say('final redshift = %.3f, initial redshift = %.3f' %
            (part_fin.snap['redshift'], part_ini.snap['redshift']))
    Say.say('center of volume at final time = [%.3f, %.3f, %.3f] kpc comoving' %
            (center_pos[0], center_pos[1], center_pos[2]))
    Say.say('radius of selection volume at final time = %.3f kpc comoving' % distance_select)
    if scale_to_halo_radius:
        Say.say('  = %.2f x R_%s' % (distance_select_input, virial_kind))
    if use_onorbe_method:
        Say.say('radius of uncontaminated volume at final time = %.3f kpc comoving' % distance_pure)
    Say.say('number of particles in selection volume at final time = %d' % np.sum(spec_select_num))
    for spec_i in xrange(len(spec_names)):
        spec_name = spec_names[spec_i]
        Say.say('  species %s: number = %d' % (spec_name, spec_select_num[spec_i]))
    Say.say('mass of all dark-matter particles:')
    Say.say('  at highest-resolution in input catalog = %.2e M_sun' %
            part_ini['dark']['mass'].sum())
    Say.say('  in selection volume at final time = %.2e M_sun' % mass_select)
    if use_onorbe_method:
        Say.say('  in uncontaminated volume at final time = %.2e M_sun' % mass_pure)
    Say.say('  in convex hull at initial time = %.2e M_sun' % mass_ini)
    Say.say('volume of convex hull at initial time = %.1f Mpc ^ 3 comoving' %
            (volume_ini * const.mega_per_kilo ** 3))

    # MUSIC does not support header information in points file, so put in separate log file
    log_file_name = file_name.replace('.txt', '_log.txt')
    file_io = open(log_file_name, 'w')
    file_io.write('# final redshift = %.3f, initial redshift = %.3f\n' %
                  (part_fin.snap['redshift'], part_ini.snap['redshift']))
    file_io.write('# center of volume at final time = [%.3f, %.3f, %.3f] kpc comoving\n' %
                  (center_pos[0], center_pos[1], center_pos[2]))
    file_io.write('# radius of selection volume at final time = %.3f kpc comoving\n' %
                  distance_select)
    if scale_to_halo_radius:
        file_io.write('#   = %.2f x R_%s\n' % (distance_select_input, virial_kind))
    if use_onorbe_method:
        file_io.write('# radius of uncontaminated volume at final time = %.3f kpc comoving\n' %
                      distance_pure)
    file_io.write('# number of particles in selection volume at final time = %d\n' %
                  poss_ini.shape[0])
    for spec_i in xrange(len(spec_names)):
        file_io.write('#   species %s: number = %d\n' %
                      (spec_names[spec_i], spec_select_num[spec_i]))
    file_io.write('# mass of all dark-matter particles:\n')
    file_io.write('#   in selection volume at final time = %.2e M_sun\n' % mass_select)
    if use_onorbe_method:
        file_io.write('#   in uncontaminated volume at final time = %.2e M_sun\n' % mass_pure)
    file_io.write('#   in convex hull at initial time = %.2e M_sun\n' % mass_ini)
    file_io.write('# volume of convex hull at initial time = %.1f Mpc ^ 3 comoving\n' %
                  (volume_ini * const.mega_per_kilo ** 3))
    for dimen_i in xrange(poss_ini.shape[1]):
        file_io.write('# initial position-%s [min, max] = %s kpc comoving, %s box units\n' %
                      (dimen_i, ut.array.get_limits(poss_ini_limits[dimen_i], digit_num=3),
                       ut.array.get_limits(poss_ini_limits[dimen_i] / part_ini.info['box.length'],
                                           digit_num=8)))

    poss_ini /= part_ini.info['box.length']    # renormalize to box units

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
def test_contamination(
    part, center_pos=[], distance_lim=[1, 5000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', vir_radius=None, scale_vir=False,
    center_species=['star'], write_plot=False, plot_directory='.'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
        note: if not input, generate
    distance limits: list or array
    distance scaling: float
        options: log, lin
    '''
    species_test = ['dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6', 'gas', 'star']

    species_ref = 'dark'

    Say = ut.io.SayClass(test_contamination)

    species_test_t = []
    for spec_test in species_test:
        if spec_test in part:
            species_test_t.append(spec_test)
        else:
            Say.say('! no %s in particle dictionary' % spec_test)
    species_test = species_test_t

    if center_pos is None or not len(center_pos):
        center_pos = get_center_position(part, center_species)

    x_lim = np.array(distance_lim)
    if vir_radius and scale_vir:
        x_lim *= vir_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    pros = {species_ref: {}}
    for spec in species_test:
        pros[spec] = {}

    ratios = {}

    for spec in pros:
        dists = ut.coord.distance('scalar', part[spec]['position'], center_pos,
                                  part.info['box.length'])
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
    dists = 10 ** pros[spec]['distance.cum']
    print_string = '  d < %.3f kpc: mass = %.2e, number = %d'
    if scale_vir:
        dists /= vir_radius
        print_string = '  d/R_halo < %.3f: mass = %.2e, number = %d'
    for dist_i in xrange(pros[spec]['mass.cum'].size):
        if pros[spec]['mass.cum'][dist_i] > 0:
            Say.say(print_string % (dists[dist_i], pros[spec]['mass.cum'][dist_i],
                                    pros[spec]['mass.cum'][dist_i] / part[spec]['mass'][0]))

    # plot ----------
    colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if vir_radius and scale_vir:
        xs /= vir_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    #subplot.set_ylim([0, 0.1])
    subplot.set_ylim([0.0001, 3])
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    subplot.set_ylabel('$M_{\\rm spec} / M_{\\rm %s}$' % species_ref, fontsize=20)
    if scale_vir:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if vir_radius:
        if scale_vir:
            x_ref = 1
        else:
            x_ref = vir_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    #import ipdb; ipdb.set_trace()

    for spec_i, spec in enumerate(species_test):
        plot_func(xs, ratios[spec]['bin'], color=colors[spec_i], alpha=0.6, label=spec)

    legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    legend.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_safe_path(plot_directory)
        dist_name = 'dist'
        if vir_radius and scale_vir:
            dist_name += '.200m'
        plot_name = 'mass.ratio_v_%s_z.%.1f.pdf' % (dist_name, part.snap['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def test_metal_v_distance(
    part, center_pos=[], distance_lim=[10, 3000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', vir_radius=None, scale_vir=False,
    center_species=['star'],
    plot_kind='metalicity', write_plot=False, plot_directory='.'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
        note: if not input, generate
    distance limits: list or array
    distance scaling: float
        options: log, lin
    '''
    metal_index = 0    # overall metalicity

    Say = ut.io.SayClass(test_metal_v_distance)

    if center_pos is None or not len(center_pos):
        center_pos = get_center_position(part, center_species)

    x_lim = np.array(distance_lim)
    if vir_radius and scale_vir:
        x_lim *= vir_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    dists = ut.coord.distance('scalar', part['gas']['position'], center_pos,
                              part.info['box.length'])
    metal_masses = part['gas']['metal'][:, metal_index] * part['gas']['mass'] / 0.02    # solar

    pro_metal = DistanceBin.get_mass_profile(dists, metal_masses, get_spline=False)
    if plot_kind == 'metalicity':
        pro_mass = DistanceBin.get_mass_profile(dists, part['gas']['mass'], get_spline=False)
        ys = pro_metal['mass'] / pro_mass['mass']
        y_lim = np.clip(plot.get_limits(ys), 0.0001, 10)
    elif plot_kind == 'metal.mass':
        ys = pro_metal['frac.cum']
        y_lim = [0.001, 1]

    # plot ----------
    #colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if vir_radius and scale_vir:
        xs /= vir_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    #subplot.set_ylim([0, 0.1])
    subplot.set_ylim(y_lim)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    if plot_kind == 'metalicity':
        subplot.set_ylabel('$Z \, / \, Z_\odot$', fontsize=20)
    elif plot_kind == 'metal.mass':
        subplot.set_ylabel('$M_{\\rm Z}(< r) \, / \, M_{\\rm Z,tot}$', fontsize=20)
    if scale_vir:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if vir_radius:
        if scale_vir:
            x_ref = 1
        else:
            x_ref = vir_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    #import ipdb; ipdb.set_trace()

    plot_func(xs, ys, color='blue', alpha=0.6)

    #legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    #legend.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_safe_path(plot_directory)
        dist_name = 'dist'
        if vir_radius and scale_vir:
            dist_name += '.200m'
        plot_name = plot_kind + '_v_' + dist_name + '_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


#===================================================================================================
# analysis
#===================================================================================================
def get_sfr_history(part, pis=None, redshift_lim=[0, 1], aexp_wid=0.001):
    '''
    .
    '''
    if pis is None:
        pis = np.arange(part['mass'].size, dtype=np.int32)
    pis_sort = np.argsort(part['form.time'])
    star_form_aexps = part['form.time'][pis_sort]
    #star_form_redshifts = 1 / star_form_aexps - 1
    star_masses = part['mass'][pis_sort]
    star_masses_cum = np.cumsum(star_masses)

    if redshift_lim:
        redshift_lim = np.array(redshift_lim)
        aexp_lim = np.sort(1 / (1 + redshift_lim))
    else:
        aexp_lim = [np.min(star_form_aexps), np.max(star_form_aexps)]

    aexp_bins = np.arange(aexp_lim.min(), aexp_lim.max(), aexp_wid)
    redshift_bins = 1 / aexp_bins - 1
    time_bins = part.Cosmo.age(redshift_bins)
    #time_bins = part.Cosmo.age(0) - time_bins
    time_bins *= 1e9    # {yr}

    star_mass_cum_bins = np.interp(aexp_bins, star_form_aexps, star_masses_cum)
    dm_dts = np.diff(star_mass_cum_bins) / np.diff(time_bins) / 0.7    # account for mass loss

    time_mids = time_bins[0: time_bins.size - 1] + np.diff(time_bins)    # midpoints

    time_mids /= 1e9

    return time_mids, dm_dts


def plot_sfr_history(part, pis=None, redshift_lim=[0, 1], aexp_wid=0.001, write_plot=False):
    '''
    .
    '''
    times, sfrs = get_sfr_history(part, pis, redshift_lim, aexp_wid)

    # plot ----------
    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.14, hspace=0.03)

    #subplot.xscale('linear')
    #subplot.yscale('log')
    subplot.set_xlabel(r'time [Gyr]')
    #pylab.ylabel(r'${\rm SFR}\ \ \dot{M}_{\ast}\ \  [{\rm M_{\odot}\,yr^{-1}}]$')
    subplot.set_ylabel(r'${\rm SFR}\,[{\rm M_{\odot}\,yr^{-1}}]$')
    subplot.semilogy(times, sfrs, linewidth=2.0, color='r')

    #plt.tight_layout(pad=0.02)
    if write_plot:
        plt.savefig('sfr_v_time.pdf', format='pdf')
