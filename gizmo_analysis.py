'''
Analysis of Gizmo simulations.

Masses in {M_sun}, positions in {kpc comoving}, distances and radii in {kpc physical}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import copy
# local ----
from utilities import utility as ut
from utilities import constants as const
from utilities import halo_property
from utilities import plot
from . import gizmo_io


#===================================================================================================
# utility
#===================================================================================================
def get_species_positions_masses(part, species):
    '''
    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string or list : list of species
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
            Say.say('! warning: allocating positions and masses for %d particles!' % particle_num)

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
    part, species=['star', 'dark', 'gas'], center_position=[], radius_max=1e10, method='cm'):
    '''
    Get position of center of mass, using iterative zoom-in on species.

    Parameters
    ----------
    part : dict : dictionary of particles
    species : string or list: names of species to use: 'all' = use all in particle dictionary
    center_position : array : initial center position
    radius_max : float : maximum initial radius to consider during iteration {kpc physical}
    method : string : method of centering: cm, potential
    '''
    if np.isscalar(species):
        species = [species]  # ensure is list

    if species == ['all']:
        species = ['star', 'dark', 'gas']

    positions, masses = get_species_positions_masses(part, species)

    #periodic_len = part.info['box.length']
    periodic_len = None  # assume zoom-in run far from box edge, for speed

    if method == 'cm':
        center_pos = ut.coord.position_center_of_mass_zoom(
            positions, masses, periodic_len, center_position, radius_max)
    elif method == 'potential':
        species = species[0]
        center_pos = part[species]['position'][np.argmin(part[species]['potential'])]

    return center_pos


def get_center_velocity(
    part, species='star', center_position=[], radius_max=100):
    '''
    Get velocity of center of mass.

    Parameters
    ----------
    part : dict : dictionary of particles
    species : string: name of species to use
    center_position : array : center position
    radius_max : float : maximum radius to consider {kpc physical}
    '''
    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    if np.unique(part[species]['mass']).size == 1:
        masses = None
    else:
        masses = part[species]['mass']

    #periodic_len = part.info['box.length']
    periodic_len = None  # assume zoom-in run far from box edge, for speed

    radius_max /= part.snapshot['scale-factor']  # convert to {kpc comoving} to match positions

    return ut.coord.velocity_center_of_mass(
        part[species]['velocity'], masses, part[species]['position'], center_position,
        radius_max, periodic_len)


def get_halo_radius(
    part, species=['dark', 'star', 'gas'], center_position=[], virial_kind='200m',
    radius_scaling='log', radius_lim=[5, 600], radius_bin_wid=0.02):
    '''
    Parameters
    ----------
    part : dict : catalog of particles at snapshot
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
    radius_bin_wid : float : wdith of radius bin (linear or logarithmic, set by radius_scaling)

    Returns
    -------
    halo virial radius {kpc physical}: float
    '''
    Say = ut.io.SayClass(get_halo_radius)

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    HaloProperty = halo_property.HaloPropertyClass(part.Cosmo, part.snapshot['redshift'])

    RadiusBin = ut.bin.DistanceBinClass(
        radius_scaling, radius_lim, width=radius_bin_wid, dimension_num=3)

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

    # get masses in bins
    if 'log' in radius_scaling:
        rads = log10(rads)
        radius_lim = log10(radius_lim)

    if np.isscalar(masses):
        mass_in_bins = np.histogram(rads, RadiusBin.num, radius_lim, False, None)[0]
    else:
        mass_in_bins = np.histogram(rads, RadiusBin.num, radius_lim, False, masses)[0]

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
    density_cum_in_bins = masses_cum / RadiusBin.volumes_cum

    # get smallest radius that satisfies virial density
    for r_bin_i in xrange(RadiusBin.num - 1):
        if (density_cum_in_bins[r_bin_i] >= virial_density and
                density_cum_in_bins[r_bin_i + 1] < virial_density):
            log_den_inner = log10(density_cum_in_bins[r_bin_i])
            log_den_outer = log10(density_cum_in_bins[r_bin_i + 1])
            # interpolate in log space
            log_rad_inner = RadiusBin.log_maxs[r_bin_i]
            log_rad_outer = RadiusBin.log_maxs[r_bin_i + 1]
            log_slope = (log_rad_outer - log_rad_inner) / (log_den_inner - log_den_outer)

            halo_radius = 10 ** (log_rad_inner + log_slope *
                                 (log_den_inner - log10(virial_density)))
            break
    else:
        Say.say('! could not find virial radius - might need to widen radius limits')
        return 0

    if 'log' in radius_scaling:
        rad_use = log10(halo_radius)
    else:
        rad_use = halo_radius

    if np.isscalar(masses):
        halo_mass = masses * np.sum(rads < rad_use)
    else:
        halo_mass = np.sum(masses[rads < rad_use])

    halo_radius *= part.snapshot['scale-factor']  # convert to {kpc physical}

    Say.say('M_%s = %.3e M_sun, log = %.3f\n  R_%s = %.3f kpc physical' %
            (virial_kind, halo_mass, log10(halo_mass), virial_kind, halo_radius))

    return halo_radius


def get_galaxy_radius(
    part, center_position=[], mass_percent=90, radius_max=30, radius_bin_wid=0.01,
    radius_scaling='log'):
    '''
    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    center_position : list/array : center position to use
    mass_percent : float : percent of mass (out to radius_max) to define radius
    radius_lim : list/array : maximum radius to consider {kpc physical}
    radius_bin_wid : float : width of radius bin {log kpc physical}
    radius_scaling : string : radius bin scaling: log, lin

    Returns
    -------
    galaxy radius {kpc physical}: float
    '''
    species = 'star'
    radius_min = 0.01  # {kpc physical}
    radius_lim = [radius_min, radius_max]

    Say = ut.io.SayClass(get_galaxy_radius)

    if not len(center_position) and len(part.center_position):
        center_position = part.center_position

    RadiusBin = ut.bin.DistanceBinClass(
        radius_scaling, radius_lim, width=radius_bin_wid, dimension_num=3)

    # {kpc comoving}
    rads = ut.coord.distance('scalar', part[species]['position'], center_position,
                             part.info['box.length'])
    rads *= part.snapshot['scale-factor']  # {kpc physical}

    # get masses in bins
    if 'log' in radius_scaling:
        rads = log10(rads)
        radius_lim = log10(radius_lim)

    mass_in_bins = np.histogram(rads, RadiusBin.num, radius_lim, False, part[species]['mass'])[0]

    # get mass within distance minimum, for computing cumulative values
    rad_indices = np.where(rads < radius_min)[0]
    log_masses_cum = np.log10(np.sum(part[species]['mass'][rad_indices]) + np.cumsum(mass_in_bins))

    log_mass = log10(mass_percent / 100) + log_masses_cum[-1]

    for r_bin_i in xrange(RadiusBin.num - 1):
        if (log_masses_cum[r_bin_i] <= log_mass and log_masses_cum[r_bin_i + 1] > log_mass):
            log_mass_inner = log_masses_cum[r_bin_i]
            log_mass_outer = log_masses_cum[r_bin_i + 1]
            # interpolate in log space
            log_rad_inner = RadiusBin.log_maxs[r_bin_i]
            log_rad_outer = RadiusBin.log_maxs[r_bin_i + 1]
            log_slope = (log_rad_outer - log_rad_inner) / (log_mass_outer - log_mass_inner)

            halo_radius = 10 ** (log_rad_inner + log_slope * (log_mass - log_mass_inner))
            break
    else:
        Say.say('! could not find virial radius - increase radius max')
        return 0

    if 'log' in radius_scaling:
        rad_use = log10(halo_radius)
    else:
        rad_use = halo_radius

    galaxy_mass = np.sum(part[species]['mass'][rads < rad_use])

    Say.say('M_star = %.2e M_sun, log = %.2f\n  R_%.0f = %.2f kpc physical' %
            (galaxy_mass, log10(galaxy_mass), mass_percent, halo_radius))

    return halo_radius


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
            positions, prop_vals = get_species_positions_masses(part, spec)

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
            for spec in species:
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
    part, species=['all'], prop_name='', center_position=[], DistanceBin=None):
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

        pros[spec] = DistanceBin.get_statistics_profile(distances, part[spec][prop_name])

    return pros


#===================================================================================================
# initial conditions
#===================================================================================================
def write_initial_condition_points(
    part_fin, part_ini, center_position=None, distance_select=None, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m',
    use_onorbe_method=False, refinement_num=1, method='particles'):
    '''
    Print positions of initial conditions of dark-matter particles selected at z = 0.
    Use rules of thumb from Onorbe et al.

    Parameters
    ----------
    part_fin : dict : catalog of particles at final snapshot
    part_ini : dict : catalog of particles at initial snapshot
    center_position : list : center position at final time
    distance_select : float : distance from center to select particles at final time
        {kpc physical, or units of R_halo}
        if None, use halo radius
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo
    virial_kind : string : virial kind for halo radius (if not input halo_radius)
    use_onorbe_method : boolean : whether to use method of Onorbe et al to get uncontaminated region
    refinement_num : int : if above is true, number of refinement levels beyond current for region
    method : string : method to identify initial zoom-in regon
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

    if not len(center_position) and len(part_fin.center_position):
        center_position = part_fin.center_position

    distance_select_input = distance_select
    if not distance_select or scale_to_halo_radius:
        if not halo_radius:
            halo_radius = get_halo_radius(part_fin, 'all', center_position, virial_kind, 'log')
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
        dists = ut.coord.distance('scalar', poss_fin, center_position, part_fin.info['box.length'])
        dists *= part_fin.snapshot['scale-factor']  # convert to {kpc physical}

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
    density_ini = part_ini.Cosmo.density_matter(part_ini.snapshot['redshift'])
    if part_ini.info['has.baryons']:
        # subtract baryonic mass
        density_ini *= part_ini.Cosmo['omega_dark'] / part_ini.Cosmo['omega_matter']
    mass_ini = volume_ini * density_ini  # assume cosmic density within volume

    Say.say('final redshift = %.3f, initial redshift = %.3f' %
            (part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
    Say.say('centering on volume at final time = [%.3f, %.3f, %.3f] kpc comoving' %
            (center_position[0], center_position[1], center_position[2]))
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
                  (part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
    file_io.write('# centering on volume at final time = [%.3f, %.3f, %.3f] kpc comoving\n' %
                  (center_position[0], center_position[1], center_position[2]))
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
    part, center_position=[], distance_lim=[1, 2000], distance_bin_wid=0.02, distance_bin_num=None,
    distance_scaling='log', axis_y_scaling='log', halo_radius=None, scale_to_halo_radius=False,
    write_plot=False, plot_directory='.'):
    '''
    Plot lower resolution particle contamination v distance from input center.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    center_position : array : position of galaxy center
    distance_lim : list : min and max limits for distance from galaxy
    distance_bin_wid : float : width of each distance bin (in units of distance_scaling)
    distance_bin_num : int : number of distance bins
    distance_scaling : string : lin or log
    axis_y_scaling : string : scaling of y-axis
    halo_radius : float : radius of halo {kpc comoving}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
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
    distances = pros[spec]['distance.cum']
    print_string = '  d < %.3f kpc: cumulative contamination mass = %.2e, number = %d'
    if scale_to_halo_radius:
        distances /= halo_radius
        print_string = '  d/R_halo < %.3f: mass = %.2e, number = %d'
    for dist_i in xrange(pros[spec]['mass.cum'].size):
        if pros[spec]['mass.cum'][dist_i] > 0:
            Say.say(print_string % (distances[dist_i], pros[spec]['mass.cum'][dist_i],
                                    pros[spec]['mass.cum'][dist_i] / part[spec]['mass'][0]))

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
        plot_name = 'mass.ratio_v_%s_z.%.1f.pdf' % (dist_name, part.snapshot['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def plot_metal_v_distance(
    part, species='gas', center_position=[],
    distance_lim=[10, 3000], distance_bin_wid=0.1, distance_bin_num=None, distance_scaling='log',
    axis_y_scaling='log', halo_radius=None, scale_to_halo_radius=False,
    plot_kind='metallicity', write_plot=False, plot_directory='.'):
    '''
    Plot metallicity (in bin or cumulative) of gas or stars v distance from galaxy.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string : particle species
    center_position : array : position of galaxy center
    distance_lim : list : min and max limits for distance from galaxy
    distance_bin_wid : float : width of each distance bin (in units of distance_scaling)
    distance_bin_num : int : number of distance bins
    distance_scaling : string : lin or log
    axis_y_scaling : string : scaling of y-axis
    halo_radius : float : radius of halo {kpc comoving}
    scale_to_halo_radius : boolean : whether to scale distance to halo_radius
    plot_kind : string : metallicity or metal.mass.cum
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
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    #fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

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

    # import ipdb; ipdb.set_trace()

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
    #else:
    #    plt.show(block=False)


#===================================================================================================
# analysis
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


def plot_property_distribution(
    parts, species='gas', prop_name='density', prop_scaling='log', prop_lim=[], prop_bin_wid=None,
    prop_bin_num=100, prop_stat='probability', center_positions=[], distance_lim=[],
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
    prop_stat : string : statistic to plot: probability,
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
        y_vals = [Stat.distr[prop_stat][part_i] for part_i in xrange(len(parts))]
        axis_y_lim = plot.get_limits(y_vals, axis_y_scaling, exclude_zero=True)
    subplot.set_ylim(axis_y_lim)

    subplot.set_xlabel(plot.get_label(prop_name, species=species, get_units=True))
    subplot.set_ylabel(plot.get_label(prop_name, prop_stat, species, get_symbol=True,
                                      get_units=False, draw_log=prop_scaling))

    #import ipdb; ipdb.set_trace()

    plot_func = plot.get_plot_function(subplot, prop_scaling, axis_y_scaling)
    for part_i, part in enumerate(parts):
        plot_func(Stat.distr['bin.mid'][part_i], Stat.distr[prop_stat][part_i],
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
    from matplotlib.colors import LogNorm

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
    parts, species='dark', prop_name='mass', prop_stat='hist', prop_scaling='log',
    distance_scaling='log', distance_lim=[0.1, 300], distance_bin_wid=0.02, distance_bin_num=None,
    center_positions=[], axis_y_lim=[], write_plot=False, plot_directory='.'):
    '''
    parts : dict or list : catalog[s] of particles (can be different simulations or snapshot)
    species : string or list : species to compute total mass of
        options: dark, star, gas, baryon, total
    prop_name : string : property to get profile of
    prop_stat : string : statistic/type to plot
        options: hist, hist.cum, density, density.cum, vel.circ, hist.fraction, hist.cum.fraction,
            med, ave
    prop_scaling : string : scaling for property (y-axis): lin, log
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
                part, species, prop_name, center_positions[part_i], DistanceBin)

        pros.append(pros_part)

        #if part_i > 0:
        #    print(pros[part_i][prop_name] / pros[0][prop_name])

    #import ipdb; ipdb.set_trace()

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    #fig, subplot = plt.subplots(1, 1, sharex=True)
    #fig, subplot = plt.subplots()
    fig.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.16, hspace=0.03)

    subplot.set_xlim(distance_lim)
    if not axis_y_lim:
        y_vals = [pro[species][prop_stat] for pro in pros]
        axis_y_lim = plot.get_limits(y_vals, prop_scaling)
        if prop_name == 'consume.time':
            axis_y_lim = plot.get_limits(
                pros[0][species][prop_stat][pros[0][species][prop_stat] < 10], prop_scaling)
    subplot.set_ylim(axis_y_lim)

    subplot.set_xlabel('radius $r$ $[\\rm kpc\,physical]$')
    label_y = plot.get_label(prop_name, prop_stat, species, get_symbol=True, get_units=True)
    subplot.set_ylabel(label_y)

    plot_func = plot.get_plot_function(subplot, distance_scaling, prop_scaling)
    colors = plot.get_colors(len(parts))

    if 'fraction' in prop_stat:
        plot_func(distance_lim, [1, 1], color='black', linestyle=':', alpha=0.5, linewidth=2)

    for part_i, pro in enumerate(pros):
        plot_func(pro[species]['distance'], pro[species][prop_stat], color=colors[part_i],
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
        plot_name = (species + '.' + prop_name + '.' + prop_stat +
                     '_v_dist_z.%.1f.pdf' % part.info['redshift'])
        plot_name = plot_name.replace('.hist', '')
        plot_name = plot_name.replace('mass.vel.circ', 'vel.circ')
        plot_name = plot_name.replace('mass.density', 'density')
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


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
                         alpha=0.5, label=part.info['catalog.name'])

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
def get_galaxy_mass_v_redshift(directory='.'):
    '''
    .
    '''
    #redshifts = [3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0]
    redshifts = [3.0, 2.0, 1.0, 0.0]

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
        part.center_position = get_center_position(part, 'star')

        gal_radius = get_galaxy_radius(part, [], 50, 30)
        hal_radius = get_halo_radius(part, species, virial_kind='200m')

        for k in ['redshift', 'scale-factor', 'time']:
            gal[k].append(part.snapshot[k])

        gal['position'].append(part.center_position)
        gal['star.velocity'].append(get_center_velocity(part, 'star', radius_max=gal_radius))
        gal['dark.velocity'].append(get_center_velocity(part, 'dark', radius_max=hal_radius / 2))

        for mass_percent in mass_percents:
            gal_radius = get_galaxy_radius(part, [], mass_percent, 30)
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
