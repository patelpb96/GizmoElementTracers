'''
Created on Jul 2, 2015

@author: awetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf  # @UnusedImport
from scipy import spatial
# local ----
from utilities import utility as ut
from utilities import constants as const


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
            halo_radius = ut.particle.get_halo_radius(
                part_fin, 'all', center_position, virial_kind, 'log')
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
