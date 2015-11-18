'''
Generate initial condition points by selecting particles at final time and tracing them back
to initial time.

Masses in {M_sun}, positions in {kpc comoving}, distances in {kpc physical}.

@author: awetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from numpy import log10, Inf  # @UnusedImport
from scipy import spatial
# local ----
import utilities as ut
from . import gizmo_io


#===================================================================================================
# initial conditions
#===================================================================================================
def write_initial_condition_points(
    part_fin, part_ini, center_position=None, distance_max=None, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m',
    use_onorbe_method=False, refinement_number=1, method='convex-hull'):
    '''
    Print positions dark-matter particles at initial snapshot as selected at final snapshot.
    Can use rules of thumb from Onorbe et al.

    Parameters
    ----------
    part_fin : dict : catalog of particles at final snapshot
    part_ini : dict : catalog of particles at initial snapshot
    center_position : list : center position at final time
    distance_max : float : distance from center to select particles at final time
        {kpc physical, or units of R_halo}
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo {kpc physical}
    virial_kind : string : virial kind for halo radius (if not input halo_radius)
    use_onorbe_method : boolean : whether to use method of Onorbe et al to get uncontaminated region
    refinement_number : int : if above true, number of refinement levels beyond current for region
    method : string : method to identify initial zoom-in regon
        options: particles, convex-hull, cube
    '''
    file_name = 'ic_agora_mX_points.txt'

    Say = ut.io.SayClass(write_initial_condition_points)

    assert method in ['particles', 'convex-hull', 'cube']

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

    center_position = ut.particle.parse_property(part_fin, 'position', center_position)

    if scale_to_halo_radius:
        if not halo_radius:
            halo_radius, _halo_mass = ut.particle.get_halo_radius_mass(
                part_fin, 'all', center_position, virial_kind)
        distance_max *= halo_radius

    if use_onorbe_method:
        # convert distance_max according to Onorbe et al
        distance_pure = distance_max
        if method == 'cube':
            distance_max = (1.5 * refinement_number + 1) * distance_pure
        elif method in ['particles', 'convex-hull']:
            distance_max = (1.5 * refinement_number + 7) * distance_pure

    mass_select = 0
    positions_ini = []
    spec_select_number = []
    for spec_name in spec_names:
        positions_fin = part_fin[spec_name]['position']

        distances = ut.coordinate.get_distances(
            'scalar', positions_fin, center_position, part_fin.info['box.length'])
        distances *= part_fin.snapshot['scalefactor']  # convert to {kpc physical}

        select_indices = ut.array.get_indices(distances, [0, distance_max])

        positions_ini.extend(part_ini[spec_name]['position'][select_indices])

        mass_select += part_ini[spec_name]['mass'][select_indices].sum()
        spec_select_number.append(select_indices.size)

    positions_ini = np.array(positions_ini)
    poss_ini_limits = [[positions_ini[:, dimen_i].min(), positions_ini[:, dimen_i].max()]
                       for dimen_i in range(positions_ini.shape[1])]

    volume_ini = ut.coordinate.get_volume_of_convex_hull(positions_ini)
    density_ini = part_ini.Cosmology.density(
        'matter', part_ini.snapshot['redshift'], 'kpc comoving')
    if part_ini.info['has.baryons']:
        # subtract baryonic mass
        density_ini *= part_ini.Cosmology['omega_dark'] / part_ini.Cosmology['omega_matter']
    mass_ini = volume_ini * density_ini  # assume cosmic density within volume

    Say.say('final redshift = %.3f, initial redshift = %.3f' %
            (part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
    Say.say('center of volume at final time = [%.3f, %.3f, %.3f] kpc comoving' %
            (center_position[0], center_position[1], center_position[2]))
    if scale_to_halo_radius:
        Say.say('selection radius = %.2f x R_%s, R_%s = %.2f kpc physical' %
                (distance_max / halo_radius, virial_kind, virial_kind, halo_radius))
    Say.say('radius of selection volume at final time = %.3f kpc physical' % distance_max)
    if use_onorbe_method:
        Say.say('radius of uncontaminated volume (Onorbe et al) at final time = %.3f kpc physical' %
                distance_pure)
    Say.say('number of particles in selection volume at final time = %d' %
            np.sum(spec_select_number))
    for spec_i in range(len(spec_names)):
        spec_name = spec_names[spec_i]
        Say.say('  species %s: number = %d' % (spec_name, spec_select_number[spec_i]))
    Say.say('mass of all dark-matter particles:')
    Say.say('  at highest-resolution in input catalog = %.2e M_sun' %
            part_ini['dark']['mass'].sum())
    Say.say('  in selection volume at final time = %.2e M_sun' % mass_select)
    Say.say('  in convex hull at initial time = %.2e M_sun' % mass_ini)
    Say.say('volume of convex hull at initial time = %.1f Mpc ^ 3 comoving' %
            (volume_ini * ut.const.mega_per_kilo ** 3))

    # MUSIC does not support header information in points file, so put in separate log file
    log_file_name = file_name.replace('.txt', '_log.txt')
    file_io = open(log_file_name, 'w')
    file_io.write('# final redshift = %.3f, initial redshift = %.3f\n' %
                  (part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
    file_io.write('# center of volume at final time = [%.3f, %.3f, %.3f] kpc comoving\n' %
                  (center_position[0], center_position[1], center_position[2]))
    if scale_to_halo_radius:
        file_io.write('# selection radius = %.2f x R_%s, R_%s = %.2f kpc physical\n' %
                      (distance_max / halo_radius, virial_kind, virial_kind, halo_radius))
    file_io.write('# radius of selection volume at final time = %.3f kpc physical\n' %
                  distance_max)
    if use_onorbe_method:
        file_io.write(
            '# radius of uncontaminated volume (Onorbe et al) at final time = %.3f kpc physical\n' %
            distance_pure)
    file_io.write('# number of particles in selection volume at final time = %d\n' %
                  positions_ini.shape[0])
    for spec_i in range(len(spec_names)):
        file_io.write('#   species %s: number = %d\n' %
                      (spec_names[spec_i], spec_select_number[spec_i]))
    file_io.write('# mass of all dark-matter particles:\n')
    file_io.write('#   at highest-resolution in input catalog = %.2e M_sun\n' %
                  part_ini['dark']['mass'].sum())
    file_io.write('#   in selection volume at final time = %.2e M_sun\n' % mass_select)
    file_io.write('#   in convex hull at initial time = %.2e M_sun\n' % mass_ini)
    file_io.write('# volume of convex hull at initial time = %.1f Mpc ^ 3 comoving\n' %
                  (volume_ini * ut.const.mega_per_kilo ** 3))
    for dimen_i in range(positions_ini.shape[1]):
        file_io.write('# initial position-%s [min, max] = %s kpc comoving, %s box units\n' %
                      (dimen_i, ut.array.get_limits(poss_ini_limits[dimen_i], digit_number=3),
                       ut.array.get_limits(poss_ini_limits[dimen_i] / part_ini.info['box.length'],
                                           digit_number=8)))

    positions_ini /= part_ini.info['box.length']  # renormalize to box units

    if method == 'convex-hull':
        # use convex hull to define initial region to reduce memory
        ConvexHull = spatial.ConvexHull(positions_ini)
        positions_ini = positions_ini[ConvexHull.vertices]
        file_io.write('# using convex hull with %d vertices to define initial volume\n' %
                      positions_ini.shape[0])

    file_io.close()

    file_io = open(file_name, 'w')
    for pi in range(positions_ini.shape[0]):
        file_io.write('%.8f %.8f %.8f\n' %
                      (positions_ini[pi, 0], positions_ini[pi, 1], positions_ini[pi, 2]))
    file_io.close()


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('must specify selection radius, in terms of R_200m')

    distance_max = float(sys.argv[1])
    if distance_max < 1 or distance_max > 100:
        raise ValueError('selection radius = %s seems odd. you shall not pass.' % distance_max)

    part_fin = gizmo_io.Gizmo.read_snapshot(
        ['dark', 'dark.2'], 'redshift', 0, 'output', ['position', 'id', 'mass'],
        sort_dark_by_id=True, force_float32=False, assign_center=False)

    part_ini = gizmo_io.Gizmo.read_snapshot(
        ['dark', 'dark.2'], 'index', 0, 'output', ['position', 'id', 'mass'],
        sort_dark_by_id=True, force_float32=False, assign_center=False)

    center_position = ut.particle.get_center_position(
        part_fin, 'dark', 'center-of-mass', compare_centers=True)

    write_initial_condition_points(
        part_fin, part_ini, center_position, distance_max, scale_to_halo_radius=True,
        method='convex-hull')
