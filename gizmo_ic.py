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
# read data
#===================================================================================================
class ReadClass():
    '''
    Read particles and halo catalog.
    '''
    def __init__(self, snapshot_indices=[14, 0], simulation_directory='.'):
        '''
        Read particles from final and initial snapshot and halos from final snapshot.

        Parameters
        ----------
        snapshot_indices : list : indices of initial and final snapshots to read
        simulation_directory : string : root directory of simulation
        '''
        # ensure lowest-redshift snapshot is first
        self.snapshot_indices = np.sort(snapshot_indices)[::-1]
        self.simulation_directory = simulation_directory

    def read_all(self, mass_limits=[1e11, Inf]):
        '''
        Read particles from final and initial snapshot and halos from final snapshot.

        Returns
        -------
        parts : list : catalogs of particles at initial and final snapshots
        hal : list : catalog of halos at final snapshot
        '''
        from rockstar import rockstar_io

        parts = self.read_particles()
        hal = self.read_halos(mass_limits)

        rockstar_io.assign_lowres_mass(hal, parts[0], mass_limits)

        return parts, hal

    def read_particles(self):
        '''
        Read particles from final and initial snapshots.

        Returns
        -------
        parts : list : catalogs of particles at initial and final snapshots
        '''
        parts = []

        for snapshot_index in self.snapshot_indices:
            part = gizmo_io.Gizmo.read_snapshot(
                'all', 'index', snapshot_index, self.simulation_directory,
                property_names=['position', 'mass', 'id'], assign_center=False,
                sort_dark_by_id=True, force_float32=False)
            parts.append(part)

        return parts

    def read_halos(self, mass_limits=[1e11, Inf]):
        '''
        Read halos from final snapshot.

        Returns
        -------
        hal : list : catalog of halos at final snapshot
        '''
        from rockstar import rockstar_io

        hal = rockstar_io.Read.read_catalog(
            'index', self.snapshot_indices[0], self.simulation_directory, sort_by_mass=True,
            sort_host_first=False)

        rockstar_io.assign_nearest_neighbor(hal, 'total.mass', mass_limits, 1000, 6000, 'halo')

        return hal

Read = ReadClass()


#===================================================================================================
# analysis
#===================================================================================================
def get_indices(
    hal, mass_limits=[1.1e12, 1.8e12], distance_halo_min=6, contaminate_mass_frac_max=0.01):
    '''
    Get distances {kpc physical} and indices of halos that are within distance_max of center of
    input halo.

    Parameters
    ----------
    hal : dict : catalog of halos
    mass_limits : list : min and max limits of mass
    distance_halo_min : float : minimum d/R_{halo,nearest}
    contaminate_mass_frac_max : float : maximum contamination by mass from low-res dark particles

    Returns
    -------
    his_iso : array : distances of neighboring halos {kpc physical}
    neig_indices : array : indices of neighboring halos
    '''
    Say = ut.io.SayClass(get_indices)

    his = ut.array.get_indices(hal['lowres.mass.frac'], [0, contaminate_mass_frac_max])
    Say.say('{} with low contamination'.format(his.size))

    his = ut.array.get_indices(hal['total.mass'], mass_limits, his)
    Say.say('{} within mass limits = [{:.3e}, {:.3e}]'.format(
            his.size, mass_limits[0], mass_limits[1]))

    his = ut.array.get_indices(hal['near.halo.distance.halo'], [distance_halo_min, Inf], his)
    Say.say('{} with nearest more massive halo >{} d/R_halo,neig away'.format(
            his.size, distance_halo_min))

    return his


def print_properties(hal, hal_index, digits_after_period=3):
    '''
    Parameters
    ----------
    hal : dict : catalog of halos
    hal_index : int : index of halo
    digits_after_period : int
    '''
    prop_names = [
        'id', 'mass.200m', 'mass.vir', 'mass.200c', 'vel.circ.max',
        'spin.bullock', 'spin.peebles', 'position',
    ]

    print('halo index = {}'.format(hal_index))
    for prop_name in prop_names:
        string = ut.io.get_string_for_value(hal[prop_name][hal_index], digits_after_period)
        print('{} = {}'.format(prop_name, string))


def get_neighbors(
    hal, hal_index, distance_max=50, scale_to_halo_radius=True, neig_mass_frac_min=0.25):
    '''
    Get distances {kpc physical} and indices of halos that are within distance_max of center of
    input halo.

    Parameters
    ----------
    hal : dict : catalog of halos
    hal_index : int : index of center halo
    distance_max : float : maximum distance {kpc physical or d/R_halo}
    neig_mass_frac_min : float : minimum mass (relative to central) to select neighboring halos

    Returns
    -------
    neig_distances : array : distances of neighboring halos {kpc physical}
    neig_indices : array : indices of neighboring halos
    '''
    Say = ut.io.SayClass(get_neighbors)

    Neighbor = ut.neighbor.NeighborClass()

    if scale_to_halo_radius:
        distance_max *= hal['radius'][hal_index]
        Say.say('halo radius = {:.3f} kpc'.format(hal['radius'][hal_index]))

    mass_min = neig_mass_frac_min * hal['total.mass'][hal_index]
    his_m = ut.array.get_indices(hal['total.mass'], [mass_min, Inf])

    neig_distances, neig_indices = Neighbor.get_neighbors(
        hal['position'][[hal_index]], hal['position'][his_m], 2000, [1e-6, distance_max],
        hal.info['box.length'], hal.snapshot['scalefactor'], neig_ids=his_m)

    neig_distances = neig_distances[0]
    neig_indices = neig_indices[0]

    if scale_to_halo_radius:
        neig_distances /= hal['radius'][hal_index]

    if neig_distances.size:
        nearest_nii = np.argmin(neig_distances)
        nearest_distance = neig_distances[nearest_nii]
        nearest_mass = hal['total.mass'][neig_indices[nearest_nii]]
    else:
        nearest_distance = Inf
        nearest_mass = 0

    Say.say('nearest neighbor: distance = {:.3f}, mass = {:.2e}'.format(
            nearest_distance, nearest_mass))

    return neig_distances, neig_indices


def print_contamination_v_distance(
    part, hal, hal_index, distance_max=7, distance_bin_width=0.5, scale_to_halo_radius=True):
    '''
    Print information on contamination from lower-resolution particles around halo as a function of
    distance.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    hal : dict : catalog of halos at snapshot
    hal_index: int : index of halo
    distance_max : float : maximum distance from halo center to check
    distance_bin_width : float : width of distance bin for printing
    scale_to_halo_radius : boolean : whether to scale distances by virial radius
    '''
    from . import gizmo_analysis

    distance_scaling = 'lin'
    distance_limits = [0, distance_max]
    axis_y_scaling = 'log'

    Say = ut.io.SayClass(print_contamination_v_distance)

    Say.say('halo radius = {:.3f} kpc'.format(hal['radius'][hal_index]))

    halo_radius = hal['radius'][hal_index]

    gizmo_analysis.plot_mass_contamination(
        part, distance_limits, distance_bin_width, None, distance_scaling, halo_radius,
        scale_to_halo_radius, hal['position'][hal_index], axis_y_scaling, write_plot=None)


def print_contamination_in_box(
    part, center_position=None, distance_limits=None, distance_bin_number=20,
    distance_scaling='lin', geometry='cube'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    part : dict : catalog of particles
    center_position : array : 3-d position of center {kpc comoving}
    distance_limits : float : maximum distance from center to check {kpc physical}
    distance_bin_number : int : number of distance bins
    distance_scaling : string : 'log', 'lin'
    geometry : string : geometry of region: 'cube', 'sphere'
    '''
    Say = ut.io.SayClass(print_contamination_in_box)

    Neighbor = ut.neighbor.NeighborClass()

    if distance_limits is None:
        distance_limits = [0, 0.5 * (1 - 1e-5) * part.info['box.length']]

    if center_position is None:
        center_position = np.zeros(part['position'].shape[1])
        for dimension_i in range(part['position'].shape[1]):
            center_position[dimension_i] = 0.5 * part.info['box.length']
    print('center position = {}'.format(center_position))

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, number=distance_bin_number)

    masses_unique = np.unique(part['mass'])
    pis_all = ut.array.arange_length(part['mass'])
    pis_contam = pis_all[part['mass'] != masses_unique.min()]

    if geometry == 'sphere':
        distances, neig_pis = Neighbor.get_neighbors(
            center_position, part['position'], part['mass'].size,
            distance_limits, part.info['box.length'], neig_ids=pis_all)
        distances = distances[0]
        neig_pis = neig_pis[0]
    elif geometry == 'cube':
        distance_vector = np.abs(ut.coordinate.get_distances(
            'vector', part['position'], center_position, part.info['box.length']))

    for dist_i in range(DistanceBin.number):
        distance_bin_limits = DistanceBin.get_bin_limits(distance_scaling, dist_i)

        if geometry == 'sphere':
            pis_all_d = neig_pis[ut.array.get_indices(distances, distance_bin_limits)]
        elif geometry == 'cube':
            pis_all_d = np.array(pis_all)
            for dimension_i in range(part['position'].shape[1]):
                pis_all_d = ut.array.get_indices(
                    distance_vector[:, dimension_i], distance_bin_limits, pis_all_d)

        pis_contam_d = np.intersect1d(pis_all_d, pis_contam)
        frac = ut.math.Fraction.get_fraction(pis_contam_d.size, pis_all_d.size)
        Say.say('distance = [{:.3f}, {:.3f}], fraction = {:.5f}'.format(
                distance_bin_limits[0], distance_bin_limits[1], frac))

        if frac >= 1:
            break


#===================================================================================================
# generate region for initial conditions
#===================================================================================================
def write_initial_condition_points(
    parts, center_position=None, distance_max=7, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m', region_kind='convex-hull'):
    '''
    Select dark matter particles at final snapshot and print their positions at initial snapshot.
    Can use rules of thumb from Onorbe et al.

    Rule of thumb from Onorbe et al:
        given distance_pure
        if region_kind == 'cube':
            distance_max = (1.5 * refinement_number + 1) * distance_pure
        elif region_kind in ['particles', 'convex-hull']:
            distance_max = (1.5 * refinement_number + 7) * distance_pure

    Parameters
    ----------
    parts : list of dicts : catalogs of particles at final and initial snapshots
    center_position : list : center position at final time
    distance_max : float : distance from center to select particles at final time
        {kpc physical, or in units of R_halo}
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo {kpc physical}
    virial_kind : string : virial kind to use to get halo radius (if not input halo_radius)
    region_kind : string : method to identify zoom-in regon at initial time:
        'particles', 'convex-hull', 'cube'
    '''
    file_name = 'ic_agora_mX_points.txt'

    Say = ut.io.SayClass(write_initial_condition_points)

    assert region_kind in ['particles', 'convex-hull', 'cube']

    part_fin, part_ini = parts
    if part_fin.snapshot['redshift'] > part_ini.snapshot['redshift']:
        part_fin, part_ini = part_ini, part_fin

    # sanity check
    spec_names = ['dark', 'dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6']
    for spec_name in list(spec_names):
        if spec_name not in part_fin:
            spec_names.remove(spec_name)
            continue

        if np.min(part_fin[spec_name]['id'] == part_ini[spec_name]['id']) == False:
            Say.say('! species = {}: ids in final and initial catalogs not match'.format(
                    spec_name))
            return

    Say.say('using species: {}'.format(spec_names))

    center_position = ut.particle.parse_property(part_fin, 'position', center_position)

    if scale_to_halo_radius:
        if not halo_radius:
            halo_radius, _halo_mass = ut.particle.get_halo_radius_mass(
                part_fin, 'all', virial_kind, center_position=center_position)
        distance_max *= halo_radius

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
    poss_ini_limits = np.array([[positions_ini[:, dimen_i].min(), positions_ini[:, dimen_i].max()]
                                for dimen_i in range(positions_ini.shape[1])])

    # properties of initial volume
    density_ini = part_ini.Cosmology.get_density(
        'matter', part_ini.snapshot['redshift'], 'kpc comoving')
    if part_ini.info['has.baryons']:
        # subtract baryonic mass
        density_ini *= part_ini.Cosmology['omega_dark'] / part_ini.Cosmology['omega_matter']

    # convex hull
    volume_ini_chull = ut.coordinate.get_volume_of_convex_hull(positions_ini)
    mass_ini_chull = volume_ini_chull * density_ini  # assume cosmic density within volume

    # encompasing cube (relevant for MUSIC FFT)
    position_dif_max = 0
    for dimen_i in range(positions_ini.shape[1]):
        if poss_ini_limits[dimen_i].max() - poss_ini_limits[dimen_i].min() > position_dif_max:
            position_dif_max = poss_ini_limits[dimen_i].max() - poss_ini_limits[dimen_i].min()
    volume_ini_cube = position_dif_max ** 3
    mass_ini_cube = volume_ini_cube * density_ini  # assume cosmic density within volume

    # MUSIC does not support header information in points file, so put in separate log file
    log_file_name = file_name.replace('.txt', '_log.txt')
    file_out = open(log_file_name, 'w')

    Write = ut.io.WriteClass(file_out=file_out, print_stdout=True)

    Write.write('# redshift: final = {:.3f}, initial = {:.3f}'.format(
                part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
    Write.write('# center of region at final time = [{:.3f}, {:.3f}, {:.3f}] kpc comoving'.format(
                center_position[0], center_position[1], center_position[2]))
    Write.write('# radius of selection region at final time = {:.3f} kpc physical'.format(
                distance_max))
    if scale_to_halo_radius:
        Write.write('  = {:.2f} x R_{}, R_{} = {:.2f} kpc physical'.format(
                    distance_max / halo_radius, virial_kind, virial_kind, halo_radius))
    Write.write('# number of particles in selection region at final time = {}'.format(
                np.sum(spec_select_number)))
    for spec_i in range(len(spec_names)):
        spec_name = spec_names[spec_i]
        Write.write('  species {:6}: number = {}'.format(spec_name, spec_select_number[spec_i]))
    Write.write('# mass of all dark-matter particles:')
    Write.write('  at highest-resolution in input catalog = {:.2e} M_sun'.format(
                part_ini['dark']['mass'].sum()))
    Write.write('  in selection region at final time = {:.2e} M_sun'.format(mass_select))

    Write.write('# within convex hull at initial time')
    Write.write('  mass = {:.2e} M_sun'.format(mass_ini_chull))
    Write.write('  volume = {:.1f} Mpc ^ 3 comoving'.format(
                volume_ini_chull * ut.const.mega_per_kilo ** 3))

    Write.write('# within encompasing cube at initial time')
    Write.write('  mass = {:.2e} M_sun'.format(mass_ini_cube))
    Write.write('  volume = {:.1f} Mpc ^ 3 comoving'.format(
                volume_ini_cube * ut.const.mega_per_kilo ** 3))

    for dimen_i in range(positions_ini.shape[1]):
        Write.write('# initial position-{} [min, max] = {} kpc comoving, {} box units'.format(
                    dimen_i, ut.array.get_limits(poss_ini_limits[dimen_i], digit_number=3),
                    ut.array.get_limits(poss_ini_limits[dimen_i] / part_ini.info['box.length'],
                                        digit_number=8)))

    positions_ini /= part_ini.info['box.length']  # renormalize to box units

    if region_kind == 'convex-hull':
        # use convex hull to define initial region to reduce memory
        ConvexHull = spatial.ConvexHull(positions_ini)
        positions_ini = positions_ini[ConvexHull.vertices]
        Write.write('# using convex hull with {} vertices to define initial volume'.format(
                    positions_ini.shape[0]))

    file_out.close()

    file_out = open(file_name, 'w')
    for pi in range(positions_ini.shape[0]):
        file_out.write('{:.8f} {:.8f} {:.8f}\n'.format(
                       positions_ini[pi, 0], positions_ini[pi, 1], positions_ini[pi, 2]))
    file_out.close()


def generate_initial_condition_points(
    snapshot_indices=[14, 0], simulation_directory='.', distance_max=7, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m', region_kind='convex-hull'):
    '''
    Catch-all function: read particles, identify center, and write points for initial conditions.

    Parameters
    ----------
    snapshot_indices : list : indices of final and initial snapshots
    simulation_directory : string : directory of simulation
    distance_max : float : distance from center to select particles at final time
        {kpc physical, or in units of R_halo}
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo {kpc physical}
    virial_kind : string : virial kind to use to get halo radius (if not input halo_radius)
    region_kind : string : method to identify zoom-in regon at initial time:
        'particles', 'convex-hull', 'cube'
    '''
    if scale_to_halo_radius and distance_max < 1 or distance_max > 100:
        print('! selection radius = {} looks odd. are you sure aboutg this?'.format(distance_max))

    Read = ReadClass(snapshot_indices, simulation_directory)

    parts = Read.read_particles()

    center_position = ut.particle.get_center_position(
        parts[0], 'all', 'center-of-mass', compare_centers=True)

    write_initial_condition_points(
        parts, center_position, distance_max, scale_to_halo_radius, halo_radius, virial_kind,
        region_kind)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('must specify selection radius, in terms of R_200m')

    distance_max = float(sys.argv[1])

    generate_initial_condition_points(distance_max=distance_max)
