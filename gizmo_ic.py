#!/usr/bin/env python


'''
Generate initial condition points by selecting particles at final time and tracing them back
to initial time.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import sys
import numpy as np
from scipy import spatial
# local ----
import utilities as ut
from . import gizmo_io


#===================================================================================================
# analysis
#===================================================================================================
def print_contamination_in_box(
    part, center_position=None, distance_limits=None, distance_bin_number=20,
    distance_scaling='linear', geometry='cube'):
    '''
    Test contamination from low-resolution particles around center.

    Parameters
    ----------
    part : dict : catalog of particles
    center_position : array : 3-d position of center [kpc comoving]
    distance_limits : float : maximum distance from center to check [kpc physical]
    distance_bin_number : int : number of distance bins
    distance_scaling : string : 'log', 'linear'
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
    pis_all = ut.array.get_arange(part['mass'])
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
# read data
#===================================================================================================
class ReadClass(ut.io.SayClass):
    '''
    Read particles and halo catalog.
    '''
    def __init__(self, snapshot_redshifts=[0, 99], simulation_directory='.'):
        '''
        Read particles from final and initial snapshot and halos from final snapshot.

        Parameters
        ----------
        snapshot_redshifts : list : redshifts of initial and final snapshots to read
        simulation_directory : string : root directory of simulation
        '''
        # ensure lowest-redshift snapshot is first
        self.snapshot_redshifts = np.sort(snapshot_redshifts)
        self.simulation_directory = simulation_directory

    def read_all(self, mass_limits=[1e11, np.Inf]):
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

        rockstar_io.Read.assign_lowres_mass(hal, parts[0], mass_limits)

        return parts, hal

    def read_particles(
        self, properties=['position', 'mass', 'id'], sort_dark_by_id=True, force_float32=False):
        '''
        Read particles from final and initial snapshots.

        Parameters
        ----------
        properties : string or list : name[s] of particle properties to read
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        force_float32 : boolean : whether to force all floats to 32-bit, to save memory

        Returns
        -------
        parts : list : catalogs of particles at initial and final snapshots
        '''
        parts = []

        for snapshot_redshift in self.snapshot_redshifts:
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                'all', 'redshift', snapshot_redshift, self.simulation_directory,
                properties=properties, assign_center=False, sort_dark_by_id=sort_dark_by_id,
                force_float32=force_float32)

            # if not sort dark particles, assign id-to-index coversion to track across snapshots
            if not sort_dark_by_id and snapshot_redshift == self.snapshot_redshifts[-1]:
                for spec in part:
                    self.say('assigning id-to-index to species: {}'.format(spec))
                    ut.catalog.assign_id_to_index(part[spec], 'id', 0)

            parts.append(part)

        return parts

    def read_halos(self, mass_limits=[1e11, np.Inf]):
        '''
        Read halos from final snapshot.

        Returns
        -------
        hal : list : catalog of halos at final snapshot
        '''
        from rockstar import rockstar_io

        Read = rockstar_io.ReadClass()
        hal = Read.read_catalogs(
            'redshift', self.snapshot_redshifts[0], self.simulation_directory, sort_by_mass=False,
            sort_host_first=False)

        rockstar_io.Read.assign_nearest_neighbor(hal, 'total.mass', mass_limits, 1000, 6000, 'halo')

        return hal


Read = ReadClass()


#===================================================================================================
# generate region for initial conditions
#===================================================================================================
def write_initial_points(
    parts, center_position=None, distance_max=7, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m', region_kind='convex-hull', dark_mass=None):
    '''
    Select dark matter particles at final snapshot and print their positions at initial snapshot.

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
        [kpc physical, or in units of R_halo]
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo [kpc physical]
    virial_kind : string : virial kind to use to get halo radius (if not input halo_radius)
    region_kind : string : method to identify zoom-in regon at initial time:
        'particles', 'convex-hull', 'cube'
    dark_mass : float : dark-matter particle mass (if simulation has only DM, at single resolution)
    '''
    file_name = 'ic_agora_mX_rad{:.1f}_points.txt'.format(distance_max)

    Say = ut.io.SayClass(write_initial_points)

    assert region_kind in ['particles', 'convex-hull', 'cube']

    # ensure 'final' is lowest redshift
    part_fin, part_ini = parts
    if part_fin.snapshot['redshift'] > part_ini.snapshot['redshift']:
        part_fin, part_ini = part_ini, part_fin

    # determine which species are in catalog
    species = ['dark', 'dark.2', 'dark.3', 'dark.4', 'dark.5', 'dark.6']
    for spec in list(species):
        if spec not in part_fin:
            species.remove(spec)
            continue

        # sanity check
        if 'id.to.index' not in part_ini[spec]:
            if np.min(part_fin[spec]['id'] == part_ini[spec]['id']) == False:
                Say.say('! species = {}: ids not match in final v initial catalogs'.format(spec))
                return

    # sanity check
    if dark_mass:
        if species != ['dark']:
            raise ValueError(
                'input dark_mass = {:.3e} Msun, but catalog contains species = {}'.format(
                    dark_mass, species))
        if scale_to_halo_radius and not halo_radius:
            raise ValueError('cannot determine halo_radius without mass in particle catalog')

    Say.say('using species: {}'.format(species))

    center_position = ut.particle.parse_property(part_fin, 'position', center_position)

    if scale_to_halo_radius:
        if not halo_radius:
            halo_prop = ut.particle.get_halo_properties(
                part_fin, 'all', virial_kind, center_position=center_position)
            halo_radius = halo_prop['radius']
        distance_max *= halo_radius

    mass_select = 0
    positions_ini = []
    spec_select_number = []
    for spec in species:
        distances = ut.coordinate.get_distances(
            'scalar', part_fin[spec]['position'], center_position,
            part_fin.info['box.length']) * part_fin.snapshot['scalefactor']  # [kpc physical]

        indices_fin = ut.array.get_indices(distances, [0, distance_max])

        # if id-to-index array is in species dictionary
        # assume id not sorted, so have to convert between id and index
        if 'id.to.index' in part_ini[spec]:
            ids = part_fin[spec]['id'][indices_fin]
            indices_ini = part_ini[spec]['id.to.index'][ids]
        else:
            indices_ini = indices_fin

        positions_ini.extend(part_ini[spec]['position'][indices_ini])

        if 'mass' in part_ini[spec]:
            mass_select += part_ini[spec]['mass'][indices_ini].sum()
        elif dark_mass:
            mass_select += dark_mass * indices_ini.size
        else:
            raise ValueError(
                'no mass for species = {} but also no input dark_mass'.format(spec))

        spec_select_number.append(indices_ini.size)

    positions_ini = np.array(positions_ini)
    poss_ini_limits = np.array([[positions_ini[:, dimen_i].min(), positions_ini[:, dimen_i].max()]
                                for dimen_i in range(positions_ini.shape[1])])

    # properties of initial volume
    density_ini = part_ini.Cosmology.get_density(
        'matter', part_ini.snapshot['redshift'], 'kpc comoving')
    if part_ini.info['has.baryons']:
        # subtract baryonic mass
        density_ini *= part_ini.Cosmology['omega_dm'] / part_ini.Cosmology['omega_matter']

    # convex hull
    volume_ini_chull = ut.coordinate.get_volume_of_convex_hull(positions_ini)
    mass_ini_chull = volume_ini_chull * density_ini  # assume cosmic density within volume

    # encompassing cube (relevant for MUSIC FFT) and cuboid
    position_difs = []
    for dimen_i in range(positions_ini.shape[1]):
        position_difs.append(poss_ini_limits[dimen_i].max() - poss_ini_limits[dimen_i].min())
    volume_ini_cube = max(position_difs) ** 3
    mass_ini_cube = volume_ini_cube * density_ini  # assume cosmic density within volume

    volume_ini_cuboid = 1.
    for dimen_i in range(positions_ini.shape[1]):
        volume_ini_cuboid *= position_difs[dimen_i]
    mass_ini_cuboid = volume_ini_cuboid * density_ini  # assume cosmic density within volume

    # MUSIC does not support header information in points file, so put in separate log file
    log_file_name = file_name.replace('.txt', '_log.txt')

    with open(log_file_name, 'w') as file_out:
        Write = ut.io.WriteClass(file_out, print_stdout=True)

        Write.write('# redshift: final = {:.3f}, initial = {:.3f}'.format(
                    part_fin.snapshot['redshift'], part_ini.snapshot['redshift']))
        Write.write(
            '# center of region at final time = [{:.3f}, {:.3f}, {:.3f}] kpc comoving'.format(
                center_position[0], center_position[1], center_position[2]))
        Write.write('# radius of selection region at final time = {:.3f} kpc physical'.format(
                    distance_max))
        if scale_to_halo_radius:
            Write.write('  = {:.2f} x R_{}, R_{} = {:.2f} kpc physical'.format(
                        distance_max / halo_radius, virial_kind, virial_kind, halo_radius))
        Write.write('# number of particles in selection region at final time = {}'.format(
                    np.sum(spec_select_number)))
        for spec_i, spec in enumerate(species):
            Write.write('  species {:6}: number = {}'.format(spec, spec_select_number[spec_i]))
        Write.write('# mass of all dark-matter particles:')
        if 'mass' in part_ini['dark']:
            mass_dark_all = part_ini['dark']['mass'].sum()
        else:
            mass_dark_all = dark_mass * part_ini['dark']['id'].size
        Write.write('  at highest-resolution in input catalog = {:.2e} M_sun'.format(mass_dark_all))
        Write.write('  in selection region at final time = {:.2e} M_sun'.format(mass_select))

        Write.write('# within convex hull at initial time')
        Write.write('  mass = {:.2e} M_sun'.format(mass_ini_chull))
        Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                    volume_ini_chull * ut.const.mega_per_kilo ** 3))

        Write.write('# within encompassing cuboid at initial time')
        Write.write('  mass = {:.2e} M_sun'.format(mass_ini_cuboid))
        Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                    volume_ini_cuboid * ut.const.mega_per_kilo ** 3))

        Write.write('# within encompassing cube at initial time (for MUSIC FFT)')
        Write.write('  mass = {:.2e} M_sun'.format(mass_ini_cube))
        Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                    volume_ini_cube * ut.const.mega_per_kilo ** 3))

        Write.write('# position range at initial time')
        for dimen_i in range(positions_ini.shape[1]):
            string = ('  {} [min, max, width] = [{:.2f}, {:.2f}, {:.2f}] kpc comoving\n' +
                      '        [{:.9f}, {:.9f}, {:.9f}] box units')
            pos_min = np.min(poss_ini_limits[dimen_i])
            pos_max = np.max(poss_ini_limits[dimen_i])
            pos_width = np.max(poss_ini_limits[dimen_i]) - np.min(poss_ini_limits[dimen_i])
            Write.write(
                string.format(
                    dimen_i, pos_min, pos_max, pos_width,
                    pos_min / part_ini.info['box.length'], pos_max / part_ini.info['box.length'],
                    pos_width / part_ini.info['box.length']
                )
            )

        positions_ini /= part_ini.info['box.length']  # renormalize to box units

        if region_kind == 'convex-hull':
            # use convex hull to define initial region to reduce memory
            ConvexHull = spatial.ConvexHull(positions_ini)
            positions_ini = positions_ini[ConvexHull.vertices]
            Write.write('# using convex hull with {} vertices to define initial volume'.format(
                        positions_ini.shape[0]))

    with open(file_name, 'w') as file_out:
        for pi in range(positions_ini.shape[0]):
            file_out.write('{:.8f} {:.8f} {:.8f}\n'.format(
                           positions_ini[pi, 0], positions_ini[pi, 1], positions_ini[pi, 2]))


def write_initial_points_from_uniform(
    parts, hal, hal_index, distance_max=10, scale_to_halo_radius=True, virial_kind='200m',
    region_kind='convex-hull', dark_mass=None):
    '''
    Pipeline to generate and write initial condition points
    *from a uniform-resolution DM-only simulation with a halo catalog*.

    Parameters
    ----------
    parts : list of dicts : catalogs of particles at final and initial snapshots
    hal : dict : halo catalog at final snapshot
    hal_index : int : index of halo in catalog
    virial_kind : string : virial kind to use to define halo radius in catalog
    distance_max : float : distance from center to select particles at final time
        [kpc physical, or in units of R_halo]
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo [kpc physical]
    region_kind : string : method to identify zoom-in regon at initial time:
        'particles', 'convex-hull', 'cube'
    dark_mass : float : dark-matter particle mass (if simulation has only DM, at single resolution)
    '''
    if scale_to_halo_radius and distance_max < 1 or distance_max > 100:
        print('! selection radius = {} looks odd. are you sure?'.format(distance_max))

    center_position = hal['position'][hal_index]
    halo_radius = hal['radius.' + virial_kind][hal_index]

    write_initial_points(
        parts, center_position, distance_max, scale_to_halo_radius, halo_radius, virial_kind,
        region_kind, dark_mass)


def read_write_initial_points_from_zoom(
    snapshot_redshifts=[0, 99], distance_max=7, scale_to_halo_radius=True,
    halo_radius=None, virial_kind='200m', region_kind='convex-hull', simulation_directory='.'):
    '''
    Complete pipeline to generate and write initial condition points
    *from an existing zoom-in simulation*:
        (1) read particles, (2) identify halo center, (3) identify zoom-in region around center,
        (4) write positions of particles at initial redshift

    Parameters
    ----------
    snapshot_redshifts : list : redshifts of final and initial snapshots
    distance_max : float : distance from center to select particles at final time
        [kpc physical, or in units of R_halo]
    scale_to_halo_radius : boolean : whether to scale distance to halo radius
    halo_radius : float : radius of halo [kpc physical]
    virial_kind : string : virial kind to use to get halo radius (if not input halo_radius)
    region_kind : string : method to determine zoom-in regon at initial time:
        'particles', 'convex-hull', 'cube'
    simulation_directory : string : directory of simulation
    '''
    if scale_to_halo_radius and distance_max < 1 or distance_max > 100:
        print('! selection radius = {} looks odd. are you sure?'.format(distance_max))

    Read = ReadClass(snapshot_redshifts, simulation_directory)
    parts = Read.read_particles()

    center_position = ut.particle.get_center_position(
        parts[0], 'dark', 'center-of-mass', compare_centers=True)

    write_initial_points(
        parts, center_position, distance_max, scale_to_halo_radius, halo_radius, virial_kind,
        region_kind)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('must specify selection radius, in terms of R_200m')

    distance_max = float(sys.argv[1])

    read_write_initial_points_from_zoom(distance_max=distance_max)
