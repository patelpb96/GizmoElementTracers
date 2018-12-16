#!/usr/bin/env python

'''
Generate initial condition points by selecting particles at final time and tracking them back
to initial time.

@author: Andrew Wetzel

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import sys
import numpy as np
from scipy import spatial
# local ----
import utilities as ut
from . import gizmo_io
from rockstar_analysis import rockstar_io


#===================================================================================================
# read data
#===================================================================================================
class ReadClass(ut.io.SayClass):
    '''
    Read particles and halo catalog.
    '''

    def __init__(self, snapshot_redshifts=[0, 99], simulation_directory='.'):
        '''
        Read particles at final and initial snapshots and halos at final snapshot.

        Parameters
        ----------
        snapshot_redshifts : list : redshifts of initial and final snapshots
        simulation_directory : str : root directory of simulation
        '''
        # ensure lowest-redshift snapshot is first
        self.snapshot_redshifts = np.sort(snapshot_redshifts)
        self.simulation_directory = simulation_directory

    def read_all(self, mass_limits=[1e11, np.Inf]):
        '''
        Read particles at final and initial snapshots and halos at final snapshot.

        Parameters
        ----------
        mass_limits : list : min and max halo mass to assign low-res DM mass

        Returns
        -------
        parts : list of dictionaries : catalogs of particles at initial and final snapshots
        hal : dictionary class : catalog of halos at final snapshot
        '''
        hal = self.read_halos(mass_limits)
        parts = self.read_particles()

        if 'dark2' in parts[0] and 'mass' in parts[0]['dark2'] and len(parts[0]['dark2']['mass']):
            rockstar_io.Particle.assign_lowres_mass(hal, parts[0])

        return parts, hal

    def read_halos(
        self, mass_limits=[1e11, np.Inf], file_kind='out', assign_nearest_neighbor=False):
        '''
        Read halos at final snapshot.

        Parameters
        ----------
        mass_limits : list : min and max halo mass to assign low-res DM mass
        file_kind : str : kind of halo file: 'hdf5', 'out', 'ascii', 'hlist'
        assign_nearest_neighbor : bool : whether to assign nearest neighboring halo

        Returns
        -------
        hal : dictionary class : catalog of halos at final snapshot
        '''
        hal = rockstar_io.IO.read_catalogs(
            'redshift', self.snapshot_redshifts[0], self.simulation_directory, file_kind=file_kind)

        if assign_nearest_neighbor:
            rockstar_io.IO.assign_nearest_neighbor(hal, 'mass', mass_limits, 2000, 'Rneig', 8000)

        return hal

    def read_particles(
        self, properties=['position', 'mass', 'id'], sort_dark_by_id=True):
        '''
        Read particles at final and initial snapshots.

        Parameters
        ----------
        properties : str or list : name[s] of particle properties to read
        sort_dark_by_id : bool : whether to sort dark-matter particles by id

        Returns
        -------
        parts : list : catalogs of particles at initial and final snapshots
        '''
        parts = []

        for snapshot_redshift in self.snapshot_redshifts:
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                'all', 'redshift', snapshot_redshift, self.simulation_directory,
                properties=properties, assign_host_coordinates=False,
                sort_dark_by_id=sort_dark_by_id)

            # if not sort dark particles, assign id-to-index coversion to track across snapshots
            if not sort_dark_by_id and snapshot_redshift == self.snapshot_redshifts[-1]:
                for spec in part:
                    self.say('assigning id-to-index to species: {}'.format(spec))
                    ut.catalog.assign_id_to_index(part[spec], 'id', 0)

            parts.append(part)

        return parts


Read = ReadClass()


#===================================================================================================
# generate region for initial conditions
#===================================================================================================
class InitialConditionClass(ut.io.SayClass):
    '''
    Generate text file of positions of particles at the initial snapshot that are within the
    selection region at the final snapshot.
    '''

    def write_initial_positions(
        self, parts, center_position=None, distance_max=7, scale_to_halo_radius=True,
        halo_radius=None, virial_kind='200m', region_kind='convex-hull', dark_mass=None):
        '''
        Select dark-matter particles at final snapshot, print their positions at initial snapshot.

        Rule of thumb from Onorbe et al:
            given distance_pure
            if region_kind == 'cube':
                distance_max = (1.5 * refinement_number + 1) * distance_pure
            elif region_kind in ['particles', 'convex-hull']:
                distance_max = (1.5 * refinement_number + 7) * distance_pure

        Parameters
        ----------
        parts : list of dicts : catalogs of particles at final and initial snapshots
        center_position : list : center position at final snapshot
        distance_max : float : distance from center to select particles at final time
            [kpc physical or in units of R_halo]
        scale_to_halo_radius : bool : whether to scale distance to halo radius
        halo_radius : float : radius of halo [kpc physical]
        virial_kind : str : virial kind to use to get halo radius (if not input halo_radius)
        region_kind : str : method to identify zoom-in regon at initial time:
            'particles', 'convex-hull', 'cube'
        dark_mass : float : DM particle mass (if simulation has only DM at single resolution)
        '''
        file_name = 'ic_L_mX_rad{:.1f}_points.txt'.format(distance_max)

        assert region_kind in ['particles', 'convex-hull', 'cube']

        # ensure 'final' is lowest redshift
        part_fin, part_ini = parts
        if part_fin.snapshot['redshift'] > part_ini.snapshot['redshift']:
            part_fin, part_ini = part_ini, part_fin

        # determine which species are in catalog
        species = ['dark', 'dark2', 'dark3', 'dark4', 'dark5', 'dark6']
        for spec in list(species):
            if spec not in part_fin:
                species.remove(spec)
                continue

            # sanity check
            if 'id.to.index' not in part_ini[spec]:
                if np.min(part_fin[spec]['id'] == part_ini[spec]['id']) == False:
                    self.say('! species = {}: ids not match in final v initial catalogs'.format(
                        spec))
                    return

        # sanity check
        if dark_mass:
            if species != ['dark']:
                raise ValueError(
                    'input dark_mass = {:.3e} Msun, but catalog contains species = {}'.format(
                        dark_mass, species))
            if scale_to_halo_radius and not halo_radius:
                raise ValueError('cannot determine halo_radius without mass in particle catalog')

        self.say('using species: {}'.format(species))

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
                part_fin[spec]['position'], center_position, part_fin.info['box.length'],
                part_fin.snapshot['scalefactor'], total_distance=True)  # [kpc physical]

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
        poss_ini_limits = np.array(
            [[positions_ini[:, dimen_i].min(), positions_ini[:, dimen_i].max()]
             for dimen_i in range(positions_ini.shape[1])]
        )

        # properties of initial volume
        density_ini = part_ini.Cosmology.get_density(
            'matter', part_ini.snapshot['redshift'], 'kpc comoving')
        if part_ini.info['baryonic']:
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
            Write.write('# mass from all dark-matter particles:')
            if 'mass' in part_ini['dark']:
                mass_dark_all = part_ini['dark']['mass'].sum()
            else:
                mass_dark_all = dark_mass * part_ini['dark']['id'].size
            Write.write('  at highest-resolution in input catalog = {:.2e} M_sun'.format(
                mass_dark_all))
            Write.write('  in selection region at final time = {:.2e} M_sun'.format(mass_select))

            Write.write('# within convex hull at initial time')
            Write.write('  mass = {:.2e} M_sun'.format(mass_ini_chull))
            Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                        volume_ini_chull * ut.constant.mega_per_kilo ** 3))

            Write.write('# within encompassing cuboid at initial time')
            Write.write('  mass = {:.2e} M_sun'.format(mass_ini_cuboid))
            Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                        volume_ini_cuboid * ut.constant.mega_per_kilo ** 3))

            Write.write('# within encompassing cube at initial time (for MUSIC FFT)')
            Write.write('  mass = {:.2e} M_sun'.format(mass_ini_cube))
            Write.write('  volume = {:.1f} Mpc^3 comoving'.format(
                        volume_ini_cube * ut.constant.mega_per_kilo ** 3))

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
                        pos_min / part_ini.info['box.length'],
                        pos_max / part_ini.info['box.length'],
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

    def write_initial_positions_from_uniform_box(
        self, parts, hal, hal_index, distance_max=10, scale_to_halo_radius=True, virial_kind='200m',
        region_kind='convex-hull', dark_mass=None):
        '''
        Generate and write initial condition positions from a uniform-resolution DM-only
        simulation with a halo catalog.

        Parameters
        ----------
        parts : list of dicts : catalogs of particles at final and initial snapshots
        hal : dict : catalog of halos at final snapshot
        hal_index : int : index of halo
        distance_max : float : distance from center to select particles at final time
            [kpc physical or in units of R_halo]
        scale_to_halo_radius : bool : whether to scale distance to halo radius
        virial_kind : str : virial overdensity to define halo radius
        region_kind : str : method to identify zoom-in regon at initial time:
            'particles', 'convex-hull', 'cube'
        dark_mass : float : DM particle mass (if simulation has only DM, at single resolution)
        '''
        if scale_to_halo_radius:
            assert distance_max > 1 and distance_max < 30

        center_position = hal['position'][hal_index]
        halo_radius = hal['radius'][hal_index]

        self.write_initial_positions(
            parts, center_position, distance_max, scale_to_halo_radius, halo_radius, virial_kind,
            region_kind, dark_mass)

    def read_write_initial_positions_from_zoom(
        self, snapshot_redshifts=[0, 99], distance_max=7, scale_to_halo_radius=True,
        halo_radius=None, virial_kind='200m', region_kind='convex-hull', simulation_directory='.'):
        '''
        Generate and write initial condition points from a zoom-in simulation:
            (1) read particles
            (2) identify halo center
            (3) identify zoom-in region around center
            (4) write positions of particles at initial redshift

        Parameters
        ----------
        snapshot_redshifts : list : redshifts of final and initial snapshots
        distance_max : float : distance from center to select particles at final time
            [kpc physical, or in units of R_halo]
        scale_to_halo_radius : bool : whether to scale distance to halo radius
        halo_radius : float : radius of halo [kpc physical]
        virial_kind : str : virial kind to use to get halo radius (if not input halo_radius)
        region_kind : str : method to determine zoom-in regon at initial time:
            'particles', 'convex-hull', 'cube'
        simulation_directory : str : directory of simulation
        '''
        if scale_to_halo_radius:
            assert distance_max > 1 and distance_max < 30

        Read = ReadClass(snapshot_redshifts, simulation_directory)
        parts = Read.read_particles()

        center_position = ut.particle.get_center_positions(
            parts[0], 'dark', method='center-of-mass', compare_centers=True)

        self.write_initial_positions(
            parts, center_position, distance_max, scale_to_halo_radius, halo_radius, virial_kind,
            region_kind)


InitialCondition = InitialConditionClass()

#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('must specify selection radius, in terms of R_200m')

    distance_max = float(sys.argv[1])

    InitialCondition.read_write_initial_positions_from_zoom(distance_max=distance_max)
