'''
Track particles across snapshots.

@author: Andrew Wetzel

Units: unless otherwise noted, all quantities are in (combinations of):
    mass in [M_sun]
    position in [kpc comoving]
    distance and radius in [kpc physical]
    velocity in [km / s]
    time in [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import sys
import collections
import numpy as np
# local ----
import utilities as ut
from . import gizmo_io

# default directory to store particle tracking files
TRACK_DIRECTORY = 'track/'


#===================================================================================================
# utility
#===================================================================================================
class ParticleIndexPointerClass(ut.io.SayClass):
    '''
    Compute particle index pointers for tracking particles across time.
    '''

    def __init__(self, species_name='star', directory=TRACK_DIRECTORY):
        '''
        Parameters
        ----------
        species_name : string : name of particle species to track
        directory : string : directory where to write files
        '''
        self.species_name = species_name
        self.directory = directory
        self.Read = gizmo_io.ReadClass()

    def _write_index_pointers_to_snapshot(self, part_z0, snapshot_index, count_tot):
        '''
        Assign to each particle a pointer to its index in the list of particles at each previous
        snapshot, to make it easier to track particles back in time.
        Write index pointers to file.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at reference snapshot
        snapshot_index : int : snapshot index at which to assign particle index pointers
        count_tot : dict : diagnostic counters
        '''
        # read particles at this snapshot
        # need to do this first to get the exact scale-factor of snapshot
        part_z = self.Read.read_snapshots(
            self.species_name, 'index', snapshot_index,
            properties=['id', self.match_property, self.test_property], element_indices=[0],
            assign_center=False, check_properties=False)

        if self.species_name not in part_z or not len(part_z[self.species_name]['id']):
            return

        # diagnostic
        pis_multiple = ut.particle.get_indices_id_kind(part_z, self.species_name, 'multiple')
        self.say('* {} {} particles have redundant id at snapshot {}'.format(
            pis_multiple.size, self.species_name, snapshot_index))

        # initialize pointer array
        # set null values to negative and will return error if called as index
        part_index_pointers = ut.array.get_array_null(part_z0[self.species_name]['id'].size)

        count = {
            'id no match': 0,
            'match prop no match': 0,
            'match prop redundant': 0,
            'test prop offset': 0,
        }

        for part_z_index, part_z_id in enumerate(part_z[self.species_name]['id']):
            try:
                part_z0_indices = part_z0[self.species_name].id_to_index[part_z_id]
            except IndexError:
                count['id no match'] += 1
                continue

            if np.isscalar(part_z0_indices):
                # particle id is unique - easy case
                part_index_pointers[part_z0_indices] = part_z_index
            else:
                # particle id is redundant - tricky case
                # loop through particles with this id, use match_property to match
                # sanity check
                match_props = part_z0[self.species_name].prop(self.match_property, part_z0_indices)
                if np.unique(match_props).size != part_z0_indices.size:
                    count['match prop redundant'] += 1

                match_prop_z = part_z[self.species_name].prop(self.match_property, part_z_index)

                for part_z0_index in part_z0_indices:
                    match_prop_z0 = part_z0[self.species_name].prop(
                        self.match_property, part_z0_index)
                    if self.match_property == 'id.child':
                        if match_prop_z0 == match_prop_z:
                            part_index_pointers[part_z0_index] = part_z_index
                            break
                    else:
                        if (np.abs((match_prop_z0 - match_prop_z) / match_prop_z) <
                                self.match_propery_tolerance):
                            part_index_pointers[part_z0_index] = part_z_index
                            break
                else:
                    count['match prop no match'] += 1

        if count['id no match']:
            self.say('! {} {} particles not have id match at snapshot {}'.format(
                     count['id no match'], self.species_name, snapshot_index))
        if count['match prop no match']:
            self.say('! {} {} particles not have {} match at snapshot {}'.format(
                     count['match prop no match'], self.species_name, self.match_property,
                     snapshot_index))
        if count['match prop redundant']:
            self.say('! {} {} particles have redundant {} at snapshot {}'.format(
                     count['match prop redundant'], self.species_name, self.match_property,
                     snapshot_index))

        # more sanity checks

        part_z0_indices = np.where(part_index_pointers >= 0)[0]
        # ensure same number of particles assigned at z0 as in snapshot at z
        if part_z0_indices.size != part_z[self.species_name]['id'].size:
            self.say('! {} {} particles at snapshot {}'.format(
                     part_z[self.species_name]['id'].size, self.species_name, snapshot_index))
            self.say('but matched to {} particles at snapshot {}'.format(
                     part_z0_indices.size, part_z0.snapshot['index']))
        else:
            # check using test property
            if (self.test_property and self.test_property != self.match_property and
                    count['id no match'] == count['match prop no match'] == 0):
                part_index_pointers_good = part_index_pointers[part_z0_indices]
                prop_difs = np.abs(
                    (part_z[self.species_name].prop(self.test_property, part_index_pointers_good) -
                     part_z0[self.species_name].prop(self.test_property, part_z0_indices)) /
                    part_z[self.species_name].prop(self.test_property, part_index_pointers_good))
                count['test prop offset'] = np.sum(prop_difs > self.match_propery_tolerance)

                if count['test prop offset']:
                    self.say('! {} matched particles have different {} at snapshot {} v {}'.format(
                             count['test prop offset'], self.test_property, snapshot_index,
                             part_z0.snapshot['index']))

        for k in count:
            count_tot[k] += count[k]

        # write file for this snapshot
        self.io_index_pointers(None, snapshot_index, part_index_pointers)

    def write_index_pointers_to_snapshots(
        self, part=None, match_property='id.child', match_propery_tolerance=1e-6,
        test_property='form.scalefactor', snapshot_indices=[], thread_number=1):
        '''
        Assign to each particle a pointer to its index in the list of particles at each previous
        snapshot, to make it easier to track particles back in time.
        Write index pointers to file, one for each snapshot prior to z = 0.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        match_property : string :
            some particles have the same id. this is the property to use to match them.
            options (in order of preference): 'id.child', 'massfraction.metals', 'form.scalefactor'
        match_propery_tolerance : float : fractional tolerance for matching via match_property
        test_property : string : additional property to use to test matching
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        thread_number : int : number of threads for parallelization
        '''
        assert match_property in ['id.child', 'massfraction.metals', 'form.scalefactor']

        if part is None:
            # read particles at z = 0
            # get list of properties relevant to use in matching
            properties_read = ['id', 'id.child']
            if match_property not in properties_read:
                properties_read.append(match_property)
            if test_property and test_property not in properties_read:
                properties_read.append(test_property)
            part = self.Read.read_snapshots(
                self.species_name, 'redshift', 0, properties=properties_read, element_indices=[0],
                assign_center=False, check_properties=False)

        # older snapshot files do not have id.child - use abundance of total metals instead
        if match_property == 'id.child' and 'id.child' not in part[self.species_name]:
            self.say('input match_property = {} does not exist in snapshot {}'.format(
                match_property, part.snapshot['index']))
            match_property = 'massfraction.metals'
            self.say('switching to using: {}'.format(match_property))
            if match_property not in properties_read:
                properties_read.append(match_property)
                part = self.Read.read_snapshots(
                    self.species_name, 'redshift', 0, properties=properties_read,
                    element_indices=[0], assign_center=False, check_properties=False)

        assert part[self.species_name].prop(match_property) is not None

        if test_property:
            assert part[self.species_name].prop(test_property) is not None

        # get list of snapshot indices to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
        snapshot_indices = np.setdiff1d(snapshot_indices, part.snapshot['index'])  # skip current
        snapshot_indices = snapshot_indices[::-1]  # work backwards in time

        # diagnostic
        pis_multiple = ut.particle.get_indices_id_kind(part, self.species_name, 'multiple')
        self.say('* {} {} particles have redundant id'.format(pis_multiple.size, self.species_name))

        # assign pointer from particle id to its index in list
        ut.particle.assign_id_to_index(
            part, self.species_name, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

        self.match_property = match_property
        self.match_propery_tolerance = match_propery_tolerance
        self.test_property = test_property

        # counters for sanity checks
        count = {
            'id no match': 0,
            'match prop no match': 0,
            'match prop redundant': 0,
            'test prop offset': 0,
        }

        # initiate threads, if asking for > 1
        if thread_number > 1:
            import multiprocessing as mp
            pool = mp.Pool(thread_number)

        for snapshot_index in snapshot_indices:
            if thread_number > 1:
                pool.apply_async(
                    self._write_index_pointers_to_snapshot, (part, snapshot_index, count))
            else:
                self._write_index_pointers_to_snapshot(part, snapshot_index, count)

        # close threads
        if thread_number > 1:
            pool.close()
            pool.join()

        # print cumulative diagnostics
        if count['id no match']:
            self.say('! {} total not have id match!'.format(count['id no match']))
        if count['match prop no match']:
            self.say('! {} total not have {} match!'.format(
                count['match prop no match'], match_property))
        if count['match prop redundant']:
            self.say('! {} total have redundant {}!'.format(
                count['match prop redundant'], match_property))
        if count['test prop offset']:
            self.say('! {} total have offset {}'.format(count['test prop offset'], test_property))

    def io_index_pointers(
        self, part=None, snapshot_index=None, part_index_pointers=None, directory=None):
        '''
        Read or write, for each star particle, its index in the catalog at another snapshot.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        snapshot_index : int : index of snapshot file (if not input particle catalog)
        part_index_pointers : array : particle index pointers (if writing)
        '''
        hdf5_dict_name = 'indices'

        if part is not None:
            snapshot_index = part.snapshot['index']
        elif not snapshot_index:
            raise ValueError('! need to input either particle catalog or snapshot_index')

        file_name = '{}_indices_{:03d}'.format(self.species_name, snapshot_index)

        if directory is None:
            directory = self.directory

        if part_index_pointers is not None:
            # write to file
            directory = ut.io.get_path(directory, create_path=True)
            ut.io.file_hdf5(directory + file_name, {hdf5_dict_name: part_index_pointers})
        else:
            # read from file
            directory = ut.io.get_path(directory)
            dict_in = ut.io.file_hdf5(directory + file_name)
            part_index_pointers = dict_in[hdf5_dict_name]
            if part is None:
                return part_index_pointers
            else:
                part.index_pointers = part_index_pointers

    def get_reverse_index_pointers(self, part_index_pointers):
        '''
        Given input index pointers from z_reference to z, get index pointers from z to z_reference.

        Parameters
        ----------
        part_index_pointers : array : particle index pointers, from z_reference to z

        Returns
        -------
        part_reverse_index_pointers : array : particle index pointers, from z to z_reference
        '''
        # pointers from z_reference to z that have valid (non-null) values
        masks_valid = (part_index_pointers >= 0)
        part_index_pointers_valid = part_index_pointers[masks_valid]

        part_number_at_z_ref = part_index_pointers.size
        part_number_at_z = part_index_pointers_valid.size

        part_reverse_index_pointers = ut.array.get_array_null(part_number_at_z)
        part_reverse_index_pointers[part_index_pointers_valid] = (
            ut.array.get_arange(part_number_at_z_ref)[masks_valid])

        return part_reverse_index_pointers


ParticleIndexPointer = ParticleIndexPointerClass()


class ParticleCoordinateClass(ParticleIndexPointerClass):
    '''
    Compute coordinates (3D distances and 3D velocities) wrt the host galaxy center for particles
    across time.
    '''

    def __init__(self, species_name='star', directory=TRACK_DIRECTORY):
        '''
        Parameters
        ----------
        species : string : name of particle species to track
        directory : string : directory to write files
        '''
        self.species_name = species_name
        self.directory = directory
        self.Read = gizmo_io.ReadClass()

        # set numpy data type to store coordinates
        self.coordinate_dtype = np.float32
        if self.coordinate_dtype is np.float32:
            self.force_float32 = True
        else:
            self.force_float32 = False

        # distance limits to select particles associated with primary host at z0
        # use only these particle to define progenitor center at higher redshifts
        self.host_z0_distance_limits = [0, 100]  # [kpc physical]

        # names of distances and velocities to write/read
        self.form_host_coordiante_kinds = ['form.host.distance', 'form.host.velocity']

    def _write_formation_coordinates(
        self, part_z0, snapshot_index, count_tot):
        '''
        Assign to each particle its coordinates (position and velocity) wrt its primary host.
        Write to file.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at reference snapshot
        snapshot_index : int : snapshot index at which to assign particle index pointers
        count_tot : dict : diagnostic counters
        '''
        part_z0_indices = ut.array.get_arange(part_z0[self.species_name]['id'])
        part_z0_indices_host = ut.array.get_indices(
            part_z0[self.species_name].prop('host.distance.total'), self.host_z0_distance_limits)

        if snapshot_index == part_z0.snapshot['index']:
            part_index_pointers = part_z0_indices
        else:
            try:
                part_index_pointers = self.io_index_pointers(snapshot_index=snapshot_index)
            except Exception:
                return

        part_z0_indices = part_z0_indices[part_index_pointers >= 0]
        self.say('\n# {} to assign at snapshot {}'.format(part_z0_indices.size, snapshot_index))

        count = {
            'id none': 0,
            'id wrong': 0,
        }

        if part_z0_indices.size > 0:
            part_z = self.Read.read_snapshots(
                self.species_name, 'index', snapshot_index,
                properties=['position', 'velocity', 'mass', 'id', 'form.scalefactor'],
                force_float32=self.force_float32,
                assign_center=False, check_properties=True)

            # limit progenitor center to those particle that end up near host at z0
            self.Read.assign_center(part_z, self.species_name, part_z0_indices_host)

            part_z_indices = part_index_pointers[part_z0_indices]

            # sanity checks
            masks = (part_z_indices >= 0)
            count['id none'] = part_z_indices.size - np.sum(masks)
            if count['id none']:
                self.say('! {} have no id match at snapshot {}!'.format(
                         count['id none'], snapshot_index))
                part_z_indices = part_z_indices[masks]
                part_z0_indices = part_z0_indices[masks]

            masks = (part_z0[self.species_name]['id'][part_z0_indices] ==
                     part_z[self.species_name]['id'][part_z_indices])
            count['id wrong'] = part_z_indices.size - np.sum(masks)
            if count['id wrong']:
                self.say('! {} have wrong id match at snapshot {}!'.format(
                         count['id wrong'], snapshot_index))
                part_z_indices = part_z_indices[masks]
                part_z0_indices = part_z0_indices[masks]

            # compute host galaxy properties, including R_90
            gal = ut.particle.get_galaxy_properties(
                part_z, self.species_name, 'mass.percent', 90, distance_max=15,
                print_results=True)

            if not gal['radius'] or np.isnan(gal['radius']):
                self.say('! no ')
                return

            # compute rotation vectors for principal axes from young stars within R_90
            rotation_vectors, _ev, _ar = ut.particle.get_principal_axes(
                part_z, self.species_name, gal['radius'], age_percent=30)

            # store host galaxy center coordinates
            part_z0[self.species_name].center_position_at_snapshots[snapshot_index] = (
                part_z.center_position)
            part_z0[self.species_name].center_velocity_at_snapshots[snapshot_index] = (
                part_z.center_velocity)

            # store rotation vectors
            part_z0[self.species_name].principal_axes_vectors_at_snapshots[snapshot_index] = (
                rotation_vectors)

            # compute coordinates
            coordinates = {}
            prop = 'form.host.distance'
            if prop in self.form_host_coordiante_kinds:
                # 3-D distance wrt host in simulation's cartesian coordinates [kpc physical]
                coordinates[prop] = ut.coordinate.get_distances(
                    part_z[self.species_name]['position'][part_z_indices],
                    part_z.center_position, part_z.info['box.length'],
                    part_z.snapshot['scalefactor'])

            prop = 'form.host.velocity'
            if prop in self.form_host_coordiante_kinds:
                # 3-D velocity wrt host in simulation's cartesian coordinates [km / s]
                coordinates[prop] = ut.coordinate.get_velocity_differences(
                    part_z[self.species_name]['velocity'][part_z_indices],
                    part_z.center_velocity,
                    part_z[self.species_name]['position'][part_z_indices],
                    part_z.center_position, part_z.info['box.length'],
                    part_z.snapshot['scalefactor'], part_z.snapshot['time.hubble'])

            # rotate coordinates to align with principal axes
            for prop in self.form_host_coordiante_kinds:
                coordinates[prop] = ut.coordinate.get_coordinates_rotated(
                    coordinates[prop], rotation_vectors)

            # assign 3-D coordinates wrt host along principal axes [kpc physical]
            for prop in self.form_host_coordiante_kinds:
                part_z0[self.species_name][prop][part_z0_indices] = coordinates[prop]

            for k in count:
                count_tot[k] += count[k]

            # continuously (re)write as go
            self.io_formation_coordinates(part_z0, 'write')

    def write_formation_coordinates(self, part_z0=None, snapshot_indices=[], thread_number=1):
        '''
        Assign to each particle its coordiates (3D distances and 3D velocities) wrt its primary host
        galaxy center at the snapshot after it formed.

        Parameters
        ----------
        part : dict : catalog of particles at reference snapshot
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        thread_number : int : number of threads for parallelization
        '''
        if part_z0 is None:
            # read particles at z = 0
            part_z0 = self.Read.read_snapshots(
                self.species_name, 'redshift', 0,
                properties=['position', 'velocity', 'mass', 'id', 'id.child', 'form.scalefactor'],
                element_indices=[0], force_float32=self.force_float32, assign_center=False,
                check_properties=False)

        part_z0_indices = ut.array.get_arange(part_z0[self.species_name]['id'])

        # get list of snapshots to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part_z0.Snapshot['index']), max(part_z0.Snapshot['index']) + 1)
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # store particle properties and initialize to nan
        for prop in self.form_host_coordiante_kinds:
            part_z0[self.species_name][prop] = np.zeros(
                [part_z0_indices.size, 3], self.coordinate_dtype) + np.nan

        # store center position and velocity of the host galaxy at each snapshot
        part_z0[self.species_name].center_position_at_snapshots = (
            np.zeros([snapshot_indices.size, 3], self.coordinate_dtype) + np.nan)
        part_z0[self.species_name].center_velocity_at_snapshots = (
            np.zeros([snapshot_indices.size, 3], self.coordinate_dtype) + np.nan)

        # store principal axes rotation vectors of the host galaxy at each snapshot
        part_z0[self.species_name].principal_axes_vectors_at_snapshots = (
            np.zeros([snapshot_indices.size, 3, 3], self.coordinate_dtype) + np.nan)

        count = {
            'id none': 0,
            'id wrong': 0,
        }

        # initiate threads, if asking for > 1
        if thread_number > 1:
            import multiprocessing as mp
            pool = mp.Pool(thread_number)

        for snapshot_index in snapshot_indices:
            if thread_number > 1:
                pool.apply(
                    self._write_formation_coordinates, (part_z0, snapshot_index, count))
            else:
                self._write_formation_coordinates(part_z0, snapshot_index, count)

        # close threads
        if thread_number > 1:
            pool.close()
            pool.join()

        # print cumulative diagnostics
        if count['id none']:
            self.say('! {} total do not have valid id!'.format(count['id none']))
        if count['id wrong']:
            self.say('! {} total not have id match!'.format(count['id wrong']))

    def io_formation_coordinates(self, part, io_direction='read'):
        '''
        Read or write, for each particle, at the first snapshot after it formed,
        its coordinates (distances and velocities) wrt the host galaxy center,
        aligned with the principal axes of the host galaxy at that time.
        If read, assign to particle catalog.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        io_direction : string : 'read' or 'write'
        '''
        assert io_direction in ('read', 'write')

        file_name = '{}_form_coordinates_{:03d}'.format(self.species_name, part.snapshot['index'])

        if io_direction == 'write':
            directory = ut.io.get_path(self.directory, create_path=True)
            dict_out = collections.OrderedDict()
            dict_out['id'] = part[self.species_name]['id']
            for prop in self.form_host_coordiante_kinds:
                dict_out[prop] = part[self.species_name][prop]
            dict_out['center.position'] = part[self.species_name].center_position_at_snapshots
            dict_out['center.velocity'] = part[self.species_name].center_velocity_at_snapshots
            dict_out['principal.axes.vectors'] = (
                part[self.species_name].principal_axes_vectors_at_snapshots)

            ut.io.file_hdf5(directory + file_name, dict_out)

        elif io_direction == 'read':
            directory = ut.io.get_path(self.directory)
            dict_in = ut.io.file_hdf5(directory + file_name)

            # sanity check
            bad_id_number = np.sum(part[self.species_name]['id'] != dict_in['id'])
            if bad_id_number:
                self.say('! {} particles have mismatched id - bad!'.format(bad_id_number))

            for prop in dict_in.keys():
                if prop == 'id':
                    pass
                elif prop == 'center.position':
                    part[self.species_name].center_position_at_snapshots = dict_in[prop]
                elif prop == 'center.velocity':
                    part[self.species_name].center_velocity_at_snapshots = dict_in[prop]
                elif prop == 'principal.axes.vectors':
                    part[self.species_name].principal_axes_vectors_at_snapshots = dict_in[prop]
                else:
                    # store coordinates at formation
                    part[self.species_name][prop] = dict_in[prop]

            # fix naming convention in older files - keep only 3-D vector distance
            for prop in self.form_host_coordiante_kinds:
                if prop + '.3d' in part[self.species_name]:
                    part[self.species_name][prop] = part[self.species_name][prop + '.3d']
                    del(part[self.species_name][prop + '.3d'])


ParticleCoordinate = ParticleCoordinateClass()

#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('specify function: indices, coordinates, indices+coordinates')

    function_kind = str(sys.argv[1])

    assert ('indices' in function_kind or 'coordinates' in function_kind)

    if 'indices' in function_kind:
        ParticleIndexPointer.write_index_pointers_to_snapshots()

    if 'coordinates' in function_kind:
        ParticleCoordinate.write_formation_coordinates()
