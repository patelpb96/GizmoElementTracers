#!/usr/bin/env python3

'''
Track particles across snapshots.

@author: Andrew Wetzel <arwetzel@gmail.com>

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

    def __init__(
        self, species_name='star', directory=TRACK_DIRECTORY, reference_snapshot_index=600):
        '''
        Parameters
        ----------
        species_name : str : name of particle species to track
        directory : str : directory where to write files
        reference_snapshot_index : int :
            reference snapshot to compute particle index pointers relative to
        '''
        self.species_name = species_name
        self.directory = directory
        self.Read = gizmo_io.ReadClass()
        self.reference_snapshot_index = reference_snapshot_index

    def _write_pointers_to_snapshot(self, part_z0, snapshot_index, count_tot):
        '''
        Assign to each particle a pointer from its index at the reference snapshot (usually z = 0)
        to its index at another snapshot_index.
        Write the particle index pointers to a file.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at reference snapshot (usually z = 0)
        snapshot_index : int : snapshot index to assign particle index pointers to
        count_tot : dict : diagnostic counters
        '''
        # read particles at this snapshot
        # need to do this first to get the exact scale-factor of snapshot
        part_z = self.Read.read_snapshots(
            self.species_name, 'index', snapshot_index,
            properties=['id', self.match_property, self.test_property], element_indices=[0],
            assign_host_coordinates=False, check_properties=False)

        if self.species_name not in part_z or not len(part_z[self.species_name]['id']):
            return

        # diagnostic
        pis_multiple = ut.particle.get_indices_id_kind(part_z, self.species_name, 'multiple')
        self.say('* {} {} particles have redundant id at snapshot {}'.format(
            pis_multiple.size, self.species_name, snapshot_index))

        # initialize pointer array
        # set null values to negative and will return error if called as index
        part_pointers = ut.array.get_array_null(part_z0[self.species_name]['id'].size)

        count = {
            'id no match': 0,
            'match prop no match': 0,
            'match prop redundant': 0,
            'test prop offset': 0,
        }

        for part_z_index, part_z_id in enumerate(part_z[self.species_name]['id']):
            try:
                part_z0_indices = part_z0[self.species_name].id_to_index[part_z_id]
            except (IndexError, KeyError):
                count['id no match'] += 1
                continue

            if np.isscalar(part_z0_indices):
                # particle id is unique - easy case
                part_pointers[part_z0_indices] = part_z_index
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
                            part_pointers[part_z0_index] = part_z_index
                            break
                    else:
                        if (np.abs((match_prop_z0 - match_prop_z) / match_prop_z) <
                                self.match_propery_tolerance):
                            part_pointers[part_z0_index] = part_z_index
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

        part_z0_indices = np.where(part_pointers >= 0)[0]
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
                part_pointers_good = part_pointers[part_z0_indices]
                prop_difs = np.abs(
                    (part_z[self.species_name].prop(self.test_property, part_pointers_good) -
                     part_z0[self.species_name].prop(self.test_property, part_z0_indices)) /
                    part_z[self.species_name].prop(self.test_property, part_pointers_good))
                count['test prop offset'] = np.sum(prop_difs > self.match_propery_tolerance)

                if count['test prop offset']:
                    self.say('! {} matched particles have different {} at snapshot {} v {}'.format(
                             count['test prop offset'], self.test_property, snapshot_index,
                             part_z0.snapshot['index']))

        for k in count:
            count_tot[k] += count[k]

        # write file for this snapshot
        self.io_pointers(None, snapshot_index, part_pointers)

    def write_pointers_to_snapshots(
        self, part=None, match_property='id.child', match_propery_tolerance=1e-6,
        test_property='form.scalefactor', snapshot_indices=[], thread_number=1):
        '''
        Assign to each particle a pointer from its index at the reference snapshot (usually z = 0)
        to its index at all other snapshots, to make it easier to track particles across time.
        Write particle index pointers to file, one file for each snapshot beyond the
        reference snapshot.

        Parameters
        ----------
        part : dict : catalog of particles at reference snapshot (usually z = 0)
        match_property : str :
            some particles have the same id. this is the property to use to match them.
            options (in order of preference): 'id.child', 'massfraction.metals', 'form.scalefactor'
        match_propery_tolerance : float : fractional tolerance for matching via match_property
        test_property : str : additional property to use to test matching
        snapshot_indices : array-like : snapshot indices at which to assign index pointers
        thread_number : int : number of threads for parallelization
        '''
        assert match_property in ['id.child', 'massfraction.metals', 'form.scalefactor']

        if part is None:
            # read particles at reference snapshot (typically z = 0)
            # get list of properties relevant to use in matching
            properties_read = ['id', 'id.child']
            if match_property not in properties_read:
                properties_read.append(match_property)
            if test_property and test_property not in properties_read:
                properties_read.append(test_property)
            part = self.Read.read_snapshots(
                self.species_name, 'index', self.reference_snapshot_index,
                properties=properties_read, element_indices=[0], assign_host_coordinates=False,
                check_properties=False)

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
                    element_indices=[0], assign_host_coordinates=False, check_properties=False)

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
            part, self.species_name, 'id', 0, store_as_dict=True, print_diagnostic=False)

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
                    self._write_pointers_to_snapshot, (part, snapshot_index, count))
            else:
                self._write_pointers_to_snapshot(part, snapshot_index, count)

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

    def io_pointers(self, part=None, snapshot_index=None, part_pointers=None, directory=None):
        '''
        Read or write, for each star particle at the reference snapshot (usually z = 0),
        its index at another snapshot.
        If input particle catalog (part), add index pointers to its dictionary class,
        else return index pointers as an array.

        Parameters
        ----------
        part : dict : catalog of particles at a snapshot
        snapshot_index : int : index of snapshot (if not input part)
        part_pointers : array : particle index pointers (if writing)

        Returns
        -------
        part_pointers : array :
            particle index pointers from reference snapshot (usually z = 0) to another snapshot
        '''
        hdf5_dict_name = 'indices'

        if part is not None:
            snapshot_index = part.snapshot['index']
        elif not snapshot_index:
            raise ValueError('! need to input either particle catalog or snapshot_index')

        file_name = '{}_indices_{:03d}'.format(self.species_name, snapshot_index)

        if directory is None:
            directory = self.directory

        if part_pointers is not None:
            # write to file
            directory = ut.io.get_path(directory, create_path=True)
            ut.io.file_hdf5(directory + file_name, {hdf5_dict_name: part_pointers})
        else:
            # read from file
            directory = ut.io.get_path(directory)
            dict_in = ut.io.file_hdf5(directory + file_name)
            part_pointers = dict_in[hdf5_dict_name]
            if part is None:
                return part_pointers
            else:
                part.index_pointers = part_pointers

    def get_pointers_reverse(self, part_pointers=None, snapshot_index=None):
        '''
        Given input particle index pointers from the reference snapshot (usually z = 0) to another
        snapshot, get 'reverse' index pointers from the other snapshot to the reference snapshot.

        Parameters
        ----------
        part_pointers : array :
            particle index pointers from the reference snapshot (usually z = 0) to another napshot
        snapshot_index : int : index of snapshot to read pointers (if not input part_pointers)

        Returns
        -------
        part_reverse_pointers : array :
            particle index pointers from the other snapshot to the reference snapshot
        '''
        if part_pointers is None:
            # read from file
            part_pointers = self.io_pointers(snapshot_index=snapshot_index)

        # get index pointers that have valid (non-null) values
        masks_valid = (part_pointers >= 0)
        part_pointers_valid = part_pointers[masks_valid]

        part_number_at_z_ref = part_pointers.size
        part_number_at_z = part_pointers_valid.size

        # sanity check
        if part_pointers_valid.max() >= part_number_at_z:
            self.say('! input part_pointers has {} valid pointers'.format(part_number_at_z))
            self.say('but its pointer index max = {}'.format(part_pointers_valid.max()))
            self.say('thus it does not point to all particles at snapshot at z')
            self.say('increasing size of reverse pointer array to accomodate missing particles')
            part_number_at_z = part_pointers_valid.max() + 1

        part_reverse_pointers = ut.array.get_array_null(part_number_at_z)
        part_reverse_pointers[part_pointers_valid] = (
            ut.array.get_arange(part_number_at_z_ref)[masks_valid])

        return part_reverse_pointers

    def get_pointers_between_snapshots(self, snapshot_index_from, snapshot_index_to):
        '''
        Get particle index pointers between any two snapshots.
        Given input snapshot indices, get array of index pointers from snapshot_index_from to
        snapshot_index_to.

        Parameters
        ----------
        snapshot_index_from : int : snapshot index to get index pointers from
        snapshot_index_to : int : snapshot index to get pointers to

        Returns
        -------
        part_pointers : array :
            particle index pointers from snapshot_index_from to snapshot_index_to
        '''
        if snapshot_index_from == self.reference_snapshot_index:
            part_pointers = self.io_pointers(snapshot_index=snapshot_index_to)
        elif snapshot_index_to == self.reference_snapshot_index:
            part_pointers_from = self.io_pointers(snapshot_index=snapshot_index_from)
            part_pointers = self.get_pointers_reverse(part_pointers_from)
        else:
            part_pointers_from = self.io_pointers(snapshot_index=snapshot_index_from)
            part_pointers_to = self.io_pointers(snapshot_index=snapshot_index_to)
            part_reverse_pointers_from = self.get_pointers_reverse(part_pointers_from)
            part_pointers = part_pointers_to[part_reverse_pointers_from]

        return part_pointers


ParticleIndexPointer = ParticleIndexPointerClass()


class ParticleCoordinateClass(ParticleIndexPointerClass):
    '''
    Compute coordinates (3-D distances and 3-D velocities) wrt each primary host galaxy for all
    particles at the snapshot immediately after they form.
    '''

    def __init__(
        self, species_name='star', directory=TRACK_DIRECTORY, reference_snapshot_index=600,
        host_distance_limits=[0, 50]):
        '''
        Parameters
        ----------
        species : str : name of particle species to track
        directory : str : directory to write files
        reference_snapshot_index : float :
            reference snapshot to compute particle index pointers relative to
        host_distance_limits : list :
            min and max distance [kpc physical] to select particles near the primary host[s] at the
            reference snapshot; use only these particles to find host center[s] at earlier snapshots
        '''
        self.species_name = species_name
        self.directory = directory
        self.reference_snapshot_index = reference_snapshot_index
        self.host_distance_limits = host_distance_limits

        self.Read = gizmo_io.ReadClass()

        # set numpy data type to store coordinates
        self.coordinate_dtype = np.float32

        # names of distances and velocities to write/read
        self.form_host_coordiante_kinds = ['form.host.distance', 'form.host.velocity']

    def _write_formation_coordinates(
        self, part_z0, hosts_part_z0_indices, host_number, snapshot_index, count_tot):
        '''
        Assign to each particle its coordinates (position and velocity) wrt the primary host.
        Write to file.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at the reference snapshot
        hosts_part_z0_indices : array :
            indices of particles near all primary host[s] at the reference snapshot
        host_number : int : number of host galaxies to assign and compute coordinates relative to
        snapshot_index : int : snapshot index at which to assign particle index pointers
        count_tot : dict : diagnostic counters
        '''
        part_z0_indices = ut.array.get_arange(part_z0[self.species_name]['id'])

        if snapshot_index == part_z0.snapshot['index']:
            part_pointers = part_z0_indices
        else:
            try:
                part_pointers = self.io_pointers(snapshot_index=snapshot_index)
            except Exception:
                return

        part_z0_indices = part_z0_indices[part_pointers >= 0]
        self.say('\n# {} to assign at snapshot {}'.format(part_z0_indices.size, snapshot_index))

        count = {
            'id none': 0,
            'id wrong': 0,
        }

        if part_z0_indices.size > 0:
            part_z = self.Read.read_snapshots(
                self.species_name, 'index', snapshot_index,
                properties=['position', 'velocity', 'mass', 'id', 'form.scalefactor'],
                assign_host_coordinates=False, check_properties=True)

            # limit the coordinates of progenitor[s] of primary host[s] to particles that are near
            # the primary host[s] at the reference snapshot
            hosts_part_z_indices = part_pointers[hosts_part_z0_indices]
            self.Read.assign_host_coordinates(
                part_z, self.species_name, hosts_part_z_indices[hosts_part_z_indices >= 0],
                host_number=host_number)

            part_z_indices = part_pointers[part_z0_indices]

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

            # store host galaxy coordinates
            part_z0[self.species_name].host_positions_at_snapshots[snapshot_index] = (
                part_z.host_positions)
            part_z0[self.species_name].host_velocities_at_snapshots[snapshot_index] = (
                part_z.host_velocities)

            # compute rotation vectors for principal axes from young stars within R_90
            self.Read.assign_host_principal_axes(part_z)

            #galaxy_radius_max = 15  # [kpc physical]
            #principal_axes = ut.particle.get_principal_axes(
            #    part_z, self.species_name, galaxy_radius_max, mass_percent=90, age_percent=30,
            #    center_positions=part_z.host_positions, return_array=False)

            # store rotation vectors
            part_z0[self.species_name].host_rotation_tensors_at_snapshots[snapshot_index] = (
                part_z.host_rotation_tensors)

            for host_i in range(host_number):
                # compute coordinates wrt primary host
                host_name = ut.catalog.get_host_name(host_i)

                for prop in self.form_host_coordiante_kinds:
                    prop = prop.replace('host.', host_name)

                    if 'distance' in prop:
                        # 3-D distance wrt host in simulation's cartesian coordinates [kpc physical]
                        coordinates = ut.coordinate.get_distances(
                            part_z[self.species_name]['position'][part_z_indices],
                            part_z.host_positions[host_i], part_z.info['box.length'],
                            part_z.snapshot['scalefactor'])

                    elif 'velocity'in prop:
                        # 3-D velocity wrt host in simulation's cartesian coordinates [km / s]
                        coordinates = ut.coordinate.get_velocity_differences(
                            part_z[self.species_name]['velocity'][part_z_indices],
                            part_z.host_velocities[host_i],
                            part_z[self.species_name]['position'][part_z_indices],
                            part_z.host_positions[host_i], part_z.info['box.length'],
                            part_z.snapshot['scalefactor'], part_z.snapshot['time.hubble'])

                    # rotate coordinates to align with principal axes
                    coordinates = ut.coordinate.get_coordinates_rotated(
                        coordinates, part_z.host_rotation_tensors[host_i])

                    # assign 3-D coordinates wrt primary host along principal axes [kpc physical]
                    part_z0[self.species_name][prop][part_z0_indices] = coordinates

                for k in count:
                    count_tot[k] += count[k]

            # continuously (re)write as go
            self.io_formation_coordinates(part_z0, write=True)

    def write_formation_coordinates(self, part_z0=None, host_number=1, thread_number=1):
        '''
        Assign to each particle its coordiates (3-D distance and 3-D velocity) wrt each primary
        host galaxy at the snapshot after it formed.

        Parameters
        ----------
        part : dict : catalog of particles at the reference snapshot
        host_number : int : number of host galaxies to assign and compute coordinates relative to
        thread_number : int : number of threads for parallelization
        '''
        if part_z0 is None:
            # read particles at z = 0
            part_z0 = self.Read.read_snapshots(
                self.species_name, 'index', self.reference_snapshot_index,
                properties=['position', 'velocity', 'mass', 'id', 'id.child', 'form.scalefactor'],
                element_indices=[0], host_number=host_number, assign_host_coordinates=True,
                check_properties=False)

        # get list of snapshots to assign
        snapshot_indices = np.arange(
            min(part_z0.Snapshot['index']), max(part_z0.Snapshot['index']) + 1)
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # store position and velocity of the primary host galaxy[s] at each snapshot
        part_z0[self.species_name].host_positions_at_snapshots = np.zeros(
            [part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype) + np.nan
        part_z0[self.species_name].host_velocities_at_snapshots = np.zeros(
            [part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype) + np.nan

        # store principal axes rotation tensor of the primary host galaxy[s] at each snapshot
        part_z0[self.species_name].host_rotation_tensors_at_snapshots = np.zeros(
            [part_z0.Snapshot['index'].size, host_number, 3, 3], self.coordinate_dtype) + np.nan

        # store indices of particles near all primary hosts at z0
        hosts_part_z0_indices = np.zeros(
            0, dtype=ut.array.parse_data_type(part_z0[self.species_name]['id'].size))

        for host_index in range(host_number):
            host_name = ut.catalog.get_host_name(host_index)

            part_z0_indices = ut.array.get_indices(
                part_z0[self.species_name].prop(host_name + 'distance.total'),
                self.host_distance_limits)
            hosts_part_z0_indices = np.concatenate((hosts_part_z0_indices, part_z0_indices))

            # store particle formation coordinate properties, initialize to nan
            for prop in self.form_host_coordiante_kinds:
                prop = prop.replace('host.', host_name)  # update host name (if necessary)
                part_z0[self.species_name][prop] = np.zeros(
                    part_z0[self.species_name]['position'].shape, self.coordinate_dtype) + np.nan

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
                    self._write_formation_coordinates,
                    (part_z0, hosts_part_z0_indices, host_number, snapshot_index, count))
            else:
                self._write_formation_coordinates(
                    part_z0, hosts_part_z0_indices, host_number, snapshot_index, count)

        # close threads
        if thread_number > 1:
            pool.close()
            pool.join()

        # print cumulative diagnostics
        if count['id none']:
            self.say('! {} total do not have valid id!'.format(count['id none']))
        if count['id wrong']:
            self.say('! {} total not have id match!'.format(count['id wrong']))

    def io_formation_coordinates(self, part, write=False):
        '''
        Read or write, for each particle, at the first snapshot after it formed,
        its coordinates (3-D distances and 3-D velocities) wrt the host galaxy center,
        aligned with the principal axes of the host galaxy at that time.
        If read, assign to particle catalog.

        Parameters
        ----------
        part : dict : catalog of particles at a snapshot
        write : bool : whether to write to file (instead of read)
        '''
        file_name = '{}_form_coordinates_{:03d}'.format(
            self.species_name, part.snapshot['index'])

        if write:
            directory = ut.io.get_path(self.directory, create_path=True)
            dict_out = collections.OrderedDict()
            dict_out['id'] = part[self.species_name]['id']
            for prop in part[self.species_name]:
                if 'form.host' in prop:
                    dict_out[prop] = part[self.species_name][prop]
            dict_out['host.positions'] = part[self.species_name].host_positions_at_snapshots
            dict_out['host.velocities'] = part[self.species_name].host_velocities_at_snapshots
            dict_out['host.rotation.tensors'] = (
                part[self.species_name].host_rotation_tensors_at_snapshots)

            ut.io.file_hdf5(directory + file_name, dict_out)

        else:
            directory = ut.io.get_path(self.directory)
            dict_in = ut.io.file_hdf5(directory + file_name)

            # sanity check
            bad_id_number = np.sum(part[self.species_name]['id'] != dict_in['id'])
            if bad_id_number:
                self.say('! {} particles have mismatched id - bad!'.format(bad_id_number))

            for prop in dict_in.keys():
                if prop == 'id':
                    pass
                elif prop in ['host.positions', 'center.position']:
                    if np.ndim(dict_in[prop]) == 2:
                        dict_in[prop] = np.array([dict_in[prop]])  # deal with older files
                    part[self.species_name].host_positions_at_snapshots = dict_in[prop]
                elif prop in ['host.velocities', 'center.velocity']:
                    if np.ndim(dict_in[prop]) == 2:
                        dict_in[prop] = np.array([dict_in[prop]])  # deal with older files
                    part[self.species_name].host_velocities_at_snapshots = dict_in[prop]
                elif prop in ['host.rotation.tensors', 'principal.axes.vectors']:
                    if np.ndim(dict_in[prop]) == 3:
                        dict_in[prop] = np.array([dict_in[prop]])  # deal with older files
                    part[self.species_name].host_rotation_tensors_at_snapshots = dict_in[prop]
                else:
                    # store coordinates at formation
                    part[self.species_name][prop] = dict_in[prop]


ParticleCoordinate = ParticleCoordinateClass()

#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function: indices, coordinates, indices+coordinates')

    function_kind = str(sys.argv[1])

    assert ('indices' in function_kind or 'coordinates' in function_kind)

    if 'indices' in function_kind:
        ParticleIndexPointer.write_pointers_to_snapshots()

    if 'coordinates' in function_kind:
        ParticleCoordinate.write_formation_coordinates()
