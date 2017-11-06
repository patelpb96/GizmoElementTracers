'''
Track particles across snapshots.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import sys
import collections
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut
from . import gizmo_io


# default directory to store particle tracking files
TRACK_DIRECTORY = 'track/'


#===================================================================================================
# utility
#===================================================================================================
class IndexPointerClass(ut.io.SayClass):
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

    def write_index_pointer(
        self, part=None, match_property='id.child', match_propery_tolerance=1e-6,
        test_property='form.scalefactor', snapshot_indices=[]):
        '''
        Assign to each particle a pointer to its index in the list of particles at each previous
        snapshot, to make it easier to track particles back in time.
        Write index pointers to a pickle file, one for each previous snapshot.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        match_property : string :
            some particles have the same id. this is the property to use to match them.
            options (in order of preference): 'id.child', 'massfraction.metals', 'form.scalefactor'
        match_propery_tolerance : float :
            tolerance for matching via match_property, if it is a float
        test_property : string : additional property to use to test matching
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        '''
        assert match_property in ['id.child', 'massfraction.metals', 'form.scalefactor']

        if part is None:
            # read particles at z = 0
            # complete list of all possible properties relevant to use in matching
            properties_read = ['id', 'id.child', 'massfraction.metals', 'form.scalefactor']
            if match_property not in properties_read:
                properties_read.append(match_property)
            if test_property and test_property not in properties_read:
                properties_read.append(test_property)
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                self.species_name, 'redshift', 0, properties=properties_read, element_indices=[0],
                assign_center=False, check_properties=False)

        # older snapshot files do not have id.child - use abundance of total metals instead
        if match_property == 'id.child' and 'id.child' not in part[self.species_name]:
            self.say('input match_property = {}, but it does not exist in the snapshot'.format(
                     match_property))
            match_property = 'massfraction.metals'
            self.say('switching to using: {}'.format(match_property))

        assert part[self.species_name].prop(match_property) is not None

        if test_property:
            assert part[self.species_name].prop(test_property) is not None

        # get list of snapshot indices to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
        snapshot_indices = np.setdiff1d(snapshot_indices, part.snapshot['index'])  # skip current
        snapshot_indices = snapshot_indices[::-1]  # work backwards in time

        # particles to assign to
        part_indices = ut.array.get_arange(part[self.species_name]['id'])

        # diagnostic
        pis_multiple = ut.particle.get_indices_id_kind(
            part, self.species_name, 'multiple', part_indices)
        self.say('{} particles have redundant id'.format(pis_multiple.size))

        # initialize pointer array
        # set null values to negative and will return error if called as index
        part_index_pointers_at_snap = ut.array.get_array_null(part[self.species_name]['id'].size)
        null_value = part_index_pointers_at_snap[0]

        # counters for sanity checks
        id_no_match_number_tot = 0
        prop_no_match_number_tot = 0
        prop_redundant_number_tot = 0
        test_prop_offset_number_tot = 0

        for snapshot_index in snapshot_indices:
            if self.species_name == 'star':
                # keep only particles that formed prior to this snapshot
                part_indices = part_indices[
                    part[self.species_name].prop('form.snapshot', part_indices) <= snapshot_index]
                if not len(part_indices):
                    self.say('# no {} particles to track at snapshot index <= {}\n'.format(
                             self.species_name, snapshot_index))
                    break

            part_ids = part[self.species_name]['id'][part_indices]

            # read particles at this snapshot
            Read = gizmo_io.ReadClass()
            part_at_snap = Read.read_snapshots(
                self.species_name, 'index', snapshot_index,
                properties=['id', match_property, test_property], element_indices=[0],
                assign_center=False, check_properties=False)

            # assign pointer from particle id to its index in list
            ut.particle.assign_id_to_index(
                part_at_snap, self.species_name, 'id', id_min=0, store_as_dict=True,
                print_diagnostic=False)

            # re-initialize with null values
            part_index_pointers_at_snap *= 0
            part_index_pointers_at_snap += null_value

            id_no_match_number = 0
            prop_no_match_number = 0
            prop_redundant_number = 0
            for pii, part_id in enumerate(part_ids):
                part_index = part_indices[pii]

                try:
                    part_indices_at_snap = part_at_snap[self.species_name].id_to_index[part_id]
                except IndexError:
                    id_no_match_number += 1
                    continue

                if np.isscalar(part_indices_at_snap):
                    # particle id is unique - easy case
                    part_index_pointers_at_snap[part_index] = part_indices_at_snap
                else:
                    # particle id is redundant - difficult case
                    # loop through particles with this id, use match_property to match
                    # sanity check
                    if (np.unique(part_at_snap[self.species_name].prop(
                            match_property, part_indices_at_snap)).size !=
                            part_indices_at_snap.size):
                        prop_redundant_number += 1

                    prop_0 = part[self.species_name].prop(match_property, part_index)

                    for part_index_at_snap in part_indices_at_snap:
                        prop_test = part_at_snap[self.species_name].prop(
                            match_property, part_index_at_snap)
                        if match_property == 'id.child':
                            if prop_test == prop_0:
                                part_index_pointers_at_snap[part_index] = part_index_at_snap
                                break
                        else:
                            if np.abs((prop_test - prop_0) / prop_0) < match_propery_tolerance:
                                part_index_pointers_at_snap[part_index] = part_index_at_snap
                                break
                    else:
                        prop_no_match_number += 1

            if id_no_match_number:
                self.say('! {} not have id match at snapshot {}!'.format(
                         id_no_match_number, snapshot_index))
                id_no_match_number_tot += id_no_match_number
            if prop_no_match_number:
                self.say('! {} not have {} match at snapshot {}!'.format(
                         prop_no_match_number, match_property, snapshot_index))
                prop_no_match_number_tot += prop_no_match_number
            if prop_redundant_number:
                self.say('! {} have redundant {} at snapshot {}!'.format(
                         prop_redundant_number, match_property, snapshot_index))
                prop_redundant_number_tot += prop_redundant_number

            # sanity check
            if (test_property and test_property != match_property and
                    id_no_match_number == prop_no_match_number == prop_redundant_number_tot == 0):
                part_index_pointers_at_snap_test = part_index_pointers_at_snap[
                    part_index_pointers_at_snap >= 0]
                prop_difs = np.abs(
                    (part_at_snap[self.species_name].prop(
                        test_property, part_index_pointers_at_snap_test) -
                     part[self.species_name].prop(test_property, part_indices)) /
                    part[self.species_name].prop(test_property, part_indices))
                test_prop_offset_number = np.sum(prop_difs > match_propery_tolerance)
                if test_prop_offset_number:
                    self.say('! {} have offset {} at snapshot {}!'.format(
                             test_prop_offset_number, test_property, snapshot_index))
                    test_prop_offset_number_tot += test_prop_offset_number

            # write file for this snapshot
            self.io_index_pointer(None, snapshot_index, part_index_pointers_at_snap)

        # print cumulative diagnostics
        if id_no_match_number_tot:
            self.say('! {} total not have id match!'.format(id_no_match_number_tot))
        if prop_no_match_number_tot:
            self.say('! {} total not have {} match!'.format(
                     prop_no_match_number_tot, match_property))
        if prop_redundant_number_tot:
            self.say('! {} total have redundant {}!'.format(
                     prop_redundant_number_tot, match_property))
        if test_prop_offset_number_tot:
            self.say('! {} total have offset {}'.format(
                     test_prop_offset_number_tot, test_property))

    def io_index_pointer(
        self, part=None, snapshot_index=None, part_index_pointers=None):
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

        if part_index_pointers is not None:
            # write to file
            directory = ut.io.get_path(self.directory, create_path=True)
            ut.io.file_hdf5(directory + file_name, {hdf5_dict_name: part_index_pointers})

        else:
            # read from file
            directory = ut.io.get_path(self.directory)
            dict_in = ut.io.file_hdf5(directory + file_name)
            part_index_pointers = dict_in[hdf5_dict_name]
            if part is None:
                return part_index_pointers
            else:
                part.index_pointers = part_index_pointers


IndexPointer = IndexPointerClass()


class HostDistanceClass(IndexPointerClass):
    '''
    Compute distance wrt the host galaxy center for particles across time.
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

        # star formation host distances to write/read
        self.form_host_distance_kind = 'form.host.distance'

    def write_form_host_distance(self, part=None, snapshot_indices=[], part_indices=None):
        '''
        Assign to each particle its distance wrt the host galaxy center at the snapshot after
        it formed.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        part_indices : array-like : list of particle indices to assign to
        '''
        force_float32 = True

        if part is None:
            # read particles at z = 0 - all properties possibly relevant for matching
            properties_read = [
                'position', 'mass', 'id', 'id.child', 'massfraction.metals', 'form.scalefactor']
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                self.species_name, 'redshift', 0, properties=properties_read, element_indices=[0],
                force_float32=force_float32, assign_center=False, check_properties=False)

        # get list of particles to assign
        if part_indices is None or not len(part_indices):
            part_indices = ut.array.get_arange(part[self.species_name]['position'].shape[0])

        # get list of snapshots to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # store particle properties as 32-bit and initialize to nan
        particle_number = part[self.species_name]['position'].shape[0]
        part[self.species_name][self.form_host_distance_kind] = np.zeros(
            [particle_number, 3], np.float32) + np.nan
        # store principal axes rotation vectors at each snapshot
        part[self.species_name].principal_axes_vectors_at_snapshots = (
            np.zeros([snapshot_indices.size, 3, 3],
                     dtype=part[self.species_name]['position'].dtype) + np.nan)

        id_wrong_number_tot = 0
        id_none_number_tot = 0

        for snapshot_index in snapshot_indices:
            part_indices_form = part_indices[
                part[self.species_name].prop('form.snapshot', part_indices) == snapshot_index]

            self.say('\n# {} to assign at snapshot {}'.format(
                     part_indices_form.size, snapshot_index))

            if part_indices_form.size:
                Read = gizmo_io.ReadClass()
                part_at_snap = Read.read_snapshots(
                    self.species_name, 'index', snapshot_index,
                    properties=['position', 'mass', 'id'], force_float32=force_float32,
                    assign_center=True, check_properties=True)

                if snapshot_index == part.snapshot['index']:
                    part_index_pointers = part_indices
                else:
                    part_index_pointers = self.io_index_pointer(snapshot_index=snapshot_index)

                part_indices_at_snap = part_index_pointers[part_indices_form]

                # sanity checks
                masks = (part_indices_at_snap >= 0)
                id_none_number = part_indices_form.size - np.sum(masks)
                if id_none_number:
                    self.say('! {} have no id match at snapshot {}!'.format(
                             id_none_number, snapshot_index))
                    id_none_number_tot += id_none_number
                    part_indices_at_snap = part_indices_at_snap[masks]
                    part_indices_form = part_indices_form[masks]

                id_wrong_number = np.sum(
                    part[self.species_name]['id'][part_indices_form] !=
                    part_at_snap[self.species_name]['id'][part_indices_at_snap])
                if id_wrong_number:
                    self.say('! {} have wrong id match at snapshot {}!'.format(
                             id_wrong_number, snapshot_index))
                    id_wrong_number_tot += id_wrong_number

                # 3-D distance wrt host along default x,y,z axes [kpc physical]
                distance_vectors = ut.coordinate.get_distances(
                    'vector', part_at_snap[self.species_name]['position'][part_indices_at_snap],
                    part_at_snap.center_position,
                    part_at_snap.info['box.length']) * part_at_snap.snapshot['scalefactor']

                # compute host galaxy properties, including R_90
                gal = ut.particle.get_galaxy_properties(
                    part_at_snap, self.species_name, 'mass.percent', 90, distance_max=15,
                    print_results=True)

                # compute rotation vectors within R_90
                rotation_vectors, _ev, _ar = ut.particle.get_principal_axes(
                    part_at_snap, self.species_name, gal['radius'], scalarize=True,
                    print_results=True)

                # store rotation vectors
                part[self.species_name].principal_axes_vectors_at_snapshots[snapshot_index] = \
                    rotation_vectors

                # rotate to align with principal axes
                distance_vectors = ut.coordinate.get_coordinates_rotated(
                    distance_vectors, rotation_vectors)

                # assign 3-D distance wrt host along principal axes [kpc physical]
                part[self.species_name][self.form_host_distance_kind][part_indices_form] = \
                    distance_vectors

                # continuously (re)write as go
                self.io_form_host_distance(part, 'write')

    def io_form_host_distance(self, part, io_direction='read'):
        '''
        Read or write, for each particle, at the first snapshot after it formed,
        its distance wrt the host galaxy center, aligned with the principal axes of the host galaxy
        at that time.
        If read, assign to particle catalog.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        io_direction : string : 'read' or 'write'
        '''
        file_name = '{}_form_host_distance_{:03d}'.format(self.species_name, part.snapshot['index'])

        if io_direction == 'write':
            directory = ut.io.get_path(self.directory, create_path=True)
            dict_out = collections.OrderedDict()
            dict_out['id'] = part[self.species_name]['id']
            dict_out[self.form_host_distance_kind] = \
                part[self.species_name][self.form_host_distance_kind]
            dict_out['principal.axes.vectors'] = \
                part[self.species_name].principal_axes_vectors_at_snapshots

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
                elif prop == 'principal.axes.vectors':
                    part[self.species_name].principal_axes_vectors_at_snapshots = dict_in[prop]
                else:
                    part[self.species_name][prop] = dict_in[prop]

            # fix naming convention in older files - keep only 3-D vector distance
            if self.form_host_distance_kind + '.3d' in part[self.species_name]:
                part[self.species_name][self.form_host_distance_kind] = \
                    part[self.species_name][self.form_host_distance_kind + '.3d']
                del(part[self.species_name][self.form_host_distance_kind + '.3d'])

        else:
            raise ValueError('! not recognize io_direction = {}'.format(io_direction))


HostDistance = HostDistanceClass()


#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('specify function: indices, distances, indices+distances')

    function_kind = str(sys.argv[1])

    assert ('indices' in function_kind or 'distances' in function_kind)

    if 'indices' in function_kind:
        IndexPointer.write_index_pointer()

    if 'distances' in function_kind:
        HostDistance.write_form_host_distance()
