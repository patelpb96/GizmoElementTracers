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
def assign_star_form_snapshot(part):
    '''
    Assign to each star particle the index of the snapshot after it formed,
    to be able to track it back as far as possible.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    '''
    # increase formation time slightly for safety, because scale-factors of snapshots that are
    # actually written do not exactly coincide with input list of snapshot scale-factors
    padding_factor = (1 + 1e-7)

    form_scalefactors = np.array(part['star']['form.scalefactor'])
    form_scalefactors[form_scalefactors < 1] *= padding_factor

    part['star']['form.index'] = part.Snapshot.get_snapshot_indices(
        'scalefactor', form_scalefactors, round_kind='up')


class IndexPointerClass(ut.io.SayClass):
    '''
    Compute particle index pointers for tracking particles across time.
    '''
    def __init__(self, species='star', directory=TRACK_DIRECTORY):
        '''
        Parameters
        ----------
        species : string : name of particle species to track
        directory : string : directory where to write files
        '''
        self.species = species
        self.directory = directory

    def write_index_pointer(
        self, part=None, match_prop_name='id.child', match_prop_tolerance=1e-6,
        test_prop_name='form.scalefactor', snapshot_indices=[]):
        '''
        Assign to each particle a pointer to its index in the list of particles at each previous
        snapshot, to make it easier to track particles back in time.
        Write index pointers to a pickle file, one for each previous snapshot.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        match_prop_name : string :
            some particles have the same id. this is the property to use to match them.
            options (in order of preference): 'id.child', 'massfraction.metals', 'form.scalefactor'
        match_prop_tolerance : float : tolerance for matching via match_prop_name, if it is a float
        test_prop_name : string : additional property to use to test matching
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        '''
        assert match_prop_name in ['id.child', 'massfraction.metals', 'form.scalefactor']

        spec_name = self.species

        if part is None:
            # read particles at z = 0
            # complete list of all possible properties relevant to use in matching
            property_names_read = ['id', 'id.child', 'massfraction.metals', 'form.scalefactor']
            if match_prop_name not in property_names_read:
                property_names_read.append(match_prop_name)
            if test_prop_name and test_prop_name not in property_names_read:
                property_names_read.append(test_prop_name)
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                spec_name, 'redshift', 0, property_names=property_names_read,
                element_indices=[0], force_float32=True, assign_center=False, check_sanity=False)

        # older snapshot files do not have id.child - use metal abundance instead
        if match_prop_name == 'id.child' and 'id.child' not in part[spec_name]:
            self.say('input match_prop_name = {}, but it does not exist in the snapshot'.format(
                     match_prop_name))
            match_prop_name = 'massfraction.metals'
            self.say('switching to using: {}'.format(match_prop_name))

        assert part[spec_name].prop(match_prop_name) is not None

        if test_prop_name:
            assert part[spec_name].prop(test_prop_name) is not None

        if spec_name == 'star' and 'form.index' not in part[spec_name]:
            assign_star_form_snapshot(part)

        # get list of snapshot indices to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
        snapshot_indices = np.setdiff1d(snapshot_indices, part.snapshot['index'])  # skip current
        snapshot_indices = snapshot_indices[::-1]  # work backwards in time

        # particles to assign to
        part_indices = ut.array.get_arange(part[spec_name]['id'])

        # diagnostic
        pis_multiple = ut.particle.get_indices_id_kind(part, spec_name, 'multiple', part_indices)
        self.say('{} particles have redundant id'.format(pis_multiple.size))

        # initialize pointer array
        # set null values to negative and will return error if called as index
        part_index_pointers_at_snap = ut.array.get_array_null(part[spec_name]['id'].size)
        null_value = part_index_pointers_at_snap[0]

        # counters for sanity checks
        id_no_match_number_tot = 0
        prop_no_match_number_tot = 0
        prop_redundant_number_tot = 0
        test_prop_offset_number_tot = 0

        for snapshot_index in snapshot_indices:
            if spec_name == 'star':
                # keep only particles that formed prior to this snapshot
                part_indices = part_indices[
                    part[spec_name]['form.index'][part_indices] <= snapshot_index]
                if not len(part_indices):
                    self.say('# no {} particles to track at snapshot index <= {}\n'.format(
                             spec_name, snapshot_index))
                    break

            part_ids = part[spec_name]['id'][part_indices]

            # read particles at this snapshot
            Read = gizmo_io.ReadClass()
            part_at_snap = Read.read_snapshots(
                spec_name, 'index', snapshot_index,
                property_names=['id', match_prop_name, test_prop_name], element_indices=[0],
                force_float32=True, assign_center=False, check_sanity=False)

            # assign pointer from particle id to its index in list
            ut.particle.assign_id_to_index(
                part_at_snap, spec_name, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

            # re-initialize with null values
            part_index_pointers_at_snap *= 0
            part_index_pointers_at_snap += null_value

            id_no_match_number = 0
            prop_no_match_number = 0
            prop_redundant_number = 0
            for pii, part_id in enumerate(part_ids):
                part_index = part_indices[pii]

                try:
                    part_indices_at_snap = part_at_snap[spec_name].id_to_index[part_id]
                except:
                    id_no_match_number += 1
                    continue

                if np.isscalar(part_indices_at_snap):
                    # particle id is unique - easy case
                    part_index_pointers_at_snap[part_index] = part_indices_at_snap
                else:
                    # particle id is redundant - difficult case
                    # loop through particles with this id, use match_prop_name to match
                    # sanity check
                    if (np.unique(part_at_snap[spec_name].prop(
                            match_prop_name, part_indices_at_snap)).size !=
                            part_indices_at_snap.size):
                        prop_redundant_number += 1

                    prop_0 = part[spec_name].prop(match_prop_name, part_index)

                    for part_index_at_snap in part_indices_at_snap:
                        prop_test = part_at_snap[spec_name].prop(
                            match_prop_name, part_index_at_snap)
                        if match_prop_name == 'id.child':
                            if prop_test == prop_0:
                                part_index_pointers_at_snap[part_index] = part_index_at_snap
                                break
                        else:
                            if np.abs((prop_test - prop_0) / prop_0) < match_prop_tolerance:
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
                         prop_no_match_number, match_prop_name, snapshot_index))
                prop_no_match_number_tot += prop_no_match_number
            if prop_redundant_number:
                self.say('! {} have redundant {} at snapshot {}!'.format(
                         prop_redundant_number, match_prop_name, snapshot_index))
                prop_redundant_number_tot += prop_redundant_number

            # sanity check
            if (test_prop_name and test_prop_name != match_prop_name and
                    id_no_match_number == prop_no_match_number == prop_redundant_number_tot == 0):
                part_index_pointers_at_snap_test = part_index_pointers_at_snap[
                    part_index_pointers_at_snap >= 0]
                prop_difs = np.abs(
                    (part_at_snap[spec_name].prop(
                        test_prop_name, part_index_pointers_at_snap_test) -
                     part[spec_name].prop(test_prop_name, part_indices)) /
                    part[spec_name].prop(test_prop_name, part_indices))
                test_prop_offset_number = np.sum(prop_difs > match_prop_tolerance)
                if test_prop_offset_number:
                    self.say('! {} have offset {} at snapshot {}!'.format(
                             test_prop_offset_number, test_prop_name, snapshot_index))
                    test_prop_offset_number_tot += test_prop_offset_number

            # write file for this snapshot
            self.io_index_pointer(None, snapshot_index, part_index_pointers_at_snap)

        # print cumulative diagnostics
        if id_no_match_number_tot:
            self.say('! {} total not have id match!'.format(id_no_match_number_tot))
        if prop_no_match_number_tot:
            self.say('! {} total not have {} match!'.format(
                     prop_no_match_number_tot, match_prop_name))
        if prop_redundant_number_tot:
            self.say('! {} total have redundant {}!'.format(
                     prop_redundant_number_tot, match_prop_name))
        if test_prop_offset_number_tot:
            self.say('! {} total have offset {}'.format(
                     test_prop_offset_number_tot, test_prop_name))

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

        file_name = '{}_indices_{:03d}'.format(self.species, snapshot_index)

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
    def __init__(self, species='star', directory=TRACK_DIRECTORY):
        '''
        Parameters
        ----------
        species : string : name of particle species to track
        directory : string : directory to write files
        '''
        self.species = species
        self.directory = directory

        # star form host distance kinds to write/read
        self.host_distance_kinds = ['form.host.distance', 'form.host.distance.3d']

    def write_form_host_distance(
        self, part=None, snapshot_indices=[], part_indices=None):
        '''
        Assign to each star particle its distance wrt the host galaxy center at the snapshot after
        it formed.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
        part_indices : array-like : list of particle indices to assign to
        '''
        spec_name = self.species

        if part is None:
            # read particles at z = 0
            # read all properties possibly relevant for matching
            property_names_read = [
                'position', 'mass', 'id', 'id.child', 'massfraction.metals', 'form.scalefactor']
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                spec_name, 'redshift', 0, property_names=property_names_read, element_indices=[0],
                force_float32=True, assign_center=False, check_sanity=False)

        # get list of particles to assign
        if part_indices is None or not len(part_indices):
            part_indices = ut.array.get_arange(part[spec_name]['position'].shape[0])

        # get list of snapshots to assign
        if snapshot_indices is None or not len(snapshot_indices):
            snapshot_indices = np.arange(
                min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        if 'form.index' not in part[spec_name]:
            assign_star_form_snapshot(part)

        # store prop_names as 32-bit and initialize to nan
        particle_number = part[spec_name]['position'].shape[0]
        for host_distance_kind in self.host_distance_kinds:
            if '2d' in host_distance_kind:
                shape = [particle_number, 2]
            elif '3d' in host_distance_kind:
                shape = [particle_number, 3]
            else:
                shape = particle_number
            part[spec_name][host_distance_kind] = np.zeros(shape, np.float32) + np.nan

        id_wrong_number_tot = 0
        id_none_number_tot = 0

        for snapshot_index in snapshot_indices:
            part_indices_form = part_indices[
                part[spec_name]['form.index'][part_indices] == snapshot_index]

            self.say('\n# {} to assign at snapshot {}'.format(
                     part_indices_form.size, snapshot_index))

            if part_indices_form.size:
                Read = gizmo_io.ReadClass()
                part_at_snap = Read.read_snapshots(
                    spec_name, 'index', snapshot_index, property_names=['position', 'mass', 'id'],
                    force_float32=True, assign_center=True, check_sanity=True)

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
                    part[spec_name]['id'][part_indices_form] !=
                    part_at_snap[spec_name]['id'][part_indices_at_snap])
                if id_wrong_number:
                    self.say('! {} have wrong id match at snapshot {}!'.format(
                             id_wrong_number, snapshot_index))
                    id_wrong_number_tot += id_wrong_number

                # 3-D distance wrt host along default x,y,z axes [kpc physical]
                distance_vectors = ut.coordinate.get_distances(
                    'vector', part_at_snap[spec_name]['position'][part_indices_at_snap],
                    part_at_snap.center_position,
                    part_at_snap.info['box.length']) * part_at_snap.snapshot['scalefactor']

                # rotate to align with principal axes
                for host_distance_kind in self.host_distance_kinds:
                    if '3d' in host_distance_kind or '2d' in host_distance_kind:
                        # compute galaxy radius
                        gal = ut.particle.get_galaxy_properties(
                            part_at_snap, spec_name, 'mass.percent', 90, distance_max=15,
                            print_results=True)

                        # compute rotation vectors
                        rotation_vectors, _ev, _ar = ut.particle.get_principal_axes(
                            part_at_snap, spec_name, gal['radius'], scalarize=True,
                            print_results=True)

                        # rotate to align with principal axes
                        distance_vectors = ut.coordinate.get_coordinates_rotated(
                            distance_vectors, rotation_vectors)

                        break

                # compute distances to store
                for host_distance_kind in self.host_distance_kinds:
                    if '3d' in host_distance_kind:
                        # 3-D distance wrt host along principal axes [kpc physical]
                        distances_t = distance_vectors

                        #if '3d' in host_distance_kind:
                        #    distance_kind = 'rotated'
                        #elif '2d' in host_distance_kind:
                        #    distance_kind = 'rotated.2d'
                        #distances_t = ut.particle.get_distances_wrt_center(
                        #    part_at_snap, spec_name, distance_kind,
                        #    principal_axes_distance_max=gal_radius,
                        #    part_indicess=part_indices_at_snap, scalarize=True)

                    elif '2d' in host_distance_kind:
                        # distance along major axes and along minor axis [kpc physical]
                        distances_t = ut.coordinate.get_distances_major_minor(distance_vectors)

                    else:
                        # total scalar distance wrt host [kpc physical]
                        distances_t = np.sqrt(np.sum(distance_vectors ** 2, 1))

                        #distances_t = ut.particle.get_distances_wrt_center(
                        #    part_at_snap, spec_name, 'scalar',
                        #    part_indicess=part_indices_at_snap, scalarize=True)

                    part[spec_name][host_distance_kind][part_indices_form] = distances_t

                # continuously (re)write as go
                self.io_form_host_distance(part, 'write')

    def io_form_host_distance(self, part, io_direction='read'):
        '''
        Read or write, for each star particle, its distance wrt the host galaxy center at the first
        snapshot after it formed.
        If read, assign to particle catalog.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        io_direction : string : 'read' or 'write'
        '''
        species_name = 'star'

        file_name = '{}_form_host_distance_{:03d}'.format(species_name, part.snapshot['index'])

        if io_direction == 'write':
            directory = ut.io.get_path(self.directory, create_path=True)
            dict_out = collections.OrderedDict()
            dict_out['id'] = part[species_name]['id']
            for host_distance_kind in self.host_distance_kinds:
                dict_out[host_distance_kind] = part[species_name][host_distance_kind]

            ut.io.file_hdf5(directory + file_name, dict_out)

        elif io_direction == 'read':
            directory = ut.io.get_path(self.directory)
            dict_in = ut.io.file_hdf5(directory + file_name)

            # sanity check
            bad_id_number = np.sum(part[species_name]['id'] != dict_in['id'])
            if bad_id_number:
                self.say('! {} particles have mismatched id - bad!'.format(bad_id_number))

            for host_distance_kind in dict_in.keys():
                part[species_name][host_distance_kind] = dict_in[host_distance_kind]

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
