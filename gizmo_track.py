'''
Track particles across snapshots.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut
from . import gizmo_io


# directory where to store particle tracking files
TRACK_DIRECTORY = ut.io.get_path('track/')


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


def write_particle_index_pointer(
    part=None, species='star', match_prop_name='id.child', match_prop_tolerance=1e-6,
    test_prop_name='form.scalefactor', snapshot_indices=[]):
    '''
    Assign to each particle a pointer to its index in the list of particles at each previous
    snapshot, to make it easier to track particles back in time.
    Write index pointers to a pickle file, one for each previous snapshot.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : string : name of particle species to track
    match_prop_name : string :
        some particles have the same id. this is the property to use to match them.
        options (in order of preference): 'id.child', 'massfraction.metals', 'form.scalefactor'
    match_prop_tolerance : float : tolerance for matching via match_prop_name, if it is a float
    test_prop_name : string : additional property to use to test matching
    snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
    '''
    Say = ut.io.SayClass(write_particle_index_pointer)

    assert match_prop_name in ['id.child', 'massfraction.metals', 'form.scalefactor']

    if part is None:
        # read particles at z = 0
        # complete list of all possible properties relevant to use in matching
        property_names_read = ['id', 'id.child', 'massfraction.metals', 'form.scalefactor']
        if match_prop_name not in property_names_read:
            property_names_read.append(match_prop_name)
        if test_prop_name and test_prop_name not in property_names_read:
            property_names_read.append(test_prop_name)
        part = gizmo_io.Read.read_snapshots(
            species, 'redshift', 0, property_names=property_names_read, element_indices=[0],
            force_float32=True, assign_center=False, check_sanity=False)

    # older snapshot files do not have id.child - use metal abundance instead
    if match_prop_name == 'id.child' and 'id.child' not in part[species]:
        Say.say('input match_prop_name = {}, but it does not exist in the snapshot'.format(
                match_prop_name))
        match_prop_name = 'massfraction.metals'
        Say.say('switching to using match_prop_name = {}'.format(match_prop_name))

    assert part[species].prop(match_prop_name) is not None

    if test_prop_name:
        assert part[species].prop(test_prop_name) is not None

    if species == 'star' and 'form.index' not in part[species]:
        assign_star_form_snapshot(part)

    # get list of snapshot indices to assign
    if snapshot_indices is None or not len(snapshot_indices):
        snapshot_indices = np.arange(min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
    snapshot_indices = np.setdiff1d(snapshot_indices, part.snapshot['index'])  # skip current
    snapshot_indices = snapshot_indices[::-1]  # work backwards in time

    # particles to assign to
    part_indices = ut.array.get_arange(part[species]['id'])

    # diagnostic
    pis_multiple = ut.particle.get_indices_id_kind(part, species, 'multiple', part_indices)
    Say.say('{} particles have redundant id'.format(pis_multiple.size))

    # initialize pointer array
    # set null values to negative and will return error if called as index
    part_index_pointers_at_snap = ut.array.get_array_null(part[species]['id'].size)
    null_value = part_index_pointers_at_snap[0]

    track_directory = ut.io.get_path(TRACK_DIRECTORY, create_path=True)

    # counters for sanity checks
    id_no_match_number_tot = 0
    prop_no_match_number_tot = 0
    prop_redundant_number_tot = 0
    test_prop_offset_number_tot = 0

    for snapshot_index in snapshot_indices:
        if species == 'star':
            # keep only particles that formed prior to this snapshot
            part_indices = part_indices[part[species]['form.index'][part_indices] <= snapshot_index]
        part_ids = part[species]['id'][part_indices]

        # read particles at this snapshot
        part_at_snap = gizmo_io.Read.read_snapshots(
            species, 'index', snapshot_index,
            property_names=['id', match_prop_name, test_prop_name], element_indices=[0],
            force_float32=True, assign_center=False, check_sanity=False)

        # assign pointer from particle id to its index in list
        ut.particle.assign_id_to_index(
            part_at_snap, species, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

        # re-initialize with null values
        part_index_pointers_at_snap *= 0
        part_index_pointers_at_snap += null_value

        id_no_match_number = 0
        prop_no_match_number = 0
        prop_redundant_number = 0
        for pii, part_id in enumerate(part_ids):
            part_index = part_indices[pii]

            try:
                part_indices_at_snap = part_at_snap[species].id_to_index[part_id]
            except:
                id_no_match_number += 1
                continue

            if np.isscalar(part_indices_at_snap):
                # particle id is unique - easy
                part_index_pointers_at_snap[part_index] = part_indices_at_snap
            else:
                # particle id is redundant
                # loop through particles with this id, use match_prop_name to match
                # sanity check
                if np.unique(part_at_snap[species].prop(
                        match_prop_name, part_indices_at_snap)).size != part_indices_at_snap.size:
                    prop_redundant_number += 1
                prop_0 = part[species].prop(match_prop_name, part_index)
                for part_index_at_snap in part_indices_at_snap:
                    prop_test = part_at_snap[species].prop(match_prop_name, part_index_at_snap)
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
            Say.say('! {} not have id match at snapshot {}!'.format(
                    id_no_match_number, snapshot_index))
            id_no_match_number_tot += id_no_match_number
        if prop_no_match_number:
            Say.say('! {} not have match_prop_name match at snapshot {}!'.format(
                    prop_no_match_number, snapshot_index))
            prop_no_match_number_tot += prop_no_match_number
        if prop_redundant_number:
            Say.say('! {} have redundant match_prop_name at snapshot {}!'.format(
                    prop_redundant_number, snapshot_index))
            prop_redundant_number_tot += prop_redundant_number

        # sanity check
        if (test_prop_name and test_prop_name != match_prop_name and
                id_no_match_number == prop_no_match_number == prop_redundant_number_tot == 0):
            part_index_pointers_at_snap_test = part_index_pointers_at_snap[
                part_index_pointers_at_snap >= 0]
            prop_difs = np.abs(
                (part_at_snap[species].prop(test_prop_name, part_index_pointers_at_snap_test) -
                 part[species].prop(test_prop_name, part_indices)) /
                part[species].prop(test_prop_name, part_indices))
            test_prop_offset_number = np.sum(prop_difs > match_prop_tolerance)
            if test_prop_offset_number:
                Say.say('! {} have offset {} at snapshot {}!'.format(
                        test_prop_offset_number, test_prop_name, snapshot_index))
                test_prop_offset_number_tot += test_prop_offset_number

        # write file for this snapshot
        file_name = '{}_indices_{:03d}'.format(species, snapshot_index)
        np.save(track_directory + file_name, part_index_pointers_at_snap, allow_pickle=False)
        #ut.io.pickle_object(track_directory + file_name, 'write', part_index_pointers_at_snap)

    # print cumulative diagnostics
    if id_no_match_number_tot:
        Say.say('! {} total not have id match!'.format(id_no_match_number_tot))
    if prop_no_match_number_tot:
        Say.say('! {} total not have match_prop_name match!'.format(prop_no_match_number_tot))
    if prop_redundant_number_tot:
        Say.say('! {} total have redundant match_prop_name!'.format(prop_redundant_number_tot))
    if test_prop_offset_number_tot:
        Say.say('! {} total have offset {}'.format(test_prop_offset_number_tot, test_prop_name))


def write_star_form_host_distance(part=None, snapshot_indices=[], part_indices=None):
    '''
    Assign to each star particle its distance wrt the host galaxy center at the snapshot after
    it formed.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    snapshot_indices : array-like : list of snapshot indices at which to assign index pointers
    part_indices : array-like : list of particle indices to assign to
    '''
    Say = ut.io.SayClass(write_star_form_host_distance)

    spec_name = 'star'
    prop_name = 'form.host.distance'

    if part is None:
        # read particles at z = 0
        # complete list of all possible properties relevant for matching
        property_names_read = [
            'position', 'mass', 'id', 'id.child', 'massfraction.metals', 'form.scalefactor']
        part = gizmo_io.Read.read_snapshots(
            spec_name, 'redshift', 0, property_names=property_names_read, element_indices=[0],
            force_float32=True, assign_center=False, check_sanity=False)

    # get list of particles to assign
    if part_indices is None or not len(part_indices):
        part_indices = ut.array.get_arange(part[spec_name]['position'].shape[0])

    # get list of snapshots to assign
    if snapshot_indices is None or not len(snapshot_indices):
        snapshot_indices = np.arange(min(part.Snapshot['index']), max(part.Snapshot['index']) + 1)
    snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

    if 'form.index' not in part[spec_name]:
        assign_star_form_snapshot(part)

    # store form.host.distance as 32-bit and initialize to -1
    part[spec_name][prop_name] = np.zeros(part[spec_name]['position'].shape[0], np.float32) - 1

    id_wrong_number_tot = 0
    id_none_number_tot = 0

    for snapshot_index in snapshot_indices:
        part_indices_form = part_indices[
            part[spec_name]['form.index'][part_indices] == snapshot_index]

        Say.say('# {} to assign at snapshot {}'.format(part_indices_form.size, snapshot_index))

        if part_indices_form.size:
            part_at_snap = gizmo_io.Read.read_snapshots(
                spec_name, 'index', snapshot_index, property_names=['position', 'mass', 'id'],
                force_float32=True, assign_center=True, check_sanity=True)

            if snapshot_index == part.snapshot['index']:
                part_index_pointers = part_indices
            else:
                file_name = 'star_indices_{:03d}'.format(snapshot_index)
                #part_index_pointers = ut.io.pickle_object(TRACK_DIRECTORY + file_name, 'read')
                part_index_pointers = np.load(TRACK_DIRECTORY + file_name, 'read')

            part_indices_at_snap = part_index_pointers[part_indices_form]

            # sanity checks
            masks = (part_indices_at_snap >= 0)
            id_none_number = part_indices_form.size - np.sum(masks)
            if id_none_number:
                Say.say('! {} have no id match at snapshot {}!'.format(
                        id_none_number, snapshot_index))
                id_none_number_tot += id_none_number
                part_indices_at_snap = part_indices_at_snap[masks]
                part_indices_form = part_indices_form[masks]

            id_wrong_number = np.sum(part[spec_name]['id'][part_indices_form] !=
                                     part_at_snap[spec_name]['id'][part_indices_at_snap])
            if id_wrong_number:
                Say.say('! {} have wrong id match at snapshot {}!'.format(
                        id_wrong_number, snapshot_index))
                id_wrong_number_tot += id_wrong_number

            # compute 3-D distance wrt host [kpc physical]
            part[spec_name][prop_name][part_indices_form] = ut.coordinate.get_distances(
                'scalar', part_at_snap[spec_name]['position'][part_indices_at_snap],
                part_at_snap.center_position,
                part_at_snap.info['box.length']) * part_at_snap.snapshot['scalefactor']

            # continuously (re)write as go
            pickle_star_form_host_distance(part, 'write')


def write_star_form_host_distance_orig(
    part, use_child_id=False, snapshot_index_limits=[], part_indices=None):
    '''
    Assign to each star particle the distance wrt the host galaxy center at the first snapshot
    after it formed.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    use_child_id : boolean : whether to use id.child to match particles with redundant ids
    snapshot_index_limits : list : min and max snapshot indices to impose matching to
    part_indices : array-like : list of particle indices to assign to
    '''
    Say = ut.io.SayClass(write_star_form_host_distance)

    spec_name = 'star'
    prop_name = 'form.host.distance'

    # store form.host.distance as 32-bit and initialize to -1
    part[spec_name][prop_name] = np.zeros(part[spec_name]['position'].shape[0], np.float32) - 1

    if 'form.index' not in part['star']:
        assign_star_form_snapshot(part)

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.get_arange(part[spec_name]['position'].shape[0])

    if use_child_id and 'id.child' in part[spec_name]:
        pass
        #pis_unsplit = part_indices[(part[spec_name]['id.child'][part_indices] == 0) *
        #                           (part[spec_name]['id.generation'][part_indices] == 0)]
        #pis_oversplit = part_indices[part[spec_name]['id.child'][part_indices] >
        #                             2 ** part[spec_name]['id.generation'][part_indices]]
    else:
        pis_unique = ut.particle.get_indices_id_kind(part, spec_name, 'unique', part_indices)
        pis_multiple = ut.particle.get_indices_id_kind(part, spec_name, 'multiple', part_indices)
        Say.say('particles with id that is: unique {}, redundant {}'.format(
                pis_unique.size, pis_multiple.size))

    part_indices = pis_unique  # particles to assign to

    # get snapshot indices to sort going back in time
    form_indices = np.unique(part[spec_name]['form.index'][part_indices])[::-1]
    if snapshot_index_limits is not None and len(snapshot_index_limits):
        form_indices = form_indices[form_indices >= min(snapshot_index_limits)]
        form_indices = form_indices[form_indices <= max(snapshot_index_limits)]

    assigned_number_tot = 0
    form_time_offset_number_tot = 0
    no_id_number_tot = 0

    for snapshot_index in form_indices:
        part_indices_form_all = part_indices[
            part[spec_name]['form.index'][part_indices] == snapshot_index]
        part_ids_form = part[spec_name]['id'][part_indices_form_all]

        part_at_snap = gizmo_io.Read.read_snapshots(
            spec_name, 'index', snapshot_index,
            property_names=['position', 'mass', 'id', 'form.scalefactor'], element_indices=[0],
            force_float32=True, assign_center=True, check_sanity=True)

        ut.particle.assign_id_to_index(
            part_at_snap, spec_name, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

        Say.say('# {} to assign this snapshot'.format(part_ids_form.size))
        assigned_number_tot += part_ids_form.size

        pis_snap = []
        pis_form = []
        no_id_number = 0
        for pii, part_id in enumerate(part_ids_form):
            try:
                pi = part_at_snap[spec_name].id_to_index[part_id]
                if np.isscalar(pi):
                    pis_snap.append(pi)
                    pis_form.append(part_indices_form_all[pii])
                else:
                    no_id_number += 1
            except:
                no_id_number += 1
        pis_snap = np.array(pis_snap, dtype=part_indices_form_all.dtype)
        pis_form = np.array(pis_form, dtype=part_indices_form_all.dtype)

        if no_id_number:
            Say.say('! {} not have id match'.format(no_id_number))
            no_id_number_tot += no_id_number

        # sanity check
        form_time_tolerance = 1e-5
        form_scalefactor_difs = np.abs(
            part_at_snap[spec_name]['form.scalefactor'][pis_snap] -
            part[spec_name]['form.scalefactor'][pis_form]) / part_at_snap.snapshot['scalefactor']
        form_time_offset_number = np.sum(form_scalefactor_difs > form_time_tolerance)
        if form_time_offset_number:
            Say.say('! {} have offset formation time'.format(form_time_offset_number))
            form_time_offset_number_tot += form_time_offset_number

        # compute 3-D distance [kpc physical]
        part[spec_name][prop_name][pis_form] = ut.coordinate.get_distances(
            'scalar', part_at_snap[spec_name]['position'][pis_snap], part_at_snap.center_position,
            part_at_snap.info['box.length']) * part_at_snap.snapshot['scalefactor']

        # print cumulative diagnostics
        Say.say('{} (of {}) total assigned'.format(assigned_number_tot, part_indices.size))
        if no_id_number_tot:
            Say.say('! {} total not have id match'.format(no_id_number_tot))
        if form_time_offset_number_tot:
            Say.say('! {} total have offset formation time'.format(form_time_offset_number_tot))

        # continuously write as go, in case happens to crash along the way
        pickle_star_form_host_distance(part, 'write')


#===================================================================================================
# read/write
#===================================================================================================
def pickle_star_form_host_distance(part, pickle_direction='read'):
    '''
    Read or write, for each star particle, its distance wrt the host galaxy center at the first
    snapshot after it formed.
    If read, assign to particle catalog.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    pickle_direction : string : pickle direction: 'read', 'write'
    '''
    Say = ut.io.SayClass(pickle_star_form_host_distance)

    spec_name = 'star'
    prop_name = 'form.host.distance'

    file_name = 'star_form_host_distance_{:03d}'.format(part.snapshot['index'])

    if pickle_direction == 'write':
        track_directory = ut.io.get_path(TRACK_DIRECTORY, create_path=True)
        pickle_object = [part[spec_name][prop_name], part[spec_name]['id']]
        ut.io.pickle_object(track_directory + file_name, pickle_direction, pickle_object)

    elif pickle_direction == 'read':
        part[spec_name][prop_name], part_ids = ut.io.pickle_object(
            TRACK_DIRECTORY + file_name, pickle_direction)

        # sanity check
        bad_id_number = np.sum(part[spec_name]['id'] != part_ids)
        if bad_id_number:
            Say.say('! {} particles have mismatched id - bad!'.format(bad_id_number))

    else:
        raise ValueError('! not recognize pickle_direction = {}'.format(pickle_direction))


#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    write_particle_index_pointer()
