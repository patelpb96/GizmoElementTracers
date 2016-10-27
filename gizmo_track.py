'''
Track particles across time.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut
from . import gizmo_io


def assign_star_form_snapshot_index(part):
    '''
    Assign to each star particle the first snapshot index after it formed,
    to be able to track it back as far as possible.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    '''
    # increase formation time slightly for safety, because output snapshots do not exactly
    # coincide with input snapshot scale-factors
    padding_factor = (1 + 1e-7)

    form_scalefactors = np.array(part['star']['form.scalefactor'])
    form_scalefactors[form_scalefactors < 1] *= padding_factor

    part['star']['form.index'] = part.Snapshot.get_snapshot_indices(
        'scalefactor', form_scalefactors, round_kind='up')


def assign_star_index_snapshots(part, use_child_id=False, snapshot_index_limits=[]):
    '''
    Assign to each star particle its index in the star particle catalog at each snapshot.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    use_child_id : boolean : whether to use id.child to match particles with redundant ids
    snapshot_index_limits : list : min and max snapshot indices to impose matching to
    '''
    Say = ut.io.SayClass(assign_star_index_snapshots)

    spec_name = 'star'

    if 'form.index' not in part['star']:
        assign_star_form_snapshot_index(part)

    if use_child_id and 'id.child' in part[spec_name]:
        part_indices = None
        pis_unsplit = part_indices[(part[spec_name]['id.child'] == 0) *
                                   (part[spec_name]['id.generation'] == 0)]

        pis_oversplit = part_indices[part[spec_name]['id.child'] >
                                     2 ** part[spec_name]['id.generation']]
    else:
        pis_unique = ut.particle.get_indices_id_kind(part, spec_name, 'unique')
        pis_multiple = ut.particle.get_indices_id_kind(part, spec_name, 'multiple')
        Say.say('particles with id that is: unique {}, redundant {}'.format(
                pis_unique.size, pis_multiple.size))

    for snapshot_index in part.Snapshot['index']:
        a = 1


def assign_star_form_host_distance(
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
    Say = ut.io.SayClass(assign_star_form_host_distance)

    spec_name = 'star'
    prop_name = 'form.host.distance'

    # store form.host.distance as 32-bit and initialize to -1
    part[spec_name][prop_name] = np.zeros(part[spec_name]['position'].shape[0], np.float32) - 1

    if 'form.index' not in part['star']:
        assign_star_form_snapshot_index(part)

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.get_arange(part[spec_name]['position'].shape[0])

    if use_child_id and 'id.child' in part[spec_name]:
        pis_unsplit = part_indices[(part[spec_name]['id.child'][part_indices] == 0) *
                                   (part[spec_name]['id.generation'][part_indices] == 0)]

        pis_oversplit = part_indices[part[spec_name]['id.child'][part_indices] >
                                     2 ** part[spec_name]['id.generation'][part_indices]]
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
        pis_form_all = part_indices[part[spec_name]['form.index'][part_indices] == snapshot_index]
        pids_form = part[spec_name]['id'][pis_form_all]

        part_snap = gizmo_io.Read.read_snapshots(
            spec_name, 'index', snapshot_index,
            property_names=['position', 'mass', 'id', 'form.scalefactor'], element_indices=[0],
            force_float32=True, assign_center=True, check_sanity=True)

        ut.particle.assign_id_to_index(
            part_snap, spec_name, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

        Say.say('# {} to assign this snapshot'.format(pids_form.size))
        assigned_number_tot += pids_form.size

        pis_snap = []
        pis_form = []
        no_id_number = 0
        for pii, pid in enumerate(pids_form):
            try:
                pi = part_snap[spec_name].id_to_index[pid]
                if np.isscalar(pi):
                    pis_snap.append(pi)
                    pis_form.append(pis_form_all[pii])
                else:
                    no_id_number += 1
            except:
                no_id_number += 1
        pis_snap = np.array(pis_snap, dtype=pis_form_all.dtype)
        pis_form = np.array(pis_form, dtype=pis_form_all.dtype)

        if no_id_number:
            Say.say('! {} not have id match'.format(no_id_number))
            no_id_number_tot += no_id_number

        # sanity check
        form_time_tolerance = 1e-5
        form_scalefactor_difs = np.abs(
            part_snap[spec_name]['form.scalefactor'][pis_snap] -
            part[spec_name]['form.scalefactor'][pis_form]) / part_snap.snapshot['scalefactor']
        form_time_offset_number = np.sum(form_scalefactor_difs > form_time_tolerance)
        if form_time_offset_number:
            Say.say('! {} have offset formation time'.format(form_time_offset_number))
            form_time_offset_number_tot += form_time_offset_number

        # compute 3-D distance [kpc physical]
        part[spec_name][prop_name][pis_form] = ut.coordinate.get_distances(
            'scalar', part_snap[spec_name]['position'][pis_snap], part_snap.center_position,
            part_snap.info['box.length']) * part_snap.snapshot['scalefactor']  # [kpc physical]

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
def pickle_star_form_snapshot_index(part, pickle_direction='read'):
    '''
    Placeholder for now.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    pickle_direction : string : pickle direction: 'read', 'write'
    '''


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

    file_name = 'output/star_form_host_distance_{:03d}'.format(part.snapshot['index'])

    if pickle_direction == 'write':
        pickle_object = [part[spec_name][prop_name], part[spec_name]['id']]
        ut.io.pickle_object(file_name, pickle_direction, pickle_object)

    elif pickle_direction == 'read':
        part[spec_name][prop_name], pids = ut.io.pickle_object(file_name, pickle_direction)

        # sanity check
        bad_id_number = np.sum(part[spec_name]['id'] != pids)
        if bad_id_number:
            Say.say('! {} particles have mismatched id. this is not right!'.format(bad_id_number))

    else:
        raise ValueError('! not recognize pickle_direction = {}'.format(pickle_direction))
