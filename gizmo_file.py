#!/usr/bin/env python


'''
Delete snapshot files or transfer them across machines.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import os
import sys
import glob
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut


#===================================================================================================
# delete files
#===================================================================================================
def delete_snapshots(directory='.', snapshot_index_limits=[3, 599]):
    '''
    Delete all snapshots within snapshot_index_limits in given directory,
    except for those in snapshot_indices_keep list below.

    Parameters
    ----------
    directory : string : directory of snapshots
    snapshot_index_limits : list : min and max snapshot indices to consider deleting
    '''
    snapshot_indices_keep = [
        0, 1, 2,
        11, 20, 26, 33, 41, 52, 59, 67, 77, 88,
        102, 120, 142, 172,
        214, 242, 277,
        322, 382,
        412, 446, 486,
        534, 561, 585, 590, 600
    ]

    if snapshot_index_limits is None or not len(snapshot_index_limits):
        snapshot_index_limits = [1, Inf]

    snapshot_name_bases = ['snapshot_*.hdf5', 'snapdir_*']

    os.chdir(directory)

    for snapshot_name_base in snapshot_name_bases:
        snapshot_names = glob.glob(snapshot_name_base)
        snapshot_names.sort()

        for snapshot_name in snapshot_names:
            snapshot_index = ut.io.get_numbers_in_string(snapshot_name)[0]

            if (snapshot_index not in snapshot_indices_keep and
                    snapshot_index >= min(snapshot_index_limits) and
                    snapshot_index <= max(snapshot_index_limits)):

                print('deleting {}'.format(snapshot_name))
                os.system('rm -rf ' + snapshot_name)


#===================================================================================================
# transfer files
#===================================================================================================
def transfer_snapshots(
    machine_name='stampede', from_directory='$STAMPEDE_SCRATCH/m12i_ref13/output',
    snapshot_kind='file', snapshot_indices=600, to_directory='.'):
    '''
    Transfer snapshot file[s] or directory[s] from remote machine to local.

    Parameters
    ----------
    machine_name : string : name of host machine
    from_directory : string : directory of snapshot file on host machine
    snapshot_kind : string : 'file' or 'directory'
    snapshot indices : int or list : index[s] of snapshots
    to_directory : string : local directory to put snapshots
    '''
    if snapshot_kind == 'file':
        snapshot_name_base = 'snapshot_{:.3d}.hdf5'
    elif snapshot_kind == 'directory':
        snapshot_name_base = 'snapdir_{:.3d}'
    else:
        raise ValueError('not recognize snapshot_kind = ' + snapshot_kind)

    #if machine_name == 'stampede':
    #    from_directory = '$STAMPEDE_SCRATCH/' + from_directory
    #elif machine_name == 'zwicky':
    #    from_directory = '$ZWICKY_SCRATCH/' + from_directory
    #elif machine_name == 'ranch':
    #    from_directory = '$RANCH_HOME/stampede/' + from_directory

    from_directory = ut.io.get_path(from_directory)

    if np.isscalar(snapshot_indices):
        snapshot_indices = [snapshot_indices]

    snapshot_path_names = ''
    for snapshot_index in snapshot_indices:
        snapshot_path_names += from_directory + snapshot_name_base.format(snapshot_index) + ' '

    os.system('rsync -ahvP {}:"{}" {}'.format(machine_name, snapshot_path_names, to_directory))


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise ValueError('specify function: delete, transfer')

    function_kind = str(sys.argv[1])
    assert ('delete' in function_kind or 'transfer' in function_kind)

    directory = '.'

    if 'delete' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_index_limits = None
        if len(sys.argv) > 4:
            snapshot_index_limits = [int(sys.argv[3]), int(sys.argv[4])]

        delete_snapshots(directory, snapshot_index_limits)

    elif 'transfer' in function_kind:
        if len(sys.argv) < 6:
            raise ValueError(
                'imports: machine_name directory snapshot_kind snapshot_time_file_name')

    machine_name = str(sys.argv[2])
    from_directory = str(sys.argv[3])
    snapshot_kind = str(sys.argv[4])
    snapshot_time_file_name = str(sys.argv[5])

    Snapshot = ut.simulation.SnapshotClass()
    Snapshot.read_snapshots(snapshot_time_file_name)

    transfer_snapshots(machine_name, from_directory, snapshot_kind, Snapshot['index'])