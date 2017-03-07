#!/usr/bin/env python


'''
Delete snapshot files or transfer files across machines.

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
def delete_snapshots(snapshot_index_limits=[3, 599], directory='.'):
    '''
    Delete all snapshots within snapshot_index_limits in given directory,
    except for those in snapshot_indices_keep list below.

    Parameters
    ----------
    snapshot_index_limits : list : min and max snapshot indices to consider deleting
    directory : string : directory of snapshots
    '''
    # 31 snapshots
    snapshot_indices_keep = [
        0, 1, 2,
        11, 20, 26, 33, 41, 52, 59, 67, 77, 88,
        102, 120, 142, 172,
        214, 242, 277,
        322, 382,
        412, 446, 486,
        534, 561, 580, 585, 590, 600
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

                print('* deleting {}'.format(snapshot_name))
                os.system('rm -rf ' + snapshot_name)


#===================================================================================================
# transfer files
#===================================================================================================
def transfer_snapshots(
    machine_name='stampede', from_directory='$STAMPEDE_FIRE/m12/m12i/m12i_res7000/output',
    snapshot_indices=[600], to_directory='.'):
    '''
    Transfer snapshot file[s] or directory[s] from remote machine to local.

    Parameters
    ----------
    machine_name : string : name of remote machine
        examples: 'stampede', 'pfe', 'ranch', 'lou'
        these assume that you have aliased these names in your .ssh/config
        else you need to supply, for example <username>@stampede.tacc.xsede.org
    from_directory : string : directory of snapshot file[s] on remote machine
    snapshot_indices : int or list : index[s] of snapshots to transfer
    to_directory : string : local directory to put snapshots
    '''
    snapshot_name_base = 'snap*_{:03d}*'

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

    os.system('rsync -ahvP --size-only {}:"{}" {}'.format(
              machine_name, snapshot_path_names, to_directory))


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
        snapshot_index_limits = None
        if len(sys.argv) > 3:
            snapshot_index_limits = [int(sys.argv[2]), int(sys.argv[3])]

        if len(sys.argv) > 4:
            directory = str(sys.argv[4])

        delete_snapshots(snapshot_index_limits, directory)

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
