#!/usr/bin/env python


'''
Transfer/sync files across machines.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
# local ----
import utilities as ut


def sync_snapshots(
    machine_name='stampede', from_directory='$STAMPEDE_SCRATCH/m12i_ref13/output',
    snapshot_kind='file', snapshot_indices=400, to_directory='.'):
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

    os.system('rsync -ahvPz {}:"{}" {}'.format(machine_name, snapshot_path_names, to_directory))


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) < 5:
        raise ValueError('imports: machine_name directory snapshot_kind snapshot_time_file_name')

    machine_name = str(sys.argv[1])
    from_directory = str(sys.argv[2])
    snapshot_kind = str(sys.argv[3])
    snapshot_time_file_name = str(sys.argv[4])

    Snapshot = ut.simulation.SnapshotClass()
    Snapshot.read_snapshots(snapshot_time_file_name)

    sync_snapshots(machine_name, from_directory, snapshot_kind, Snapshot['index'])
