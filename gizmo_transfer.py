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
from utilities import utility as ut


def sync_snapshots(
    machine_name='stampede', directory='$STAMPEDE_SCRATCH/m12i_ref12_rad4_wk-area/output',
    snapshot_kind='file', snapshot_indices=400, to_directory='.'):
    '''
    machine_name : string : name of host machine
    directory : string : directory of snapshot file on host machine
    snapshot_kind : string : 'file' or 'directory'
    snapshot indices : int or list : index[s] of snapshots
    to_directory : string : local directory to put snapshots
    '''
    if snapshot_kind == 'file':
        snapshot_name_base = 'snapshot_%.3d.hdf5'
    elif snapshot_kind == 'directory':
        snapshot_name_base = 'snapdir_%.3d'
    else:
        raise ValueError('not recognize snapshot_kind = %s' % snapshot_kind)

    #if machine_name == 'stampede':
    #    directory = '$STAMPEDE_SCRATCH/' + directory
    #elif machine_name == 'zwicky':
    #    directory = '$ZWICKY_SCRATCH/' + directory
    #elif machine_name == 'ranch':
    #    directory = '$RANCH_HOME/stampede/' + directory

    directory = ut.io.get_path(directory)

    if np.isscalar(snapshot_indices):
        snapshot_indices = [snapshot_indices]

    for snapshot_index in snapshot_indices:
        snapshot_name = snapshot_name_base % snapshot_index
        os.system('rsync -ahvPz %s:%s%s %s' %
                  (machine_name, directory, snapshot_name, to_directory))


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) < 5:
        raise ValueError('must specify function kind: runtime, contamination, delete')

    machine_name = str(sys.argv[1])
    directory = str(sys.argv[2])
    snapshot_kind = str(sys.argv[3])
    snapshot_index = int(sys.argv[4])

    sync_snapshots(machine_name, directory, snapshot_kind, snapshot_index)
