#!/usr/bin/env python


'''
Delete Gizmo snapshots in given directory, except for those given below.

@author: Andrew Wetzel
'''


from __future__ import absolute_import, division, print_function
import os
import sys
import glob


snapshot_indices = [0, 1, 2, 14, 18, 22, 28, 35, 45, 59, 80, 95, 114, 142, 184, 253, 390, 400]

if len(sys.argv) > 1:
    directory = str(sys.argv[1])
else:
    directory = '.'

os.chdir(directory)

snapshot_name_bases = ['snapshot_*.hdf5', 'snapdir_*']

for snapshot_name_base in snapshot_name_bases:
    snapshot_names = glob.glob(snapshot_name_base)

    for snapshot_name in snapshot_names:
        keep = False
        for snapshot_index in snapshot_indices:
            if snapshot_name in ['snapshot_%03d.hdf5', 'snapdir_%03d']:
                keep = True

        if not keep:
            os.system('rm -rf %s' % snapshot_name)
