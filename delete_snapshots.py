#!/usr/bin/env python


'''
Delete Gizmo snapshots in given directory, except for those given below.

@author: Andrew Wetzel
'''


from __future__ import absolute_import, division, print_function
import os
import sys


snapshot_indices = [0, 1, 2, 14, 18, 22, 28, 35, 45, 59, 80, 95, 114, 142, 184, 253, 390, 400]

if len(sys.argv) > 1:
    directory = str(sys.argv[1])
else:
    directory = '.'

os.chdir(directory)

os.system('mkdir temp')
for snapshot_index in snapshot_indices:
    os.system('mv snapshot_%03d.hdf5 temp/.' % snapshot_index)
for snapshot_index in snapshot_indices:
    os.system('mv snapdir_%03d temp/.' % snapshot_index)
os.system('rm -rf snapshot_???.hdf5')
os.system('rm -rf snapdir_???')
os.system('mv temp/* .')
os.system('rm -rf temp')
