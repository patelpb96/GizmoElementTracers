#!/usr/bin/env python


'''
Plot contamination from lower-resolution particles versus radius around halo for Gizmo simulation.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
# local ----
from utilities import utility as ut
from gizmo import gizmo_io
from gizmo import gizmo_analysis


if len(sys.argv) > 1:
    directory = str(sys.argv[1])
else:
    directory = '.'

if len(sys.argv) > 2:
    snapshot_index = int(sys.argv[2])
else:
    snapshot_index = 400

pos_dif_max = 5  # {kpc comoving}, more than this print warning
radius_bin_num = 100
radius_lim_phys = [1, 4000]  # {kpc comoving}
radius_lim_vir = [0.01, 10]  # {units of R_halo}
virial_kind = '200m'


os.chdir(directory)

part = gizmo_io.Gizmo.read_snapshot(['all'], snapshot_index, 'output')

center_pos_dark_cm = gizmo_analysis.get_center_position(part, 'dark')
center_pos_dark_pot = part['dark']['position'][np.argmin(part['dark']['potential'])]

print('# dark center_pos {kpc comoving}')
print('  center-of-mass: ', end='')
ut.io.print_array(center_pos_dark_cm, '%.3f')

print('  potential.min:  ', end='')
ut.io.print_array(center_pos_dark_pot, '%.3f')

for dimen_i in xrange(center_pos_dark_cm.size):
    pos_dif = np.abs(center_pos_dark_cm[dimen_i] - center_pos_dark_pot[dimen_i])
    if pos_dif > pos_dif_max:
        print('! position-%d offset = %.3f' % (dimen_i, pos_dif))

center_pos = center_pos_dark_cm

halo_radius = gizmo_analysis.get_halo_radius(part, 'all', center_pos, virial_kind)

gizmo_analysis.plot_mass_contamination(
    part, center_pos, radius_lim_phys, radius_bin_num, halo_radius=halo_radius, write_plot=True,
    plot_directory='plot')

gizmo_analysis.plot_mass_contamination(
    part, center_pos, radius_lim_vir, radius_bin_num, halo_radius=halo_radius,
    scale_to_halo_radius=True, write_plot=True, plot_directory='plot')
