#!/usr/bin/env python


'''
Diagnostic and utility functions for Gizmo simulations.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import glob
# local ----
from utilities import utility as ut


#===================================================================================================
# simulation diagnostic
#===================================================================================================
def print_run_times(
    directory='.',
    scalefactors=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                  0.666, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    print_lines=False):
    '''
    Print run times (average per MPI taks) at given scale factors from cpu.txt for Gizmo run.

    Parameters
    ----------
    directory : string : directory of cpu.txt file
    scalefactors : array-like : list of scale factors at which to print run times
    print_lines : boolean : whether to print full lines from cpu.txt as get them
    '''
    scalefactors = np.array(scalefactors)

    run_times = []

    file_name = 'cpu.txt'

    file_path_name = ut.io.get_path(directory) + file_name

    file_in = open(file_path_name, 'r')

    a_i = 0
    print_next_line = False
    for line in file_in:
        if 'Time: %.3f' % scalefactors[a_i] in line or 'Time: 1,' in line:
            if print_lines:
                print(line, end='')
            print_next_line = True
        elif print_next_line:
            if print_lines:
                print(line)
            run_times.append(float(line.split()[1]))
            print_next_line = False
            a_i += 1
            if a_i >= len(scalefactors):
                break

    run_times = np.array(run_times)

    print('# scale-factor redshift run-time-percent run-time[hr]')
    for a_i in xrange(len(run_times)):
        print('%.3f %5.2f  %5.1f  %8.2f' %
              (scalefactors[a_i], 1 / scalefactors[a_i] - 1,
               100 * run_times[a_i] / run_times.max(), run_times[a_i] / 3600))

    #for scale_factor in scale_factors:
    #    os.system('grep "Time: %.2f" %s --after-context=1 --max-count=2' %
    #              (scale_factor, file_path_name))


def plot_halo_contamination(directory='.', snapshot_redshift=0):
    '''
    Plot contamination from lower-resolution particles in/near halo as a function of radius.

    Parameters
    ----------
    directory : string : directory of simulation (one level above directory of snapshot file)
    snapshot_redshift : float : redshift of snapshot file
    '''
    from . import gizmo_io
    from . import gizmo_analysis

    position_dif_max = 5  # {kpc comoving} - if centers differ by more than this, print warning
    radius_bin_wid = 0.02
    radius_lim_phys = [1, 4000]  # {kpc physical}
    radius_lim_vir = [0.01, 10]  # {units of R_halo}
    virial_kind = '200m'

    os.chdir(directory)

    part = gizmo_io.Gizmo.read_snapshot(
        ['dark', 'dark.2'], 'redshift', snapshot_redshift, 'output',
        ['position', 'mass', 'potential'], force_float32=True, assign_center=False)

    center_position_dark_cm = ut.particle.get_center_position(part, 'dark')
    center_pos_dark_pot = part['dark']['position'][np.argmin(part['dark']['potential'])]

    print('# dark center position {kpc comoving}')
    print('  center-of-mass: ', end='')
    ut.io.print_array(center_position_dark_cm, '%.3f')
    print('  potential min:  ', end='')
    ut.io.print_array(center_pos_dark_pot, '%.3f')

    for dimen_i in xrange(center_position_dark_cm.size):
        position_dif = np.abs(center_position_dark_cm[dimen_i] - center_pos_dark_pot[dimen_i])
        if position_dif > position_dif_max:
            print('! position-%d offset = %.3f' % (dimen_i, position_dif))

    part.center_position = center_position_dark_cm

    halo_radius = ut.particle.get_halo_radius(part, 'all', virial_kind=virial_kind)

    gizmo_analysis.plot_mass_contamination(
        part, radius_lim_phys, radius_bin_wid, halo_radius=halo_radius, scale_to_halo_radius=False,
        write_plot=True, plot_directory='plot')

    gizmo_analysis.plot_mass_contamination(
        part, radius_lim_vir, radius_bin_wid, halo_radius=halo_radius,
        scale_to_halo_radius=True, write_plot=True, plot_directory='plot')


#===================================================================================================
# simulation utility
#===================================================================================================
def delete_snapshots(directory='.'):
    '''
    Delete all snapshots in given directory, except for those given below.

    Parameters
    ----------
    directory : string : directory of snapshots
    '''
    snapshot_indices = [0, 1, 2, 14, 18, 22, 28, 35, 45, 59, 80, 95, 114, 142, 184, 253, 390, 400]

    snapshot_name_bases = ['snapshot_*.hdf5', 'snapdir_*']

    os.chdir(directory)

    for snapshot_name_base in snapshot_name_bases:
        snapshot_names = glob.glob(snapshot_name_base)

        for snapshot_name in snapshot_names:
            keep = False
            for snapshot_index in snapshot_indices:
                if snapshot_name in ['snapshot_%03d.hdf5' % snapshot_index,
                                     'snapdir_%03d' % snapshot_index]:
                    keep = True

            if not keep:
                os.system('rm -rf %s' % snapshot_name)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise ValueError('must specify function kind: runtime, contamination, delete')

    function_kind = str(sys.argv[1])
    assert ('runtime' in function_kind or 'contamination' in function_kind or
            'delete' in function_kind)

    if 'runtime' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        print_run_times(directory)

    elif 'contamination' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_redshift = 0
        if len(sys.argv) > 3:
            snapshot_redshift = float(sys.argv[2])

        plot_halo_contamination(directory, snapshot_redshift)

    elif 'delete' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        delete_snapshots(directory)

    else:
        print('! not recognize function kind')
