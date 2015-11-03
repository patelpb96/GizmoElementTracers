#!/usr/bin/env python


'''
Diagnostic and utility functions for Gizmo simulations.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import os
import sys
import glob
import numpy as np
from numpy import log10, Inf  # @UnusedImport
import matplotlib.pyplot as plt
# local ----
from utilities import utility as ut
from utilities import constants as const
from utilities import plot
from utilities import simulation
from . import gizmo_io
from . import gizmo_analysis


#===================================================================================================
# simulation diagnostic
#===================================================================================================
def print_run_times(
    directory='.', cpu_number=None,
    scale_factors=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                   0.666, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    print_lines=False, get_values=False):
    '''
    Print wall [and CPU] times (average per MPI task) at input scale factors from cpu.txt for
    Gizmo simulation.

    Parameters
    ----------
    directory : string : directory of cpu.txt file
    cpu_number : int : number of CPUs used for simulation (to compute total CPU hours)
    scale_factors : array-like : list of scale factors at which to print run times
    print_lines : boolean : whether to print full lines from cpu.txt as get them
    get_values : boolean : whether to return arrays of scale factors, redshifts, run times
    '''
    scale_factors = np.array(scale_factors)

    run_times = []

    file_name = 'cpu.txt'

    file_path_name = ut.io.get_path(directory) + file_name

    file_in = open(file_path_name, 'r')

    t_i = 0
    print_next_line = False
    for line in file_in:
        if 'Time: %.3f' % scale_factors[t_i] in line or 'Time: 1,' in line:
            if print_lines:
                print(line, end='')
            print_next_line = True
        elif print_next_line:
            if print_lines:
                print(line)
            run_times.append(float(line.split()[1]))
            print_next_line = False
            t_i += 1
            if t_i >= len(scale_factors):
                break

    run_times = np.array(run_times) / 3600  # convert to {hr}
    if cpu_number:
        cpu_times = run_times * cpu_number

    # sanity check - simulation might not have run to all input scale factors
    scale_factors = scale_factors[: run_times.size]
    redshifts = 1 / scale_factors - 1

    print('# scale-factor redshift run-time-percent run-time[hr, day]', end='')
    if cpu_number:
        print(' cpu-time[hr]', end='')
    print()
    for t_i in xrange(len(run_times)):
        print('%.3f %5.2f   %5.1f%% %7.2f %5.2f' %
              (scale_factors[t_i], redshifts[t_i], 100 * run_times[t_i] / run_times.max(),
               run_times[t_i], run_times[t_i] / 24), end='')
        if cpu_number:
            print(' %7.0f' % cpu_times[t_i], end='')
        print()

    #for scale_factor in scale_factors:
    #    os.system('grep "Time: %.2f" %s --after-context=1 --max-count=2' %
    #              (scale_factor, file_path_name))

    if get_values:
        return scale_factors, redshifts, run_times


def print_run_time_ratios(
    directories=['.'], cpu_numbers=None,
    scalefactors=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                  0.666, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]):
    '''
    Print ratios of wall [and CPU] times (average per MPI taks) for input simulations at input
    scale factors from cpu.txt for Gizmo run

    Parameters
    ----------
    directories : string or list : directory[s] of cpu.txt file for each simulation
    cpu_numbers : int or list : number[s] of CPUs used for each simulation
    scalefactors : array-like : list of scale factors at which to print run times
    '''
    run_timess = []

    if np.isscalar(directories):
        directories = [directories]

    print_cpu = False
    if cpu_numbers is not None:
        if np.isscalar(cpu_numbers):
            cpu_numbers = [cpu_numbers]
        if len(cpu_numbers):
            assert len(cpu_numbers) == len(directories)
            print_cpu = True
            cpu_timess = []

    for d_i, directory in enumerate(directories):
        scalefactors, redshifts, run_times = print_run_times(
            directory, cpu_numbers[d_i], scalefactors, get_values=True)
        run_timess.append(run_times)

    run_time_num_min = Inf
    for d_i, run_times in enumerate(run_timess):
        if print_cpu:
            cpu_timess.append(run_times * cpu_numbers[d_i])
        if len(run_times) < run_time_num_min:
            run_time_num_min = len(run_times)

    # sanity check - simulations might not have run to each input scale factor
    scalefactors = scalefactors[: run_time_num_min]
    redshifts = redshifts[: run_time_num_min]

    print('# scale-factor redshift', end='')
    for _ in xrange(1, len(run_timess)):
        print(' wall-time-ratio', end='')
        if print_cpu:
            print(' cpu-time-ratio', end='')
    print()

    for a_i in xrange(run_time_num_min):
        print('%.3f %5.2f  ' % (scalefactors[a_i], redshifts[a_i]), end='')
        for d_i in xrange(1, len(run_timess)):
            print(' %7.3f' % (run_timess[d_i][a_i] / run_timess[0][a_i]), end='')
            if print_cpu:
                print(' %7.3f' % (cpu_timess[d_i][a_i] / cpu_timess[0][a_i]), end='')
        print()


def plot_halo_contamination(directory='.', snapshot_redshift=0):
    '''
    Plot contamination from lower-resolution particles in/near halo as a function of radius.

    Parameters
    ----------
    directory : string : directory of simulation (one level above directory of snapshot file)
    snapshot_redshift : float : redshift of snapshot file
    '''
    #position_dif_max = 5  # {kpc comoving} - if centers differ by more than this, print warning
    radius_bin_wid = 0.02
    radius_lim_phys = [1, 4000]  # {kpc physical}
    radius_lim_vir = [0.01, 10]  # {units of R_halo}
    virial_kind = '200m'

    os.chdir(directory)

    part = gizmo_io.Gizmo.read_snapshot(
        ['dark', 'dark.2'], 'redshift', snapshot_redshift, 'output',
        ['position', 'mass', 'potential'], force_float32=True, assign_center=True)

    #part.center_position = ut.particle.get_center_position(
    #    part, 'dark', 'center-of-mass', compare_centers=True)

    halo_radius, _halo_mass = ut.particle.get_halo_radius_mass(part, 'all', virial_kind=virial_kind)

    gizmo_analysis.plot_mass_contamination(
        part, radius_lim_phys, radius_bin_wid, halo_radius=halo_radius, scale_to_halo_radius=False,
        write_plot=True, plot_directory='plot')

    gizmo_analysis.plot_mass_contamination(
        part, radius_lim_vir, radius_bin_wid, halo_radius=halo_radius,
        scale_to_halo_radius=True, write_plot=True, plot_directory='plot')


def print_properties_extrema_all_snapshots(
    directory='.', species_property_dict={'gas': ['smooth.length', 'density']}):
    '''
    Read every snapshot, for each input properties, get its extremum at each snapshot.
    Print statistics of this across all snapshots.

    directory : string : directory of simulation (one level above directory of snapshot file)
    species_property_dict : dict : keys = species, values are string or list of property[s]
    '''
    Say = ut.io.SayClass(print_properties_extrema_all_snapshots)

    property_statistic = {
        'smooth.length': {'function.name': 'min', 'function': np.min},
        'density': {'function.name': 'max', 'function': np.max},
    }

    directory = ut.io.get_path(directory)

    Snapshot = simulation.SnapshotClass()
    Snapshot.read_snapshots(directory=directory)

    species_read = species_property_dict.keys()

    properties_read = []
    for spec_name in species_property_dict:
        prop_names = species_property_dict[spec_name]
        if np.isscalar(prop_names):
            prop_names = [prop_names]

        prop_dict = {}
        for prop_name in species_property_dict[spec_name]:
            prop_dict[prop_name] = []

            if prop_name not in properties_read:
                properties_read.append(prop_name)

        # re-assign property list as dictionary so can store list of values
        species_property_dict[spec_name] = prop_dict

    for snapshot_i in Snapshot['index']:
        try:
            part = gizmo_io.Gizmo.read_snapshot(
                species_read, 'index', snapshot_i, directory + 'output', properties_read,
                sort_dark_by_id=False, force_float32=True, assign_center=False)

            for spec_name in species_property_dict:
                for prop_name in species_property_dict[spec_name]:
                    if prop_name in part[spec_name]:
                        prop_ext = property_statistic[prop_name]['function'](
                            part[spec_name][prop_name])
                        species_property_dict[spec_name][prop_name].append(prop_ext)
                    else:
                        Say.say('! %s %s not in particle dictionary' % (spec_name, prop_name))
        except:
            Say.say('! cannot read snapshot index %d in %s' % (snapshot_i, directory))

    Statistic = ut.math.StatisticClass()

    for spec_name in species_property_dict:
        for prop_name in species_property_dict[spec_name]:
            prop_func_name = property_statistic[prop_name]['function.name']
            prop_values = np.array(species_property_dict[spec_name][prop_name])

            if 'gas' in spec_name and 'density' in prop_name:
                prop_values *= const.proton_per_sun * const.kpc_per_cm ** 3  # convert to {cm ^ -3}

            Statistic.stat = Statistic.get_statistic_dict(prop_values)

            Say.say('\n%s %s %s:' % (spec_name, prop_name, prop_func_name))
            for stat_name in ['min', 'percent.16', 'median', 'percent.84', 'max']:
                Say.say('%10s = %.3f' % (stat_name, Statistic.stat[stat_name]))

            #Statistic.print_statistics()


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
# code performance and scaling
#===================================================================================================
def plot_scaling(
    plot_kind='strong', time_kind='cpu', axis_x_scaling='log', axis_y_scaling='lin',
    write_plot=False, plot_directory='.'):
    '''
    .
    '''
    dark = {
        'ref12': {'particle.num': 8.82e6, 'cpu.num': 64, 'cpu.time': 385, 'wall.time': 6.0},
        'ref13': {'particle.num': 7.05e7, 'cpu.num': 512, 'cpu.time': 7135, 'wall.time': 13.9},
        'ref14': {'particle.num': 5.64e8, 'cpu.num': 2048, 'cpu.time': 154355, 'wall.time': 75.4},
    }

    mfm = {
        'ref12': {'particle.num': 8.82e6 * 2, 'cpu.num': 512, 'cpu.time': 116482, 'wall.time': 228},
        'ref13': {'particle.num': 7.05e7 * 2, 'cpu.num': 2048, 'cpu.time': 1322781,
                  'wall.time': 646},
        #'ref14': {'particle.num': 5.64e8 * 2, 'cpu.num': 8192, 'cpu.time': 568228,
        #          'wall.time': 69.4},
        # projected
        'ref14': {'particle.num': 5.64e8 * 2, 'cpu.num': 8192, 'cpu.time': 1.95e7,
                  'wall.time': 2380},
    }

    mfm_ref14 = {
        2048: {'particle.num': 5.64e8 * 2, 'cpu.num': 2048, 'wall.time': 15.55, 'cpu.time': 31850},
        4096: {'particle.num': 5.64e8 * 2, 'cpu.num': 4096, 'wall.time': 8.64, 'cpu.time': 35389},
        8192: {'particle.num': 5.64e8 * 2, 'cpu.num': 8192, 'wall.time': 4.96, 'cpu.time': 40632},
        16384: {'particle.num': 5.64e8 * 2, 'cpu.num': 16384, 'wall.time': 4.57, 'cpu.time': 74875},
    }

    # plot ----------
    plt.clf()
    plt.minorticks_on()
    fig = plt.figure(1)
    subplot = fig.add_subplot(111)
    fig.subplots_adjust(left=0.21, right=0.95, top=0.96, bottom=0.16, hspace=0.03, wspace=0.03)

    plot_func = plot.get_plot_function(subplot, axis_x_scaling, axis_y_scaling)

    if plot_kind == 'strong':
        cpu_nums = [k for k in mfm_ref14]
        # 2x == convert from a = 0.068 to a = 0.1
        if time_kind == 'cpu':
            times = [mfm_ref14[k]['cpu.time'] * 2 for k in mfm_ref14]
        elif time_kind == 'wall':
            times = [mfm_ref14[k]['wall.time'] * 2 for k in mfm_ref14]

        subplot.set_xlim([1e3, 2.5e4])
        subplot.set_xlabel('core number')

        if time_kind == 'cpu':
            subplot.set_ylim([0, 1.6e5])
            subplot.set_ylabel('CPU time to $z = 9$ [hr]')
        elif time_kind == 'wall':
            subplot.set_ylim([0, 35])
            subplot.set_ylabel('wall time to $z = 9$ [hr]')

        plot_func(cpu_nums, times, '*-', linewidth=2.0, color='blue')

        subplot.text(0.05, 0.1, 'strong scaling:\nparticle number = 1.1e9', color='black',
                     transform=subplot.transAxes)

    elif plot_kind == 'weak':
        dm_particle_nums = np.array([dark[k]['particle.num'] for k in sorted(dark.keys())])
        mfm_particle_nums = np.array([mfm[k]['particle.num'] for k in sorted(mfm.keys())])

        if time_kind == 'cpu':
            dm_times = np.array([dark[k]['cpu.time'] for k in sorted(dark.keys())])
            mfm_times = np.array([mfm[k]['cpu.time'] for k in sorted(mfm.keys())])
        elif time_kind == 'wall':
            ratio_ref = mfm['ref14']['particle.num'] / mfm['ref14']['cpu.num']
            dm_times = np.array([dark[k]['wall.time'] *
                                 ratio_ref / (dark[k]['particle.num'] / dark[k]['cpu.num'])
                                 for k in sorted(dark.keys())])
            mfm_times = np.array([mfm[k]['wall.time'] *
                                  ratio_ref / (mfm[k]['particle.num'] / mfm[k]['cpu.num'])
                                  for k in sorted(mfm.keys())])

        subplot.set_xlim([6e6, 1.5e9])
        subplot.set_xlabel('particle number')

        if time_kind == 'cpu':
            subplot.set_ylim([2e2, 4e7])
            subplot.set_ylabel('CPU time to $z = 0$ [hr]')
        elif time_kind == 'wall':
            subplot.set_ylim([4, 4000])
            subplot.set_ylabel('wall time to $z = 0$ [hr]')
            subplot.text(0.05, 0.5,
                         'weak scaling:\nfixed particle number / core = %.1e' % ratio_ref,
                         color='black', transform=subplot.transAxes)

        plot_func(dm_particle_nums, dm_times, '.-', linewidth=2.0, color='red')
        plot_func(mfm_particle_nums[:-1], mfm_times[:-1], '*-', linewidth=2.0, color='blue')
        plot_func(mfm_particle_nums[1:], mfm_times[1:], '*--', linewidth=2.0, color='blue',
                  alpha=0.7)

    plot_name = 'test'
    plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise ValueError('must specify function kind: runtime, contamination, extreme, delete')

    function_kind = str(sys.argv[1])
    assert ('runtime' in function_kind or 'contamination' in function_kind or
            'extreme' in function_kind or 'delete' in function_kind)

    directory = '.'

    if 'runtime' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        cpu_number = None
        if len(sys.argv) > 3:
            cpu_number = int(sys.argv[3])

        print_run_times(directory, cpu_number)

    elif 'contamination' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_redshift = 0
        if len(sys.argv) > 3:
            snapshot_redshift = float(sys.argv[2])

        plot_halo_contamination(directory, snapshot_redshift)

    elif 'extreme' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        print_properties_extrema_all_snapshots(directory)

    elif 'delete' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        delete_snapshots(directory)

    else:
        print('! not recognize function kind')
