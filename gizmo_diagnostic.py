#!/usr/bin/env python


'''
Diagnostic and utility functions for Gizmo simulations.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import collections
import os
import sys
import glob
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut
from . import gizmo_io
from . import gizmo_analysis


#===================================================================================================
# utility
#===================================================================================================
def get_cpu_numbers(simulation_directory='.', runtime_file_name='gizmo.out'):
    '''
    Get number of MPI tasks and OpenMP threads from run-time file.
    If cannot find any, default to 1.

    Parameters
    ----------
    simulation_directory : string : directory of simulation
    runtime_file_name : string : name of run-time file name (set in submission script)

    Returns
    -------
    mpi_number : int : number of MPI tasks
    omp_number : int : number of OpenMP threads per MPI task
    '''
    loop_number_max = 1000

    Say = ut.io.SayClass(get_cpu_numbers)
    file_path_name = ut.io.get_path(simulation_directory) + runtime_file_name
    file_in = open(file_path_name, 'r')

    loop_i = 0
    mpi_number = None
    omp_number = None

    for line in file_in:
        if 'MPI tasks' in line:
            mpi_number = int(line.split()[2])
        elif 'OpenMP threads' in line:
            omp_number = int(line.split()[1])

        if mpi_number and omp_number:
            break

        loop_i += 1
        if loop_i > loop_number_max:
            break

    if mpi_number:
        Say.say('MPI tasks = {}'.format(mpi_number))
    else:
        Say.say('! unable to find number of MPI tasks')
        mpi_number = 1

    if omp_number:
        Say.say('OpenMP threads = {}'.format(omp_number))
    else:
        Say.say('did not find any OpenMP threads')
        omp_number = 1

    return mpi_number, omp_number


#===================================================================================================
# simulation diagnostic
#===================================================================================================
def print_run_times(
    simulation_directory='.', output_directory='output/', runtime_file_name='gizmo.out',
    scale_factors=[
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
        0.666, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    wall_time_restart=0, get_values=False):
    '''
    Print wall [and CPU] times (average per MPI task) at input scale-factors from cpu.txt for
    Gizmo simulation.

    Parameters
    ----------
    simulation_directory : string : directory of simulation
    output_directory : string : directory of output files within simulation directory
    runtime_file_name : string : name of run-time file name (set in submission script)
    scale_factors : array-like : list of scale-factors at which to print run times
    wall_time_restart : float : wall time [sec] of previous run (if restarted from snapshot)
    get_values : boolean : whether to return arrays of scale-factors, redshifts, run times

    Returns
    -------
    scale_factors, redshifts, wall_times, cpu_times : arrays : return these if get_values is True
    '''
    def get_scale_factor_string(scale_factor):
        if scale_factor == 1:
            scale_factor_string = '1'
        else:
            scale_factor_string = '{}'.format(scale_factor)
        return scale_factor_string

    file_name = 'cpu.txt'

    file_path_name = (ut.io.get_path(simulation_directory) + ut.io.get_path(output_directory) +
                      file_name)
    file_in = open(file_path_name, 'r')

    scale_factors = ut.array.arrayize(scale_factors)
    wall_times = []

    i = 0
    scale_factor = 'Time: {}'.format(get_scale_factor_string(scale_factors[i]))
    print_next_line = False
    for line in file_in:
        if print_next_line:
            wall_times.append(float(line.split()[1]))
            print_next_line = False
            i += 1
            if i >= len(scale_factors):
                break
            else:
                scale_factor = 'Time: {}'.format(get_scale_factor_string(scale_factors[i]))
        elif scale_factor in line:
            print_next_line = True

    """
    for i, a in enumerate(scale_factors):
        scale_factor = 'Time: {}'.format(get_scale_factor_string(a))
        os.system('grep "{}" {} --after-context=1 --max-count=1'.format(
                  scale_factor, file_path_name))
    """

    wall_times = np.array(wall_times)

    if wall_time_restart and len(wall_times) > 1:
        for i in range(1, len(wall_times)):
            if wall_times[i] < wall_times[i - 1]:
                break
        wall_times[i:] += wall_time_restart

    wall_times /= 3600  # convert to [hr]

    # get cpu number from run-time file
    mpi_number, omp_number = get_cpu_numbers(simulation_directory, runtime_file_name)
    cpu_number = mpi_number * omp_number
    cpu_times = wall_times * cpu_number

    # sanity check - simulation might not have run to all input scale-factors
    scale_factors = scale_factors[: wall_times.size]
    redshifts = 1 / scale_factors - 1

    print('# scale-factor redshift wall-time-percent wall-time[hr, day]', end='')
    if cpu_number:
        print(' cpu-time[hr]', end='')
    print()
    for t_i in range(len(wall_times)):
        print('{:.3f} {:5.2f}   {:5.1f}% {:7.2f} {:5.2f}'.format(
              scale_factors[t_i], redshifts[t_i], 100 * wall_times[t_i] / wall_times.max(),
              wall_times[t_i], wall_times[t_i] / 24), end='')
        if cpu_number:
            print(' {:7.0f}'.format(cpu_times[t_i]), end='')
        print()

    if get_values:
        return scale_factors, redshifts, wall_times, cpu_times


def print_run_times_ratios(
    simulation_directories=['.'], output_directory='output/', runtime_file_name='gizmo.out',
    scale_factors=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                   0.666, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    wall_times_restart=[0]):
    '''
    Print ratios of wall times and CPU times (average per MPI taks) for input simulation directories
    at input scale-factors from cpu.txt for Gizmo run.
    'Reference' simulation is first in list.

    Parameters
    ----------
    simulation_directories : string or list : directory[s] of simulation[s]
    output_directory : string : directory of output files within simulation directory
    runtime_file_name : string : name of run-time file name (set in submission script)
    scale_factors : array-like : list of scale-factors at which to print run times
    wall_times_restart : float or list : wall time[s] [sec] of previous run[s] (snapshot restart)
    '''
    wall_timess = []
    cpu_timess = []

    if np.isscalar(simulation_directories):
        simulation_directories = [simulation_directories]
    if np.isscalar(wall_times_restart):
        wall_times_restart = [wall_times_restart]
        if len(wall_times_restart) != len(simulation_directories):
            wall_times_restart = np.zeros(len(simulation_directories))

    for d_i, directory in enumerate(simulation_directories):
        scale_factors, redshifts, wall_times, cpu_times = print_run_times(
            directory, output_directory, runtime_file_name, scale_factors, wall_times_restart[d_i],
            get_values=True)
        wall_timess.append(wall_times)
        cpu_timess.append(cpu_times)

    snapshot_number_min = Inf
    for d_i, wall_times in enumerate(wall_timess):
        if len(wall_times) < snapshot_number_min:
            snapshot_number_min = len(wall_times)

    # sanity check - simulations might not have run to each input scale-factor
    scale_factors = scale_factors[: snapshot_number_min]
    redshifts = redshifts[: snapshot_number_min]

    print('# scale-factor redshift', end='')
    for _ in range(1, len(wall_timess)):
        print(' wall-time-ratio', end='')
        print(' cpu-time-ratio', end='')
    print()

    for a_i in range(snapshot_number_min):
        print('{:.3f} {:5.2f}  '.format(scale_factors[a_i], redshifts[a_i]), end='')
        for d_i in range(1, len(wall_timess)):
            print(' {:7.2f}'.format(wall_timess[d_i][a_i] / wall_timess[0][a_i]), end='')
            print(' {:7.2f}'.format(cpu_timess[d_i][a_i] / cpu_timess[0][a_i]), end='')
        print()


def print_properties_statistics(
    species_names='all', snapshot_number_kind='index', snapshot_number=600,
    simulation_directory='.', snapshot_directory='output/'):
    '''
    For each property of each species in particle catalog, print range and median.

    Parameters
    ----------
    species_names : string or list : species to print
    snapshot_number_kind : string : input snapshot number kind: index, redshift
    snapshot_number : int or float : index (number) of snapshot file
    simulation_directory : root directory of simulation
    snapshot_directory: string : directory of snapshot files within simulation_directory

    Returns
    -------
    part : dict : catalog of particles
    '''
    species_names = ut.array.arrayize(species_names)
    if 'all' in species_names:
        species_names = ['dark.2', 'dark', 'star', 'gas']

    part = gizmo_io.Read.read_snapshots(
        species_names, snapshot_number_kind, snapshot_number, simulation_directory,
        snapshot_directory, '', None, None, assign_center=False,
        separate_dark_lowres=False, sort_dark_by_id=False, force_float32=False)

    gizmo_analysis.print_properties_statistics(part, species_names)


def print_properties_snapshots(
    simulation_directory='.', snapshot_directory='output',
    species_property_dict={'gas': ['smooth.length', 'density.number']}):
    '''
    For each input property, get its extremum at each snapshot.
    Print statistics of property across all snapshots.

    Parameters
    ----------
    simulation_directory : string : directory of simulation
    snapshot_directory : string : directory of snapshot files
    species_property_dict : dict : keys = species, values are string or list of property[s]
    '''
    element_indices = [0, 1]

    property_statistic = {
        'smooth.length': {'function.name': 'min', 'function': np.min},
        'density': {'function.name': 'max', 'function': np.max},
        'density.number': {'function.name': 'max', 'function': np.max},
    }

    Say = ut.io.SayClass(print_properties_snapshots)

    simulation_directory = ut.io.get_path(simulation_directory)

    Snapshot = ut.simulation.SnapshotClass()
    Snapshot.read_snapshots(directory=simulation_directory)

    species_read = species_property_dict.keys()

    properties_read = []
    for spec_name in species_property_dict:
        prop_names = species_property_dict[spec_name]
        if np.isscalar(prop_names):
            prop_names = [prop_names]

        prop_dict = {}
        for prop_name in species_property_dict[spec_name]:
            prop_dict[prop_name] = []

            prop_name_read = prop_name.replace('.number', '')
            if prop_name_read not in properties_read:
                properties_read.append(prop_name_read)

            if '.number' in prop_name and 'massfraction' not in properties_read:
                properties_read.append('massfraction')

        # re-assign property list as dictionary so can store list of values
        species_property_dict[spec_name] = prop_dict

    for snapshot_i in Snapshot['index']:
        try:
            part = gizmo_io.Read.read_snapshots(
                species_read, 'index', snapshot_i, simulation_directory, snapshot_directory, '',
                properties_read, element_indices, assign_center=False, sort_dark_by_id=False,
                force_float32=False)

            for spec_name in species_property_dict:
                for prop_name in species_property_dict[spec_name]:
                    try:
                        prop_ext = property_statistic[prop_name]['function'](
                            part[spec_name].prop(prop_name))
                        species_property_dict[spec_name][prop_name].append(prop_ext)
                    except:
                        Say.say('! {} {} not in particle dictionary'.format(spec_name, prop_name))
        except:
            Say.say('! cannot read snapshot index {} in {}'.format(
                    snapshot_i, simulation_directory + snapshot_directory))

    Statistic = ut.statistic.StatisticClass()

    for spec_name in species_property_dict:
        for prop_name in species_property_dict[spec_name]:
            prop_func_name = property_statistic[prop_name]['function.name']
            prop_values = np.array(species_property_dict[spec_name][prop_name])

            Statistic.stat = Statistic.get_statistic_dict(prop_values)

            Say.say('\n{} {} {}:'.format(spec_name, prop_name, prop_func_name))
            for stat_name in ['min', 'percent.16', 'median', 'percent.84', 'max']:
                Say.say('{:10s} = {:.3f}'.format(stat_name, Statistic.stat[stat_name]))

            #Statistic.print_statistics()


def plot_contamination(directory='.', redshift=0):
    '''
    Plot contamination from lower-resolution particles around halo as a function of distance.

    Parameters
    ----------
    directory : string : directory of simulation (one level above directory of snapshot file)
    redshift : float : redshift of snapshot
    '''
    distance_bin_width = 0.01
    distance_limits_phys = [1, 4000]  # [kpc physical]
    distance_limits_halo = [0.01, 10]  # [units of R_halo]
    virial_kind = '200m'

    os.chdir(directory)

    part = gizmo_io.Read.read_snapshots(
        ['dark', 'dark.2'], 'redshift', redshift, directory,
        property_names=['position', 'mass', 'potential'], force_float32=True, assign_center=True)

    halo_radius, _halo_mass = ut.particle.get_halo_radius_mass(part, 'all', virial_kind)

    gizmo_analysis.plot_mass_contamination(
        part, distance_limits_phys, distance_bin_width, halo_radius=halo_radius,
        scale_to_halo_radius=False, write_plot=True, plot_directory='plot')

    gizmo_analysis.plot_mass_contamination(
        part, distance_limits_halo, distance_bin_width, halo_radius=halo_radius,
        scale_to_halo_radius=True, write_plot=True, plot_directory='plot')


#===================================================================================================
# simulation utility
#===================================================================================================
def delete_snapshots(directory='.', snapshot_index_limits=[2, 300]):
    '''
    Delete all snapshots in given directory, except for those given below.

    Parameters
    ----------
    directory : string : directory of snapshots
    snapshot_index_limits : list : min and max snapshot indices to consider deleting
    '''
    snapshot_indices = [
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
            if (snapshot_index not in snapshot_indices and
                    snapshot_index >= min(snapshot_index_limits) and
                    snapshot_index <= max(snapshot_index_limits)):
                print('deleting {}'.format(snapshot_name))
                os.system('rm -rf ' + snapshot_name)


#===================================================================================================
# simulation performance and scaling
#===================================================================================================
def plot_scaling(
    scaling_kind='strong', resolution='ref14', time_kind='cpu',
    axis_x_scaling='log', axis_y_scaling='linear', write_plot=False, plot_directory='.'):
    '''
    Print simulation run times (all or CPU).
    'speedup' := WT(1 CPU) / WT(N CPU) =
    'efficiency' := WT(1 CPU) / WT(N CPU) / N = CT(1 CPU) / CT(N CPU)

    Parameters
    ----------
    scaling_kind : string : 'strong', 'weak'
    time_kind : string : 'cpu', 'wall', 'speedup', 'efficiency'
    axis_x_scaling : string : scaling along x-axis: 'log', 'linear'
    axis_y_scaling : string : scaling along y-axis: 'log', 'linear'
    write_plot : boolean : whether to write plot to file
    plot_directory : string : directory to write plot file
    '''
    weak_dark = {
        'ref12': {'particle.number': 8.82e6, 'cpu.number': 64,
                  'cpu.time': 385, 'wall.time': 6.0},
        'ref13': {'particle.number': 7.05e7, 'cpu.number': 512,
                  'cpu.time': 7135, 'wall.time': 13.9},
        'ref14': {'particle.number': 5.64e8, 'cpu.number': 2048,
                  'cpu.time': 154355, 'wall.time': 75.4},
    }

    weak_baryon = {
        'ref11': {'particle.number': 1.10e6 * 2, 'cpu.number': 32,
                  'cpu.time': 1003, 'wall.time': 31.34 * 1.5},
        'ref12': {'particle.number': 8.82e6 * 2, 'cpu.number': 512,
                  'cpu.time': 33143, 'wall.time': 64.73},
        'ref13': {'particle.number': 7.05e7 * 2, 'cpu.number': 2048,
                  'cpu.time': 1092193, 'wall.time': 350.88},
        #'ref14': {'particle.number': 5.64e8 * 2, 'cpu.number': 8192,
        #          'cpu.time': 568228, 'wall.time': 69.4},
        # projected
        #'ref14': {'particle.number': 5.64e8 * 2, 'cpu.number': 8192,
        #          'cpu.time': 1.95e7, 'wall.time': 2380},
    }

    strong_baryon = collections.OrderedDict()
    strong_baryon['ref14'] = collections.OrderedDict()
    strong_baryon['ref14'][2048] = {'particle.number': 5.64e8 * 2, 'cpu.number': 2048,
                                    'wall.time': 15.55, 'cpu.time': 31850}
    strong_baryon['ref14'][4096] = {'particle.number': 5.64e8 * 2, 'cpu.number': 4096,
                                    'wall.time': 8.64, 'cpu.time': 35389}
    strong_baryon['ref14'][8192] = {'particle.number': 5.64e8 * 2, 'cpu.number': 8192,
                                    'wall.time': 4.96, 'cpu.time': 40632}
    strong_baryon['ref14'][16384] = {'particle.number': 5.64e8 * 2, 'cpu.number': 16384,
                                     'wall.time': 4.57, 'cpu.time': 74875}

    # did not have time to run these, so just scale down from ref14
    # scaled to run time to z = 3 using 2048
    strong_baryon['ref13'] = collections.OrderedDict()
    strong_baryon['ref13'][512] = {'particle.number': 7e7 * 2, 'cpu.number': 512,
                                   'wall.time': 72.23, 'cpu.time': 36984}
    strong_baryon['ref13'][1024] = {'particle.number': 7e7 * 2, 'cpu.number': 1024,
                                    'wall.time': 40.13, 'cpu.time': 41093}
    strong_baryon['ref13'][2048] = {'particle.number': 7e7 * 2, 'cpu.number': 2048,
                                    'wall.time': 23.04, 'cpu.time': 47182}
    strong_baryon['ref13'][4096] = {'particle.number': 7e7 * 2, 'cpu.number': 4096,
                                    'wall.time': 21.22, 'cpu.time': 86945}

    # conversion from running to scale-factor = 0.068 to 0.1 via 2x
    for cpu_num in strong_baryon['ref14']:
        strong_baryon['ref14'][cpu_num]['cpu.time'] *= 2
        strong_baryon['ref14'][cpu_num]['wall.time'] *= 2

    # plot ----------
    _fig, subplot = ut.plot.make_figure(1, left=0.22, right=0.95, top=0.96, bottom=0.16)

    if scaling_kind == 'strong':
        strong = strong_baryon[resolution]
        cpu_numbers = [cpu_num for cpu_num in strong]
        wall_time_ref = strong[2048]['wall.time'] * 2048

        if time_kind == 'cpu':
            times = [strong[cpu_num]['cpu.time'] for cpu_num in strong]
        elif time_kind == 'wall':
            times = [strong[cpu_num]['wall.time'] for cpu_num in strong]
        elif time_kind == 'speedup':
            times = [wall_time_ref / strong[cpu_num]['wall.time'] for cpu_num in strong]
        elif time_kind == 'efficiency':
            times = [wall_time_ref / strong[cpu_num]['wall.time'] / cpu_num for cpu_num in strong]

        subplot.set_xlabel('core number')

        if resolution == 'ref14':
            axis_x_limits = [1e2, 1.9e4]
        elif resolution == 'ref13':
            axis_x_limits = [3e2, 1e4]

        if time_kind == 'cpu':
            if resolution == 'ref14':
                axis_y_limits = [0, 1.6e5]
                subplot.set_ylabel('CPU time to $z = 9$ [hr]')
            elif resolution == 'ref13':
                axis_y_limits = [0, 1e5]
                subplot.set_ylabel('CPU time to $z = 3$ [hr]')
        elif time_kind == 'wall':
            axis_y_limits = [0, 35]
            subplot.set_ylabel('wall time to $z = 9$ [hr]')
        elif time_kind == 'speedup':
            axis_y_limits = [0, 9000]
            subplot.set_ylabel('parallel speedup $T(1)/T(N)$')
        elif time_kind == 'efficiency':
            axis_y_limits = [0, 1.05]
            subplot.set_ylabel('parallel efficiency $T(1)/T(N)/N$')

        ut.plot.set_axes_scaling_limits(
            subplot, axis_x_scaling, axis_x_limits, None, axis_y_scaling, axis_y_limits)

        subplot.plot(cpu_numbers, times, '*-', linewidth=2.0, color='blue')

        if time_kind == 'speedup':
            subplot.plot([0, 3e4], [0, 3e4], '--', linewidth=1.5, color='black')

        if resolution == 'ref14':
            subplot.text(0.1, 0.1, 'strong scaling:\nparticle number = 1.1e9', color='black',
                         transform=subplot.transAxes)
        elif resolution == 'ref13':
            subplot.text(0.1, 0.1, 'strong scaling:\nparticle number = 1.5e8', color='black',
                         transform=subplot.transAxes)

    elif scaling_kind == 'weak':
        dm_particle_numbers = np.array(
            [weak_dark[cpu_num]['particle.number'] for cpu_num in sorted(weak_dark.keys())])
        mfm_particle_numbers = np.array([weak_baryon[cpu_num]['particle.number']
                                         for cpu_num in sorted(weak_baryon.keys())])

        if time_kind == 'cpu':
            dm_times = np.array([weak_dark[cpu_num]['cpu.time']
                                 for cpu_num in sorted(weak_dark.keys())])
            mfm_times = np.array([weak_baryon[cpu_num]['cpu.time']
                                  for cpu_num in sorted(weak_baryon.keys())])
        elif time_kind == 'wall':
            #resolutinon_ref = 'ref14'
            resolutinon_ref = 'ref13'
            ratio_ref = (weak_baryon[resolutinon_ref]['particle.number'] /
                         weak_baryon[resolutinon_ref]['cpu.number'])
            dm_times = np.array(
                [weak_dark[cpu_num]['wall.time'] * ratio_ref /
                 (weak_dark[cpu_num]['particle.number'] / weak_dark[cpu_num]['cpu.number'])
                 for cpu_num in sorted(weak_dark.keys())])
            mfm_times = np.array(
                [weak_baryon[cpu_num]['wall.time'] *
                 ratio_ref / (weak_baryon[cpu_num]['particle.number'] /
                              weak_baryon[cpu_num]['cpu.number'])
                 for cpu_num in sorted(weak_baryon.keys())])

        subplot.set_xlabel('particle number')

        #axis_x_limits = [6e6, 1.5e9]
        axis_x_limits = [1e6, 2e8]

        if time_kind == 'cpu':
            axis_y_limits = [5e2, 2e6]
            subplot.set_ylabel('CPU time to $z = 0$ [hr]')
        elif time_kind == 'wall':
            axis_y_limits = [10, 1000]
            subplot.set_ylabel('wall time to $z = 0$ [hr]')
            subplot.text(0.05, 0.05,
                         'weak scaling:\nfixed particles / core = {:.1e}'.format(ratio_ref),
                         color='black', transform=subplot.transAxes)

        ut.plot.set_axes_scaling_limits(
            subplot, axis_x_scaling, axis_x_limits, None, axis_y_scaling, axis_y_limits)

        subplot.plot(dm_particle_numbers, dm_times, '.-', linewidth=2.0, color='red')
        #subplot.plot(mfm_particle_numbers[:-1], mfm_times[:-1], '*-', linewidth=2.0, color='blue')
        #subplot.plot(mfm_particle_numbers[1:], mfm_times[1:], '*--', linewidth=2.0, color='blue',
        #             alpha=0.7)
        subplot.plot(mfm_particle_numbers, mfm_times, '*-', linewidth=2.0, color='blue')

    plot_name = 'test'
    ut.plot.parse_output(write_plot, plot_directory, plot_name)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise ValueError('specify function: runtime, properties, extrema, contamination, delete')

    function_kind = str(sys.argv[1])
    assert ('runtime' in function_kind or 'properties' in function_kind or
            'extrema' in function_kind or 'contamination' in function_kind or
            'delete' in function_kind)

    directory = '.'

    if 'runtime' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        wall_time_restart = 0
        if len(sys.argv) > 3:
            wall_time_restart = float(sys.argv[3])

        print_run_times(directory, wall_time_restart=wall_time_restart)

    elif 'properties' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        print_properties_statistics('all', directory)

    elif 'extrema' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        print_properties_snapshots(directory)

    elif 'contamination' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_redshift = 0
        if len(sys.argv) > 3:
            snapshot_redshift = float(sys.argv[3])

        plot_contamination(directory, snapshot_redshift)

    elif 'delete' in function_kind:
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_index_limits = None
        if len(sys.argv) > 4:
            snapshot_index_limits = [int(sys.argv[3]), int(sys.argv[4])]

        delete_snapshots(directory, snapshot_index_limits)

    else:
        print('! not recognize function')
