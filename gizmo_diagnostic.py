#!/usr/bin/env python

'''
Diagnose Gizmo simulations.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function
import collections
import os
import sys
import glob
import numpy as np
# local ----
import utilities as ut
from . import gizmo_io
from . import gizmo_analysis


class RuntimeClass(ut.io.SayClass):

    def get_cpu_numbers(self, simulation_directory='.', runtime_file_name='gizmo.out*'):
        '''
        Get number of MPI tasks and OpenMP threads from run-time file.
        If cannot find any, default to 1.

        Parameters
        ----------
        simulation_directory : str : top-level directory of simulation
        runtime_file_name : str : name of run-time file name (set in submission script)

        Returns
        -------
        mpi_number : int : number of MPI tasks
        omp_number : int : number of OpenMP threads per MPI task
        '''
        loop_number_max = 1000

        file_name = ut.io.get_path(simulation_directory) + runtime_file_name
        path_file_names = glob.glob(file_name)
        file_in = open(path_file_names[0], 'r')

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
            self.say('MPI tasks = {}'.format(mpi_number))
        else:
            self.say('! unable to find number of MPI tasks')
            mpi_number = 1

        if omp_number:
            self.say('OpenMP threads = {}'.format(omp_number))
        else:
            self.say('did not find any OpenMP threads')
            omp_number = 1

        return mpi_number, omp_number

    def print_run_times(
        self, simulation_directory='.', output_directory='output/', core_number=None,
        runtime_file_name='gizmo.out*', wall_time_restart=0, scalefactors=[]):
        '''
        Print wall [and CPU] times (based on average per MPI task from cpu.txt) at scale-factors,
        for Gizmo simulation.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        output_directory : str : directory of output files within simulation directory
        core_number : int : total number of CPU cores (input instead of reading from run-time file)
        runtime_file_name : str : name of run-time file to read CPU info
        wall_time_restart : float : wall time [sec] of previous run (if restarted from snapshot)
        scalefactors : array-like : list of scale-factors at which to print run times

        Returns
        -------
        scalefactors, redshifts, wall_times, cpu_times : arrays
        '''

        def get_scalefactor_string(scalefactor):
            if scalefactor == 1:
                scalefactor_string = '1'
            elif np.abs(scalefactor % 0.1) < 0.01:
                scalefactor_string = '{:.1f}'.format(scalefactor)
            elif np.abs(scalefactor % 0.01) < 0.001:
                scalefactor_string = '{:.2f}'.format(scalefactor)
            else:
                scalefactor_string = '{:.3f}'.format(scalefactor)
            return scalefactor_string

        file_name = 'cpu.txt'

        if scalefactors is None or (not np.isscalar(scalefactors) and not len(scalefactors)):
            scalefactors = [
                0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
        scalefactors = ut.array.arrayize(scalefactors)

        path_file_name = (
            ut.io.get_path(simulation_directory) + ut.io.get_path(output_directory) + file_name)
        file_in = open(path_file_name, 'r')

        wall_times = []

        i = 0
        scalefactor = 'Time: {}'.format(get_scalefactor_string(scalefactors[i]))
        print_next_line = False

        for line in file_in:
            if print_next_line:
                wall_times.append(float(line.split()[1]))
                print_next_line = False
                i += 1
                if i >= len(scalefactors):
                    break
                else:
                    scalefactor = 'Time: {}'.format(get_scalefactor_string(scalefactors[i]))
            elif scalefactor in line:
                print_next_line = True

        wall_times = np.array(wall_times)

        if wall_time_restart and len(wall_times) > 1:
            for i in range(1, len(wall_times)):
                if wall_times[i] < wall_times[i - 1]:
                    break
            wall_times[i:] += wall_time_restart

        wall_times /= 3600  # convert to [hr]

        if not core_number:
            # get core number from run-time file
            mpi_number, omp_number = self.get_cpu_numbers(simulation_directory, runtime_file_name)
            core_number = mpi_number * omp_number
            print('# core = {} (mpi = {}, omp = {})'.format(core_number, mpi_number, omp_number))
        else:
            print('# core = {}'.format(core_number))

        cpu_times = wall_times * core_number

        # sanity check - simulation might not have run to all input scale-factors
        scalefactors = scalefactors[: wall_times.size]
        redshifts = 1 / scalefactors - 1

        print('# scale-factor redshift wall-time[day] cpu-time[khr] run-time-percent')
        for t_i in range(len(wall_times)):
            print('{:.2f} {:5.2f} | {:6.2f}  {:7.1f}  {:3.0f}%'.format(
                  scalefactors[t_i], redshifts[t_i], wall_times[t_i] / 24,
                  cpu_times[t_i] / 1000, 100 * wall_times[t_i] / wall_times.max()))

        return scalefactors, redshifts, wall_times, cpu_times

    def print_run_times_ratios(
        self, simulation_directories=['.'], output_directory='output/',
        runtime_file_name='gizmo.out*', wall_times_restart=[],
        scalefactors=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9,
                      1.0]):
        '''
        Print ratios of wall times and CPU times (based on average per MPI taks from cpu.txt) at
        scale-factors, from different simulation directories, for Gizmo simulations.
        'reference' simulation is first in list.

        Parameters
        ----------
        simulation_directories : str or list : top-level directory[s] of simulation[s]
        output_directory : str : directory of output files within simulation directory
        runtime_file_name : str : name of run-time file  to read CPU info
        wall_times_restart : float or list :
            wall time[s] [sec] of previous run[s] (if restart from snapshot)
        scalefactors : array-like : list of scale-factors at which to print run times
        '''
        wall_timess = []
        cpu_timess = []

        if np.isscalar(simulation_directories):
            simulation_directories = [simulation_directories]

        if not wall_times_restart:
            wall_times_restart = np.zeros(len(simulation_directories))
        elif np.isscalar(wall_times_restart):
            wall_times_restart = [wall_times_restart]

        for d_i, directory in enumerate(simulation_directories):
            scalefactors, redshifts, wall_times, cpu_times = self.print_run_times(
                directory, output_directory, None, runtime_file_name, wall_times_restart[d_i],
                scalefactors)
            wall_timess.append(wall_times)
            cpu_timess.append(cpu_times)

        snapshot_number_min = np.Inf
        for d_i, wall_times in enumerate(wall_timess):
            if len(wall_times) < snapshot_number_min:
                snapshot_number_min = len(wall_times)

        # sanity check - simulations might not have run to each input scale-factor
        scalefactors = scalefactors[: snapshot_number_min]
        redshifts = redshifts[: snapshot_number_min]

        print('# scale-factor redshift', end='')
        for _ in range(1, len(wall_timess)):
            print(' wall-time-ratio cpu-time-ratio', end='')
        print()

        for a_i in range(snapshot_number_min):
            print('{:.2f} {:5.2f} |'.format(scalefactors[a_i], redshifts[a_i]), end='')
            for d_i in range(1, len(wall_timess)):
                print(' {:5.1f}'.format(wall_timess[d_i][a_i] / wall_timess[0][a_i]), end='')
                print(' {:5.1f}'.format(cpu_timess[d_i][a_i] / cpu_timess[0][a_i]), end='')
            print()


Runtime = RuntimeClass()


class ContaminationClass(ut.io.SayClass):
    '''
    Contamination by low-resolution dark matter.
    '''

    def plot_contamination_v_distance(
        self, part,
        distance_limits=[10, 2000], distance_bin_width=0.01, distance_scaling='log',
        halo_radius=None, scale_to_halo_radius=False, center_position=None,
        axis_y_limits=[0.0001, 1], axis_y_scaling='log',
        write_plot=False, plot_directory='.', figure_index=1):
        '''
        Plot contamination from low-resolution particles v distance from center.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        distance_limits : list : min and max limits for distance from galaxy
        distance_bin_width : float : width of each distance bin (in units of distance_scaling)
        distance_scaling : str : 'log', 'linear'
        halo_radius : float : radius of halo [kpc physical]
        scale_to_halo_radius : bool : whether to scale distance to halo_radius
        center_position : array : position of galaxy/halo center
        axis_y_limits : list : min and max limits for y-axis
        axis_y_scaling : str : scaling of y-axis: 'log', 'linear'
        write_plot : bool : whether to write figure to file
        plot_directory : str : directory to write figure file
        figure_index : int : index of figure for matplotlib
        '''
        virial_kind = '200m'

        center_position = ut.particle.parse_property(part, 'center_position', center_position)

        if scale_to_halo_radius:
            assert halo_radius and halo_radius > 0

        DistanceBin = ut.binning.DistanceBinClass(
            distance_scaling, distance_limits, distance_bin_width)

        profile_mass = collections.OrderedDict()
        profile_mass['total'] = {}
        for spec in part:
            profile_mass[spec] = {}

        profile_mass_ratio = {}
        profile_number = {}

        for spec in part:
            distances = ut.coordinate.get_distances(
                part[spec]['position'], center_position, part.info['box.length'],
                part.snapshot['scalefactor'], total_distance=True)  # [kpc physical]
            if scale_to_halo_radius:
                distances /= halo_radius
            profile_mass[spec] = DistanceBin.get_sum_profile(distances, part[spec]['mass'])

        # initialize total mass
        for prop in profile_mass[spec]:
            if 'distance' not in prop:
                profile_mass['total'][prop] = 0
            else:
                profile_mass['total'][prop] = profile_mass[spec][prop]

        # compute mass fractions relative to total mass
        for spec in part:
            for prop in profile_mass[spec]:
                if 'distance' not in prop:
                    profile_mass['total'][prop] += profile_mass[spec][prop]

        for spec in part:
            profile_mass_ratio[spec] = {
                'sum': profile_mass[spec]['sum'] / profile_mass['total']['sum'],
                'sum.cum': profile_mass[spec]['sum.cum'] / profile_mass['total']['sum.cum'],
            }
            profile_number[spec] = {
                'sum': np.int64(np.round(profile_mass[spec]['sum'] / part[spec]['mass'][0])),
                'sum.cum': np.int64(np.round(
                    profile_mass[spec]['sum.cum'] / part[spec]['mass'][0])),
            }

        # print diagnostics
        if scale_to_halo_radius:
            distances_halo = profile_mass['dark2']['distance.cum']
            distances_phys = distances_halo * halo_radius
        else:
            distances_phys = profile_mass['dark2']['distance.cum']
            if halo_radius and halo_radius > 0:
                distances_halo = distances_phys / halo_radius
            else:
                distances_halo = distances_phys

        species_lowres_dark = []
        for i in range(2, 10):
            dark_name = 'dark{}'.format(i)
            if dark_name in part:
                species_lowres_dark.append(dark_name)

        for spec in species_lowres_dark:
            self.say('* {}'.format(spec))
            if profile_mass[spec]['sum.cum'][-1] == 0:
                self.say('  none. yay!')
                continue

            if scale_to_halo_radius:
                print_string = 'd/R_halo < {:5.2f}, d < {:6.2f} kpc: '
            else:
                print_string = 'd < {:6.1f} kpc, d/R_halo < {:5.2f}: '
            print_string += 'mass_frac = {:.4f}, mass = {:.2e}, number = {:.0f}'

            for dist_i in range(profile_mass[spec]['sum.cum'].size):
                if profile_mass[spec]['sum.cum'][dist_i] > 0:
                    if scale_to_halo_radius:
                        distances_0 = distances_halo[dist_i]
                        distances_1 = distances_phys[dist_i]
                    else:
                        distances_0 = distances_phys[dist_i]
                        if halo_radius and halo_radius > 0:
                            distances_1 = distances_halo[dist_i]
                        else:
                            distances_1 = np.nan

                    self.say(print_string.format(
                        distances_0, distances_1,
                        profile_mass_ratio[spec]['sum.cum'][dist_i],
                        profile_mass[spec]['sum.cum'][dist_i],
                        profile_number[spec]['sum.cum'][dist_i])
                    )

                    if spec != 'dark2':
                        # print only 1 distance bin for lower-resolution particles
                        break

        print()
        print('contamination summary')
        species = 'dark2'
        if halo_radius and halo_radius > 0:
            dist_i_halo = np.searchsorted(distances_phys, halo_radius)
        else:
            dist_i_halo = 0
        if profile_number[species]['sum.cum'][dist_i_halo] > 0:
            print('* {} {} particles within R_halo'.format(
                  profile_number[species]['sum.cum'][dist_i_halo], species))
        dist_i = np.where(profile_number[species]['sum.cum'] > 0)[0][0]
        print('* {} closest d = {:.1f} kpc, {:.1f} R_halo'.format(
              species, distances_phys[dist_i], distances_halo[dist_i]))
        dist_i = np.where(profile_mass_ratio[species]['sum.cum'] > 0.0001)[0][0]
        print('* {} mass_ratio = 0.01% at d < {:.1f} kpc, {:.1f} R_halo'.format(
              species, distances_phys[dist_i], distances_halo[dist_i]))
        dist_i = np.where(profile_mass_ratio[species]['sum.cum'] > 0.001)[0][0]
        print('* {} mass_ratio = 0.1% at d < {:.1f} kpc, {:.1f} R_halo'.format(
              species, distances_phys[dist_i], distances_halo[dist_i]))
        dist_i = np.where(profile_mass_ratio[species]['sum.cum'] > 0.01)[0][0]
        print('* {} mass_ratio = 1% at d < {:.1f} kpc, {:.1f} R_halo'.format(
              species, distances_phys[dist_i], distances_halo[dist_i]))

        for spec in species_lowres_dark:
            if species != 'dark2' and profile_number[spec]['sum.cum'][dist_i_halo] > 0:
                print('! {} {} particles within R_halo'.format(
                      profile_number[species]['sum.cum'][dist_i_halo], species))
                dist_i = np.where(profile_number[spec]['sum.cum'] > 0)[0][0]
                print('! {} closest d = {:.1f} kpc, {:.1f} R_halo'.format(
                      species, distances_phys[dist_i], distances_halo[dist_i]))
        print()

        if write_plot is None:
            return

        # plot ----------
        _fig, subplot = ut.plot.make_figure(figure_index)

        ut.plot.set_axes_scaling_limits(
            subplot, distance_scaling, distance_limits, None, axis_y_scaling, axis_y_limits)

        subplot.set_ylabel(
            '$M_{{\\rm species}} / M_{{\\rm total}}$')
        if scale_to_halo_radius:
            axis_x_label = '$d \, / \, R_{{\\rm {}}}$'.format(virial_kind)
        else:
            axis_x_label = 'distance $[\\rm kpc]$'
        subplot.set_xlabel(axis_x_label)

        colors = ut.plot.get_colors(len(species_lowres_dark), use_black=False)

        if halo_radius:
            if scale_to_halo_radius:
                x_ref = 1
            else:
                x_ref = halo_radius
            subplot.plot([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

        for spec_i, spec in enumerate(species_lowres_dark):
            subplot.plot(
                DistanceBin.mids, profile_mass_ratio[spec]['sum'], color=colors[spec_i], alpha=0.7,
                label=spec)

        ut.plot.make_legends(subplot, 'best')

        distance_name = 'dist'
        if halo_radius and scale_to_halo_radius:
            distance_name += '.' + virial_kind
        plot_name = ut.plot.get_file_name(
            'mass.ratio', distance_name, snapshot_dict=part.snapshot)
        ut.plot.parse_output(write_plot, plot_name, plot_directory)

    def plot_contamination_v_distance_halo(
        self, part, hal, hal_index, distance_max=7, distance_bin_width=0.5,
        scale_to_halo_radius=True):
        '''
        Print information on contamination from lower-resolution particles around halo as a function
        of distance.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        hal : dict : catalog of halos at snapshot
        hal_index: int : index of halo
        distance_max : float : maximum distance from halo center to check
        distance_bin_width : float : width of distance bin for printing
        scale_to_halo_radius : bool : whether to scale distances by virial radius
        '''
        distance_scaling = 'linear'
        distance_limits = [0, distance_max]
        axis_y_scaling = 'log'

        self.say('halo radius = {:.1f} kpc'.format(hal['radius'][hal_index]))

        halo_radius = hal['radius'][hal_index]

        self.plot_lowres_contamination_v_distance(
            part, distance_limits, distance_bin_width, None, distance_scaling, halo_radius,
            scale_to_halo_radius, hal['position'][hal_index], axis_y_scaling, write_plot=None)

    def plot_contamination_v_distance_both(self, redshift=0, simulation_directory='.'):
        '''
        Plot contamination from lower-resolution particles around halo center as a function of
        distance.

        Parameters
        ----------
        redshift : float : redshift of snapshot
        simulation_directory : str : top-level directory of simulation
        '''
        distance_bin_width = 0.01
        distance_limits_phys = [10, 2000]  # [kpc physical]
        distance_limits_halo = [0.01, 7]  # [units of R_halo]
        virial_kind = '200m'

        os.chdir(simulation_directory)

        Read = gizmo_io.ReadClass()
        part = Read.read_snapshots(
            ['dark', 'dark2'], 'redshift', redshift, simulation_directory,
            properties=['position', 'mass', 'potential'], assign_host_coordinates=True)

        halo_prop = ut.particle.get_halo_properties(part, 'all', virial_kind)

        self.plot_contamination_v_distance(
            part, distance_limits_phys, distance_bin_width, halo_radius=halo_prop['radius'],
            scale_to_halo_radius=False, write_plot=True, plot_directory='plot')

        self.plot_contamination_v_distance(
            part, distance_limits_halo, distance_bin_width, halo_radius=halo_prop['radius'],
            scale_to_halo_radius=True, write_plot=True, plot_directory='plot')


Contamination = ContaminationClass()


def print_properties_statistics(
    species='all', snapshot_value_kind='index', snapshot_value=600,
    simulation_directory='.', snapshot_directory='output/'):
    '''
    For each property of each species in particle catalog, print range and median.

    Parameters
    ----------
    species : str or list : name[s] of particle species to print
    snapshot_value_kind : str : input snapshot number kind: index, redshift
    snapshot_value : int or float : index (number) of snapshot file
    simulation_directory : root directory of simulation
    snapshot_directory: str : directory of snapshot files within simulation_directory

    Returns
    -------
    part : dict : catalog of particles
    '''
    species = ut.array.arrayize(species)
    if 'all' in species:
        species = ['dark2', 'dark', 'star', 'gas']

    Read = gizmo_io.ReadClass()
    part = Read.read_snapshots(
        species, snapshot_value_kind, snapshot_value, simulation_directory,
        snapshot_directory, '', None, None, assign_host_coordinates=False,
        separate_dark_lowres=False, sort_dark_by_id=False)

    gizmo_analysis.print_properties_statistics(part, species)


def print_properties_snapshots(
    simulation_directory='.', snapshot_directory='output',
    species_property_dict={'gas': ['smooth.length', 'number.density']}):
    '''
    For each input property, get its extremum at each snapshot.
    Print statistics of property across all snapshots.

    Parameters
    ----------
    simulation_directory : str : directory of simulation
    snapshot_directory : str : directory of snapshot files
    species_property_dict : dict : keys = species, values are string or list of property[s]
    '''
    element_indices = [0, 1]

    property_statistic = {
        'smooth.length': {'function.name': 'min', 'function': np.min},
        'density': {'function.name': 'max', 'function': np.max},
        'number.density': {'function.name': 'max', 'function': np.max},
    }

    Say = ut.io.SayClass(print_properties_snapshots)

    simulation_directory = ut.io.get_path(simulation_directory)

    Snapshot = ut.simulation.SnapshotClass()
    Snapshot.read_snapshots(directory=simulation_directory)

    species_read = species_property_dict.keys()

    properties_read = []
    for spec in species_property_dict:
        properties = species_property_dict[spec]
        if np.isscalar(properties):
            properties = [properties]

        prop_dict = {}
        for prop in species_property_dict[spec]:
            prop_dict[prop] = []

            prop_read = prop.replace('.number', '')
            if prop_read not in properties_read:
                properties_read.append(prop_read)

            if '.number' in prop and 'massfraction' not in properties_read:
                properties_read.append('massfraction')

        # re-assign property list as dictionary so can store list of values
        species_property_dict[spec] = prop_dict

    for snapshot_i in Snapshot['index']:
        try:
            Read = gizmo_io.ReadClass()
            part = Read.read_snapshots(
                species_read, 'index', snapshot_i, simulation_directory, snapshot_directory, '',
                properties_read, element_indices, assign_host_coordinates=False,
                sort_dark_by_id=False)

            for spec in species_property_dict:
                for prop in species_property_dict[spec]:
                    try:
                        prop_ext = property_statistic[prop]['function'](part[spec].prop(prop))
                        species_property_dict[spec][prop].append(prop_ext)
                    except Exception:
                        Say.say('! {} {} not in particle dictionary'.format(spec, prop))
        except Exception:
            Say.say('! cannot read snapshot index {} in {}'.format(
                    snapshot_i, simulation_directory + snapshot_directory))

    Statistic = ut.statistic.StatisticClass()

    for spec in species_property_dict:
        for prop in species_property_dict[prop]:
            prop_func_name = property_statistic[prop]['function.name']
            prop_values = np.array(species_property_dict[spec][prop])

            Statistic.stat = Statistic.get_statistic_dict(prop_values)

            Say.say('\n{} {} {}:'.format(spec, prop, prop_func_name))
            for stat_name in ['min', 'percent.16', 'median', 'percent.84', 'max']:
                Say.say('{:10s} = {:.3f}'.format(stat_name, Statistic.stat[stat_name]))

            #Statistic.print_statistics()


def test_stellar_mass_loss(
    part_z0, part_z, metallicity_limits=[0.001, 10], metallicity_bin_width=0.2,
    form_time_width=5):
    '''
    .
    '''
    from . import gizmo_track
    from . import gizmo_star

    Say = ut.io.SayClass(test_stellar_mass_loss)

    species = 'star'

    if 'index_pointers' not in part_z.__dict__:
        gizmo_track.ParticleIndexPointer.io_pointers(part_z)

    MetalBin = ut.binning.BinClass(
        metallicity_limits, metallicity_bin_width, include_max=True, scaling='log')

    #MassLoss = gizmo_star.MassLossClass()
    #MassLoss._make_mass_loss_fraction_spline(age_bin_width=0.2, metallicity_bin_width=0.1)

    form_time_limits = [part_z.snapshot['time'] * 1000 - form_time_width,
                        part_z.snapshot['time'] * 1000]

    part_indices_z0 = ut.array.get_indices(
        part_z0[species].prop('form.time') * 1000, form_time_limits)
    part_indices_z = part_z.index_pointers[part_indices_z0]

    Say.say('* stellar mass loss across {:.3f} Gyr in metallicity bins for {} particles'.format(
            part_z0.snapshot['time'] - part_z.snapshot['time'], part_indices_z0.size))

    # compute metallicity using solar abundance assumed in Gizmo
    metallicities = (part_z0[species].prop('massfraction.metals', part_indices_z0) /
                     gizmo_star.StellarWind.solar_metal_mass_fraction)

    metal_bin_indices = MetalBin.get_bin_indices(metallicities)

    for metal_i, metallicity in enumerate(MetalBin.mids):
        masks = (metal_bin_indices == metal_i)
        if np.sum(masks):
            pis_z0 = part_indices_z0[masks]
            pis_z = part_indices_z[masks]

            mass_loss_fractions = (
                (part_z[species]['mass'][pis_z] - part_z0[species]['mass'][pis_z0]) /
                part_z[species]['mass'][pis_z])

            mass_loss_fractions_py = part_z0[species].prop('mass.loss.fraction', pis_z0)
            #mass_loss_fractions_py = MassLoss.get_mass_loss_fraction_from_spline(
            #    part_z0[species].prop('age', pis_z0) * 1000,
            #    metal_mass_fractions=part_z0[species].prop('massfraction.metals', pis_z0))

            Say.say('Z = {:.3f}, N = {:4d} | gizmo {:.1f}%, python {:.1f}%, p/g = {:.3f}'.format(
                metallicity, pis_z0.size,
                100 * np.median(mass_loss_fractions), 100 * np.median(mass_loss_fractions_py),
                np.median(mass_loss_fractions_py / mass_loss_fractions)))

    mass_loss_fractions = (
        (part_z[species]['mass'][part_indices_z] - part_z0[species]['mass'][part_indices_z0]) /
        part_z[species]['mass'][part_indices_z])
    mass_loss_fractions_py = part_z0[species].prop('mass.loss.fraction', part_indices_z0)
    print('* all Z, N = {} | gizmo = {:.1f}%, python = {:.1f}%, p/g = {:.3f}'.format(
          part_indices_z0.size, 100 * np.median(mass_loss_fractions),
          100 * np.median(mass_loss_fractions_py),
          np.median(mass_loss_fractions_py / mass_loss_fractions)))


#===================================================================================================
# performance and scaling
#===================================================================================================
def plot_scaling(
    scaling_kind='strong', resolution='res7100', time_kind='core',
    axis_x_scaling='log', axis_y_scaling='linear', write_plot=False, plot_directory='.'):
    '''
    Print simulation run times (wall or core).
    'speedup' := WT(1 CPU) / WT(N CPU) =
    'efficiency' := WT(1 CPU) / WT(N CPU) / N = CT(1 CPU) / CT(N CPU)

    Parameters
    ----------
    scaling_kind : str : 'strong', 'weak'
    time_kind : str : 'node', 'core', 'wall', 'speedup', 'efficiency'
    axis_x_scaling : str : scaling along x-axis: 'log', 'linear'
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    write_plot : bool : whether to write plot to file
    plot_directory : str : directory to write plot file
    '''
    _weak_dark = {
        'res57000': {'particle.number': 8.82e6, 'core.number': 64,
                     'core.time': 385, 'wall.time': 6.0},
        'res7100': {'particle.number': 7.05e7, 'core.number': 512,
                    'core.time': 7135, 'wall.time': 13.9},
        'res880': {'particle.number': 5.64e8, 'core.number': 2048,
                   'core.time': 154355, 'wall.time': 75.4},
    }

    # stampede
    """
    weak_baryon = {
        'res450000': {'particle.number': 1.10e6 * 2, 'core.number': 32,
                      'core.time': 1003, 'wall.time': 31.34 * 1.5},
        'res57000': {'particle.number': 8.82e6 * 2, 'core.number': 512,
                     'core.time': 33143, 'wall.time': 64.73},
        'res7100': {'particle.number': 7.05e7 * 2, 'core.number': 2048,
                    'core.time': 1092193, 'wall.time': 350.88},
        #'res880': {'particle.number': 5.64e8 * 2, 'core.number': 8192,
        #           'core.time': 568228, 'wall.time': 69.4},
        # projected
        #'res880': {'particle.number': 5.64e8 * 2, 'core.number': 8192,
        #           'core.time': 1.95e7, 'wall.time': 2380},
    }
    """

    # conversion to stampede 2
    weak_baryon = collections.OrderedDict()
    weak_baryon['res450000'] = {
        'particle.number': 1.10e6 * 2, 'node.number': 1, 'node.time': 73, 'wall.time': 73}
    weak_baryon['res57000'] = {
        'particle.number': 8.82e6 * 2, 'node.number': 8, 'node.time': 1904, 'wall.time': 239}
    weak_baryon['res7100'] = {
        'particle.number': 7.05e7 * 2, 'node.number': 64, 'node.time': 52000, 'wall.time': 821}

    strong_baryon = collections.OrderedDict()

    # convert from running to scale-factor = 0.068 to 0.1 via 2x
    strong_baryon['res880'] = {
        'particle.number': 5.64e8 * 2,
        'core.number': np.array([2048, 4096, 8192, 16384]),
        'node.number': np.array([128, 256, 512, 1024]),
        'wall.time': np.array([15.55, 8.64, 4.96, 4.57]) * 2,
        'core.time': np.array([31850, 35389, 40632, 74875]) * 2,
    }

    # did not have time to run these, so scale down from res880
    # scaled to run time to z = 3 using 2048
    # stampede
    """
    strong_baryon['res7100'] = {
        'particle.number': 7e7 * 2,
        'node.number': np.array([32, 64, 128, 256]),
        'core.number': np.array([512, 1024, 2048, 4096]),
        'wall.time': np.array([72.23, 40.13, 23.04, 21.22]),
        'core.time': np.array([36984, 41093, 47182, 86945]),
        'node.time': np.array([2312, 2568, 2949, 5434]),
    }
    """

    # conversion to stampede 2
    # half the number of nodes and multipy node time by 1.17, multiply wall time by 2.34
    # based on res57000 simulation to z = 0
    strong_baryon['res7100'] = {
        'particle.number': 7e7 * 2,
        'node.number': np.array([16, 32, 64, 128]),
        'core.number': np.array([2048, 4096, 8192, 16384]),
        'wall.time': np.array([72.23, 40.13, 23.04, 21.22]) * 2.34,
        'core.time': np.array([36984, 41093, 47182, 86945]) * 1.17,
        'node.time': np.array([2312, 2568, 2949, 5434]) * 1.17,
    }

    # plot ----------
    _fig, subplot = ut.plot.make_figure(1, left=0.22, right=0.95, top=0.96, bottom=0.16)

    if scaling_kind == 'strong':
        strong = strong_baryon[resolution]

        if time_kind == 'core':
            times = strong['core.time']
        if time_kind == 'node':
            times = strong['node.time']
        elif time_kind == 'wall':
            times = strong['wall.time']
        elif time_kind == 'speedup':
            times = strong['wall.time'][0] / strong['wall.time']
        elif time_kind == 'efficiency':
            times = strong['wall.time'][0] / strong['wall.time']

        #subplot.set_xlabel('number of cores')
        subplot.set_xlabel('number of nodes')

        if resolution == 'res880':
            axis_x_limits = [1e2, 1.9e4]
        elif resolution == 'res7100':
            #axis_x_limits = [3e2, 1e4]
            axis_x_limits = [10, 200]

        axis_x_kind = 'core.number'
        if time_kind == 'core':
            if resolution == 'res880':
                axis_y_limits = [0, 1.6e5]
                subplot.set_ylabel('CPU time to $z = 9$ [hr]')
            elif resolution == 'res7100':
                axis_y_limits = [0, 1e5]
                subplot.set_ylabel('CPU time to $z = 3$ [hr]')
        elif time_kind == 'node':
            axis_x_kind = 'node.number'
            if resolution == 'res880':
                axis_y_limits = [0, 1e4]
                subplot.set_ylabel('node time to $z = 9$ [hr]')
            elif resolution == 'res7100':
                axis_y_limits = [0, 8000]
                subplot.set_ylabel('node-hours to $z = 3$')
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

        subplot.plot(strong[axis_x_kind], times, '*-', linewidth=2.0, color='blue')

        if time_kind == 'speedup':
            subplot.plot([0, 3e4], [0, 3e4], '--', linewidth=1.5, color='black')

        if resolution == 'res880':
            subplot.text(0.1, 0.1, 'strong scaling:\nparticle number = 1.1e9', color='black',
                         transform=subplot.transAxes)
        elif resolution == 'res7100':
            subplot.text(0.1, 0.1, 'strong scaling:\nparticle number = 1.5e8', color='black',
                         transform=subplot.transAxes)

    elif scaling_kind == 'weak':
        #dm_particle_numbers = np.array(
        #    [weak_dark[core_num]['particle.number'] for core_num in sorted(weak_dark.keys())])
        baryon_particle_numbers = np.array(
            [weak_baryon[i]['particle.number'] for i in weak_baryon])

        if time_kind == 'node':
            #dm_times = np.array(
            #    [weak_dark[core_num]['core.time'] for core_num in sorted(weak_dark.keys())])
            baryon_times = np.array([weak_baryon[i]['node.time'] for i in weak_baryon])
        elif time_kind == 'wall':
            #resolutinon_ref = 'res880'
            resolutinon_ref = 'res7100'
            ratio_ref = (weak_baryon[resolutinon_ref]['particle.number'] /
                         weak_baryon[resolutinon_ref]['node.number'])
            #dm_times = np.array(
            #    [weak_dark[core_num]['wall.time'] * ratio_ref /
            #     (weak_dark[core_num]['particle.number'] / weak_dark[core_num]['core.number'])
            #     for core_num in sorted(weak_dark.keys())])
            baryon_times = np.array([weak_baryon[i]['wall.time'] for i in weak_baryon])

        subplot.set_xlabel('number of particles')

        #axis_x_limits = [6e6, 1.5e9]
        axis_x_limits = [1e6, 2e8]

        if time_kind == 'node':
            axis_y_limits = [10, 2e5]
            subplot.set_ylabel('node-hours to $z = 0$')
        elif time_kind == 'wall':
            axis_y_limits = [10, 1000]
            subplot.set_ylabel('wall time to $z = 0$ [hr]')
            subplot.text(
                0.05, 0.05,
                'weak scaling:\nparticles / node = {:.1e}'.format(ratio_ref),
                color='black', transform=subplot.transAxes,
            )

        ut.plot.set_axes_scaling_limits(
            subplot, axis_x_scaling, axis_x_limits, None, axis_y_scaling, axis_y_limits)

        #subplot.plot(dm_particle_numbers, dm_times, '.-', linewidth=2.0, color='red')
        #subplot.plot(mfm_particlgizmoe_numbers[:-1], mfm_times[:-1], '*-', linewidth=2.0,
        #color='blue')
        #subplot.plot(mfm_particle_numbers[1:], mfm_times[1:], '*--', linewidth=2.0, color='blue',
        #             alpha=0.7)
        subplot.plot(baryon_particle_numbers, baryon_times, '*-', linewidth=2.0, color='blue')

    plot_name = 'scaling'
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise OSError('specify function: runtime, properties, extrema, contamination, delete')

    function_kind = str(sys.argv[1])
    assert ('runtime' in function_kind or 'properties' in function_kind or
            'extrema' in function_kind or 'contamination' in function_kind)

    directory = '.'

    if 'runtime' in function_kind:
        wall_time_restart = 0
        if len(sys.argv) > 2:
            wall_time_restart = float(sys.argv[2])

        scalefactors = None  # use default
        if len(sys.argv) > 3:
            scalefactor_min = float(sys.argv[3])
            scalefactor_width = 0.05
            if len(sys.argv) > 4:
                scalefactor_width = float(sys.argv[4])
            scalefactors = np.arange(scalefactor_min, 1.01, scalefactor_width)

        _ = Runtime.print_run_times(wall_time_restart=wall_time_restart, scalefactors=scalefactors)

    elif 'properties' in function_kind:
        print_properties_statistics('all')

    elif 'extrema' in function_kind:
        print_properties_snapshots()

    elif 'contamination' in function_kind:
        snapshot_redshift = 0
        if len(sys.argv) > 2:
            snapshot_redshift = float(sys.argv[2])

        Contamination.plot_contamination_v_distance_both(snapshot_redshift)

    else:
        print('! not recognize function')
