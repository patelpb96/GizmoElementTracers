#!/usr/bin/env python3

'''
Edit Gizmo snapshot files: compress, delete, transfer across machines.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

import os
import sys
import glob
import numpy as np

import utilities as ut

# default subset of snapshots (65 snapshots)
snapshot_indices_keep = [
    0,  # z = 99
    20,
    26,
    33,
    41,
    52,  # z = 10 - 6
    55,
    57,
    60,
    64,
    67,  # z = 5.8 - 5.0
    71,
    75,
    79,
    83,
    88,  # z = 4.8 - 4.0
    91,
    93,
    96,
    99,
    102,
    105,
    109,
    112,
    116,
    120,  # z = 3.9 - 3.0
    124,
    128,
    133,
    137,
    142,
    148,
    153,
    159,
    165,
    172,  # z = 2.9 - 2.0
    179,
    187,
    195,
    204,
    214,
    225,
    236,
    248,
    262,
    277,  # z = 1.9 - 1.0
    294,
    312,
    332,
    356,
    382,
    412,
    446,
    486,
    534,  # z = 0.9 - 0.1
    539,
    544,
    550,
    555,
    561,
    567,
    573,
    579,
    585,  # z = 0.09 - 0.01
    600,
]


# --------------------------------------------------------------------------------------------------
# compress files
# --------------------------------------------------------------------------------------------------
class CompressClass(ut.io.SayClass):
    '''
    Compress snapshot files, losslessly.
    '''

    def compress_snapshots(
        self,
        simulation_directory='.',
        snapshot_directory='output',
        snapshot_directory_out='',
        snapshot_index_limits=[0, 600],
        analysis_directory='~/analysis',
        python_executable='python3',
        thread_number=1,
    ):
        '''
        Compress all snapshots in input directory.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        snapshot_directory : str : directory of snapshots
        snapshot_directory_out : str : directory to write compressed snapshots
        snapshot_index_limits : list : min and max snapshot indices to compress
        analysis_directory : str : directory of analysis code
        python_executable : str : python executable to use to run compression script
        thread_number : int : number of parallel threads
        '''
        snapshot_indices = np.arange(snapshot_index_limits[0], snapshot_index_limits[1] + 1)

        args_list = [
            (
                simulation_directory,
                snapshot_directory,
                snapshot_directory_out,
                analysis_directory,
                python_executable,
                snapshot_index,
            )
            for snapshot_index in snapshot_indices
        ]

        ut.io.run_in_parallel(self.compress_snapshot, args_list, thread_number=thread_number)

    def compress_snapshot(
        self,
        simulation_directory='.',
        snapshot_directory='output',
        snapshot_directory_out='',
        analysis_directory='~/analysis',
        python_executable='python3',
        snapshot_index=600,
    ):
        '''
        Compress single snapshot (which may be multiple files) in input directory.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        snapshot_directory : str : directory of snapshot
        snapshot_directory_out : str : directory to write compressed snapshot
        analysis_directory : str : directory of analysis code
        python_executable : str : python executable to use to run compression script
        snapshot_index : int : index of snapshot
        '''
        executable = (
            f'{python_executable} {analysis_directory}/manipulate_hdf5/compactify_hdf5.py -L 0'
        )

        snapshot_name_base = 'snap*_{:03d}*'

        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)
        if snapshot_directory_out and snapshot_directory_out[-1] != '/':
            snapshot_directory_out += '/'

        path_file_names = glob.glob(
            simulation_directory + snapshot_directory + snapshot_name_base.format(snapshot_index)
        )

        if len(path_file_names) > 0:
            if 'snapdir' in path_file_names[0]:
                path_file_names = glob.glob(path_file_names[0] + '/*')

            path_file_names.sort()

            for path_file_name in path_file_names:
                if snapshot_directory_out:
                    path_file_name_out = path_file_name.replace(
                        snapshot_directory, snapshot_directory_out
                    )
                else:
                    path_file_name_out = path_file_name

                executable_i = f'{executable} -o {path_file_name_out} {path_file_name}'
                self.say(f'executing:  {executable_i}')
                os.system(executable_i)

    def test_compression(
        self,
        simulation_directory='.',
        snapshot_directory='output',
        snapshot_indices='all',
        compression_level=0,
        verbose=False,
    ):
        '''
        Read headers from all snapshot files in simulation_directory to check whether files have
        been compressed.
        '''
        from . import gizmo_io

        Read = gizmo_io.ReadClass()

        header_compression_name = 'compression.level'

        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        compression_wrong_snapshots = []
        compression_none_snapshots = []

        snapshot_block_number = 1

        # get all snapshot file names and indices in directory
        path_file_names, file_snapshot_indices = Read.get_snapshot_file_names_indices(
            simulation_directory + snapshot_directory
        )

        if 'snapdir' in path_file_names[0]:
            # get number of block files per snapshot
            snapshot_file_names = glob.glob(path_file_names[0] + '/*')
            snapshot_block_number = len(snapshot_file_names)

        if snapshot_indices is not None and snapshot_indices != 'all':
            # input snapshot indices, so limit to those
            if np.isscalar(snapshot_indices):
                snapshot_indices = [snapshot_indices]
            snapshot_indices = np.intersect1d(snapshot_indices, file_snapshot_indices)
        else:
            snapshot_indices = file_snapshot_indices

        for snapshot_index in snapshot_indices:
            for snapshot_block_index in range(snapshot_block_number):
                header = Read.read_header(
                    'index',
                    snapshot_index,
                    simulation_directory,
                    snapshot_block_index=snapshot_block_index,
                    verbose=verbose,
                )
                if header_compression_name in header:
                    if (
                        compression_level is not None
                        and header[header_compression_name] != compression_level
                        and snapshot_index not in compression_wrong_snapshots
                    ):
                        compression_wrong_snapshots.append(snapshot_index)
                elif snapshot_index not in compression_none_snapshots:
                    compression_none_snapshots.append(snapshot_index)

        self.say(
            '* tested {} snapshots [{}, {}]'.format(
                len(snapshot_indices), min(snapshot_indices), max(snapshot_indices)
            )
        )
        self.say('* {} are uncompressed'.format(len(compression_none_snapshots)))
        if len(compression_none_snapshots) > 0:
            self.say(f'{compression_none_snapshots}')
        self.say(
            '* {} have wrong compression (level != {})'.format(
                len(compression_wrong_snapshots), compression_level
            )
        )
        if len(compression_wrong_snapshots) > 0:
            self.say(f'{compression_wrong_snapshots}')


Compress = CompressClass()


# --------------------------------------------------------------------------------------------------
# clean and archive simulation directories and files
# --------------------------------------------------------------------------------------------------
def clean_directories(
    directories='.',
    gizmo_directory='gizmo',
    snapshot_directory='output',
    restart_directory='restartfiles',
    gizmo_out_file='gizmo.out',
    gizmo_err_file='gizmo.err',
    snapshot_scalefactor_file='snapshot_scale-factors.txt',
):
    '''
    Clean a simulation directory, a list of simulation directories, or a directory of multiple
    simulation directories.
    Run this after a simulation finishes.
    Remove unnecessary run-time files, and tar directories (into a single tar-ball file) that we
    generally do not need for post-processing analysis.

    Parameters
    ----------
    directories : str or list thereof : directory[s] to run this on
        can be a single simulation directory, a list of simulation directories,
        or a directory that contains multiple simulation directories for which this function will
        run recursively on each one
    gizmo_directory : str : directory of Gizmo source code
    snapshot_directory : str : output directory that contains snapshots
    restart_directory : str : directory within snapshot_directory that contains restart files
    gizmo_out_file : str : Gizmo 'out' file
    gizmo_err_file : str : Gizmo error file
    snapshot_scalefactor_file : str : file that contains snapshot scale-factors (only)
    '''
    gizmo_config_file_used = 'GIZMO_config.h'
    gizmo_config_file_save = 'gizmo_config.h'  # file to save used config settings and gizmo version
    gizmo_job_directory = 'gizmo_jobs'

    if np.isscalar(directories):
        directories = [directories]

    gizmo_directory = gizmo_directory.rstrip('/')
    snapshot_directory = snapshot_directory.rstrip('/')

    cwd = os.getcwd()  # save current directory

    # move into each directory
    for directory in directories:
        directory = directory.rstrip('/')
        if directory != '.':
            print(f'\n* moving into:  {directory}/')
            os.chdir(f'{directory}')

        # check if this directory has relevant simulation directories,
        # or if need to run recursively on different simulation directories
        directory_names = glob.glob('*/')  # get names of all directories
        directory_names.sort()
        if len(directory_names) == 0:
            # this is an empty directory, exit
            print(f'\n! could not find any directories to clean in {directory}')
            os.chdir(f'{cwd}')
            return
        elif snapshot_directory + '/' not in directory_names:
            # this is a directory of simulation directories, recursively run on each one, then exit
            for directory_name in directory_names:
                clean_directories(
                    directory_name,
                    gizmo_directory,
                    snapshot_directory,
                    restart_directory,
                    gizmo_out_file,
                    gizmo_err_file,
                    snapshot_scalefactor_file,
                )
            os.chdir(f'{cwd}')
            return

        if os.path.exists(f'{gizmo_directory}'):
            # clean directory of gizmo source code
            # save config file, move to simulation directory
            os.chdir(f'{gizmo_directory}')
            print(f'* cleaning + tar-ing:  {gizmo_directory}/')
            os.system(f'mv {gizmo_config_file_used} ../{gizmo_config_file_save}')
            os.system('make clean')

            if os.path.exists('.git'):
                version_control = 'git'
            elif os.path.exists('.hg'):
                version_control = 'hg'
            else:
                version_control = None
            # append to the gizmo_config_file the version of Gizmo used (if not already there)
            if version_control == 'git':
                if os.system(f'grep "# git" ../{gizmo_config_file_save}') > 0:
                    os.system(f'printf "\n# git version of Gizmo\n" >> ../{gizmo_config_file_save}')
                    os.system(f'git log -n 1 >> ../{gizmo_config_file_save}')
                os.system('git gc --aggressive --prune')  # prune old commits
            elif version_control == 'hg':
                if os.system(f'grep "# hg" ../{gizmo_config_file_save}') > 0:
                    os.system(f'printf "\n# hg version of Gizmo\n" >> ../{gizmo_config_file_save}')
                    os.system(f'hg log -l 1 >> ../{gizmo_config_file_save}')

            os.system('mv ../ewald_spc_table_64_dbl.dat ../spcool_tables ../TREECOOL -t .')
            os.chdir('..')

            # tar gizmo directory
            os.system(f'tar -cf {gizmo_directory}.tar {gizmo_directory}; rm -rf {gizmo_directory}')
        else:
            print(f'! could not find:  {gizmo_directory}/')

        # clean output files
        os.system(f'rm -f {gizmo_err_file}')
        if os.path.exists(f'{gizmo_out_file}'):
            os.system(f'head -1000 {gizmo_out_file} > {gizmo_out_file}.txt')
            os.system(f'rm -f {gizmo_out_file}')
        os.system(f'rm -f {snapshot_scalefactor_file}')

        # tar directory of gizmo_jobs
        if os.path.exists(f'{gizmo_job_directory}'):
            print(f'* tar-ing:  {gizmo_job_directory}/')
            os.system(f'tar -cvf {gizmo_job_directory}.tar {gizmo_job_directory}')
            os.system(f'rm -rf {gizmo_job_directory}')
        else:
            print(f'! could not find:  {gizmo_job_directory}/')

        # clean snapshot directory
        if os.path.exists(f'{snapshot_directory}'):
            os.chdir(f'{snapshot_directory}')
            print(f'* cleaning:  {snapshot_directory}/')
            os.system(f'rm -rf {restart_directory}')
            os.system('rm -f HIIheating.txt MomWinds.txt sfr.txt SNeIIheating.txt')
            os.chdir('..')
        else:
            print(f'! could not find:  {snapshot_directory}/')

        # clean directory of initial conditions
        # if os.path.exists(f'{ic_directory}'):
        #    os.chdir(f'{ic_directory}')
        #    print(f'* cleaning:  {ic_directory}/')
        #    os.system('rm -f input_powerspec*.txt')
        #    os.system('rm -f *.wnoise')
        #    os.chdir('..')
        # else:
        #    print(f'! could not find {ic_directory}/ to clean')

        # clean backup files
        os.system('rm -f *~ .#* ._* /#*#')

        # move back to original directory
        os.chdir(f'{cwd}')


def archive_directories(
    directories='.',
    snapshot_directory='output',
    ic_directory='initial_condition',
    halo_directory='halo',
    rockstar_directory='rockstar_dm',
    rockstar_job_directory='rockstar_jobs',
    rockstar_catalog_directory='catalog',
    rockstar_hdf5_directory='catalog_hdf5',
    delete_directories=False,
    delete_tarballs=False,
    thread_number=1,
):
    '''
    Use tar to combine simulation sub-directories into single tar-ball files.
    Run this on a single simulation directory, a list of simulation directories,
    or a directory of multiple simulation directories.
    Run this after you run clean_directory(), to reduce the file count for archival/tape storage.
    By default, this stores the original sub-directories after tar-ring them, but you can delete
    the directories (if you are running this on the archival/tape server directly) by inputing
    delete_directories=True.
    To delete the tar-balls that this function creates (if you are running on live scratch space),
    simply input delete_tarballs=True.

    Parameters
    ----------
    directories : str or list thereof : directory[s] to run this on
        can be a single simulation directory, a list of simulation directories,
        or a directory that contains multiple simulation directories for which this function will
        run recursively on each one
    snapshot_directory : str : output directory that contains snapshots
    ic_directory : str : directory that contains initial condition files from MUSIC
    halo_directory : str : directory of (all) halo files/directories
    rockstar_directory : str : directory of (all) Rockstar files/directories
    rockstar_job_directory : str : directory of Rockstar run-time log/job files
    rockstar_catalog_directory : str : directory of Rockstar (text) halo catalog + tree files
    rockstar_hdf5_directory : str : directory of Rockstar post-processed hdf5 catalog + tree files
    delete_directories : bool :
        whether to delete the (raw) directories after tar-ing them into a single file
    delete_tarballs : bool : whether to delete existing tar-balls
        use this to clean safely the tar-balls that this function creates
    thread_number : int : number of parallel threads to use for tar-ing snapshots
    '''

    def tar_halo_directory(directory_name, delete_directories):
        if os.path.exists(f'{directory_name}'):
            print(f'\n* tar-ing:  {directory_name}/')
            os.system(f'tar -cf {directory_name}.tar {directory_name}')
            if delete_directories:
                print(f'* deleting:  {directory_name}/')
                os.system(f'rm -rf {directory_name}')

    def tar_snapshot_directory(directory_name, delete_directories):
        print(f'* tar-ing:  {directory_name}/')
        os.system(f'tar -cf {directory_name}.tar {directory_name}')
        if delete_directories:
            print(f'* deleting:  {directory_name}/')
            os.system(f'rm -rf {directory_name}')

    if np.isscalar(directories):
        directories = [directories]

    if thread_number > 1:
        import multiprocessing as mp

    # move to this directory
    cwd = os.getcwd()

    # move into each directory
    for directory in directories:
        directory = directory.rstrip('/')
        if directory != '.':
            print(f'\n\n* moving into:  {directory}/')
            os.chdir(f'{directory}')

        # check if this directory has relevant simulation directories,
        # or if need to run recursively on different simulation directories
        directory_names = glob.glob('*/')  # get names of all directories
        directory_names.sort()
        if len(directory_names) == 0:
            # this is an empty directory, exit
            print(f'\n! could not find any directories to tar in {directory}')
            os.chdir(f'{cwd}')
            return
        elif snapshot_directory + '/' not in directory_names:
            # this is a directory of simulation directories, recursively run on each one, then exit
            for directory_name in directory_names:
                archive_directories(
                    directory_name,
                    snapshot_directory,
                    ic_directory,
                    halo_directory,
                    rockstar_directory,
                    rockstar_job_directory,
                    rockstar_catalog_directory,
                    rockstar_hdf5_directory,
                    delete_directories,
                    delete_tarballs,
                )
            os.chdir(f'{cwd}')
            return

        # tar directory of initial conditions
        if delete_tarballs:
            print(f'\n* deleting:  {ic_directory}.tar')
            os.system(f'rm -f {ic_directory}.tar')
        else:
            if os.path.exists(f'{ic_directory}'):
                print(f'\n* tar-ing:  {ic_directory}/')
                os.system(f'tar -cvf {ic_directory}.tar {ic_directory}')
                if delete_directories:
                    print(f'* deleting:  {ic_directory}/')
                    os.system('rm -rf {ic_directory}')
            else:
                print(f'\n! could not find:  {ic_directory}/')

        # tar directories of halo catalogs + trees
        if os.path.exists(f'{halo_directory}/{rockstar_directory}'):
            print(f'\n* moving into:  {halo_directory}/{rockstar_directory}/')
            os.chdir(f'{halo_directory}/{rockstar_directory}')
            if delete_tarballs:
                print(f'* deleting:  {rockstar_job_directory}.tar')
                os.system(f'rm -f {rockstar_job_directory}.tar')
                print(f'* deleting:  {rockstar_catalog_directory}.tar')
                os.system(f'rm -f {rockstar_catalog_directory}.tar')
                print(f'* deleting:  {rockstar_hdf5_directory}.tar')
                os.system(f'rm -f {rockstar_hdf5_directory}.tar')
            else:
                halo_subdirectories = [
                    rockstar_job_directory,
                    rockstar_catalog_directory,
                    rockstar_hdf5_directory,
                ]

                if thread_number > 1:
                    # tar each snapshot directory in parallel
                    pool = mp.Pool(thread_number)

                for halo_subdirectory in halo_subdirectories:
                    if thread_number > 1:
                        pool.apply_async(
                            tar_halo_directory, (halo_subdirectory, delete_directories)
                        )
                    else:
                        tar_halo_directory(halo_subdirectory, delete_directories)

                # close threads
                if thread_number > 1:
                    pool.close()
                    pool.join()

            os.chdir('../..')
        else:
            print(f'\n! could not find:  {halo_directory}/{rockstar_directory}/')

        # tar each snapshot directory
        if os.path.exists(f'{snapshot_directory}'):
            os.chdir(f'{snapshot_directory}')

            if delete_tarballs:
                print(f'\n* moving into:  {snapshot_directory}/')
                print('* deleting snapshot tar-balls')
                snapshot_names = glob.glob('snapdir_*.tar')
                snapshot_names.sort()
                for snapshot_directory_name in snapshot_names:
                    print(f'* deleting:  {snapshot_directory_name}.tar')
                    os.system(f'rm -f {snapshot_directory_name}.tar')

            else:
                snapshot_names = glob.glob('snapdir_*')
                # ensure not tar an existing tar file
                snapshot_names = [s for s in snapshot_names if '.tar' not in s]
                snapshot_names.sort()
                if len(snapshot_names) > 0:
                    print(f'\n* moving into:  {snapshot_directory}/')
                    print('* tar-ing snapshot directories')

                if thread_number > 1:
                    # tar each snapshot directory in parallel
                    pool = mp.Pool(thread_number)

                for snapshot_name in snapshot_names:
                    if thread_number > 1:
                        pool.apply_async(
                            tar_snapshot_directory, (snapshot_name, delete_directories)
                        )
                    else:
                        tar_snapshot_directory(snapshot_name, delete_directories)

                # close threads
                if thread_number > 1:
                    pool.close()
                    pool.join()

            os.chdir('..')
        else:
            print(f'\n! could not find:  {snapshot_directory}/')

        # clean backup files
        os.system('rm -f *~ .#* ._* /#*#')

        # move back to original directory
        os.chdir(f'{cwd}')


def delete_snapshots(
    simulation_directory='.',
    snapshot_directory='output',
    snapshot_index_limits=[1, 599],
    delete_halos=False,
):
    '''
    Delete all snapshots in simulation_directory/snapshot_directory/ that are within
    snapshot_index_limits, except for those in snapshot_indices_keep list.

    Parameters
    ----------
    simulation_directory : str : directory of simulation
    snapshot_directory : str : directory of snapshot files
    snapshot_index_limits : list : min and max snapshot indices to delete
    delete_halos : bool : whether to delete halo catalog files at the same snapshots
    '''
    if not simulation_directory:
        simulation_directory = '.'
    simulation_directory = ut.io.get_path(simulation_directory)

    snapshot_name_base = 'snap*_{:03d}*'
    if not snapshot_directory:
        snapshot_directory = 'output/'
    snapshot_directory = ut.io.get_path(snapshot_directory)

    halo_name_base = 'halos_{:03d}*'
    halo_directory = 'halo/rockstar_dm/catalog/'

    if snapshot_index_limits is None or len(snapshot_index_limits) == 0:
        snapshot_index_limits = [1, 599]
    snapshot_indices = np.arange(snapshot_index_limits[0], snapshot_index_limits[1] + 1)

    print()
    for snapshot_index in snapshot_indices:
        if snapshot_index not in snapshot_indices_keep:
            snapshot_name = (
                simulation_directory
                + snapshot_directory
                + snapshot_name_base.format(snapshot_index)
            )
            print(f'* deleting:  {snapshot_name}')
            os.system(f'rm -rf {snapshot_name}')

            if delete_halos:
                halo_name = (
                    simulation_directory + halo_directory + halo_name_base.format(snapshot_index)
                )
                print(f'* deleting:  {halo_name}')
                os.system(f'rm -rf {halo_name}')
    print()


# --------------------------------------------------------------------------------------------------
# transfer files via globus
# --------------------------------------------------------------------------------------------------
class GlobusClass(ut.io.SayClass):
    '''
    Tranfer files via Globus command-line utility.
    '''

    def submit_transfer(
        self,
        simulation_path_directory='.',
        snapshot_directory='output',
        batch_file_name='globus_batch.txt',
        machine_name='peloton',
    ):
        '''
        Submit transfer of simulation files via Globus command-line utility.
        Must initiate from Stampede.

        Parameters
        ----------
        simulation_path_directory : str : '.' or full path + directory of simulation
        snapshot_directory : str : directory of snapshot files within simulation_directory
        batch_file_name : str : name of file to write
        machine_name : str : name of machine transfering files to
        '''
        # set directory from which to transfer
        simulation_path_directory = ut.io.get_path(simulation_path_directory)
        if simulation_path_directory == './':
            simulation_path_directory = os.getcwd()
        if simulation_path_directory[-1] != '/':
            simulation_path_directory += '/'

        # preceeding '/' already in globus bookmark
        command = f'globus transfer $(globus bookmark show stampede){simulation_path_directory[1:]}'

        path_directories = simulation_path_directory.split('/')
        simulation_directory = path_directories[-2]

        # parse machine + directory to transfer to
        if machine_name == 'peloton':
            if 'elvis' in simulation_directory:
                directory_to = 'm12_elvis'
            else:
                directory_to = simulation_directory.split('_')[0]
            directory_to += '/' + simulation_directory + '/'

            command += f' $(globus bookmark show peloton-scratch){directory_to}'

        # set globus parameters
        command += ' --sync-level=checksum --preserve-mtime --verify-checksum'
        command += f' --label "{simulation_directory}" --batch < {batch_file_name}'

        # write globus batch file
        self.write_batch_file(simulation_path_directory, snapshot_directory, batch_file_name)

        self.say(f'* executing:\n{command}\n')
        os.system(command)

    def write_batch_file(
        self, simulation_directory='.', snapshot_directory='output', file_name='globus_batch.txt'
    ):
        '''
        Write a batch file that sets files to transfer via globus.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        snapshot_directory : str : directory of snapshot files within simulation_directory
        file_name : str : name of batch file to write
        '''
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        transfer_string = ''

        # general files
        transfer_items = [
            'gizmo/',
            'gizmo_config.sh',
            'gizmo_parameters.txt',
            'gizmo_parameters.txt-usedvalues',
            'gizmo.out.txt',
            'snapshot_times.txt',
            'notes.txt',
            'track/',
            'halo/rockstar_dm/catalog_hdf5/',
        ]
        for transfer_item in transfer_items:
            if os.path.exists(simulation_directory + transfer_item):
                command = '{} {}'
                if transfer_item[-1] == '/':
                    transfer_item = transfer_item[:-1]
                    command += ' --recursive'
                command = command.format(transfer_item, transfer_item) + '\n'
                transfer_string += command

        # initial condition files
        transfer_items = glob.glob(simulation_directory + 'initial_condition*/*')
        transfer_items.sort()
        for transfer_item in transfer_items:
            if '.ics' not in transfer_item:
                transfer_item = transfer_item.replace(simulation_directory, '')
                command = '{} {}\n'.format(transfer_item, transfer_item)
                transfer_string += command

        # snapshot files
        for snapshot_index in snapshot_indices_keep:
            snapshot_name = '{}snapdir_{:03d}'.format(snapshot_directory, snapshot_index)
            if os.path.exists(simulation_directory + snapshot_name):
                snapshot_string = f'{snapshot_name} {snapshot_name} --recursive\n'
                transfer_string += snapshot_string

            snapshot_name = '{}snapshot_{:03d}.hdf5'.format(snapshot_directory, snapshot_index)
            if os.path.exists(simulation_directory + snapshot_name):
                snapshot_string = f'{snapshot_name} {snapshot_name}\n'
                transfer_string += snapshot_string

        with open(file_name, 'w') as file_out:
            file_out.write(transfer_string)


Globus = GlobusClass()


# --------------------------------------------------------------------------------------------------
# transfer files via rsync
# --------------------------------------------------------------------------------------------------
class RsyncClass(ut.io.SayClass):
    '''
     Use rsync to copy simulations files from remote machine to local directory.
    '''

    def __init__(self):
        '''
        .
        '''
        self.rsync_command = 'rsync -ahvP --size-only '
        self.snapshot_name_base = 'snap*_{:03d}*'

    def rsync_snapshot_files(
        self,
        machine_from,
        simulation_directory_from='',
        simulation_directory_to='.',
        snapshot_indices=snapshot_indices_keep,
    ):
        '''
        Use rsync to copy snapshot files from a single simulations directory on a remote machine to
        a local simulation directory.

        Parameters
        ----------
        machine_from : str : name of (remote) machine to copy from:
            'pfe', 'stampede', 'frontera', 'peloton'
        directory_from : str : directory to copy from
        directory_to : str : local directory to put snapshots
        snapshot_indices : int or list : index[s] of snapshots to transfer
        '''
        directory_from = ut.io.get_path(simulation_directory_from) + 'output/'
        directory_to = ut.io.get_path(simulation_directory_to) + 'output/.'

        if np.isscalar(snapshot_indices):
            snapshot_indices = [snapshot_indices]

        snapshot_path_names = ''
        for snapshot_index in snapshot_indices:
            snapshot_path_names += (
                directory_from + self.snapshot_name_base.format(snapshot_index) + ' '
            )

        command = self.rsync_command + f'{machine_from}:"{snapshot_path_names}" {directory_to}'
        print(f'\n* executing:\n{command}\n')
        os.system(command)

        # fix file permissions (especially relevant if transfer from Stampede)
        os.system('chmod u=rw,go=r $(find . -type f); chmod u=rwX,go=rX $(find . -type d)')

    def rsync_simulation_files(
        self,
        machine_from,
        directory_from='/scratch/projects/xsede/GalaxiesOnFIRE',
        directory_to='.',
        include_snapshot600=False,
    ):
        '''
        Use rsync to copy (non-snapshot) files from remote machine to local directory.
        Directory can be a single simulation directory or a directory of simulation directories.

        Parameters
        ----------
        machine_from : str : name of (remote) machine to copy from:
            'pfe', 'stampede', 'frontera', 'peloton'
        directory_from : str : directory to copy from
        directory_to : str : directory to copy files to
        '''
        include_names = []
        if include_snapshot600:
            include_names.append('output/snap*_600*')

        exclude_names = [
            #'output/',
            'snapdir_*',
            'snapshot_*.hdf5',
            'ewald_spc_table_64_dbl.dat',
            'spcool_tables/',
            'TREECOOL',
            'restartfiles/',
            'energy.txt',
            'balance.txt',
            'GasReturn.txt',
            'HIIheating.txt',
            'MomWinds.txt',
            'SNeIIheating.txt',
            '*.ics',
            'submit_music*',
            'input_powerspec.txt',
            'snapshot_scale-factors.txt',
            'submit_gizmo*.py',
            '*.bin',
            '*.particles',
            '*.bak',
            '*.err',
            '*.pyc',
            '*.o',
            '*.pro',
            '*.perl',
            '.ipynb_checkpoints',
            '.slurm',
            '.DS_Store',
            '*~',
            '._*',
            '#*#',
        ]

        directory_from = machine_from + ':' + ut.io.get_path(directory_from)
        directory_to = ut.io.get_path(directory_to)

        arguments = ''

        if len(include_names) > 0:
            for include_name in include_names:
                arguments += f'--include="{include_name}" '

        for exclude_name in exclude_names:
            arguments += f'--exclude="{exclude_name}" '

        command = self.rsync_command + arguments + directory_from + ' ' + directory_to + '.'
        print(f'\n* executing:\n{command}\n')
        os.system(command)

        # fix file permissions (especially relevant if transfer from Stampede)
        os.system('chmod u=rw,go=r $(find . -type f); chmod u=rwX,go=rX $(find . -type d)')


Rsync = RsyncClass()


# --------------------------------------------------------------------------------------------------
# running from command line
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function to run: compress, clean, archive, delete, globus, rsync')

    function_kind = str(sys.argv[1])

    assert (
        'compress' in function_kind
        or 'clean' in function_kind
        or 'archive' in function_kind
        or 'delete' in function_kind
        or 'rsync' in function_kind
        or 'globus' in function_kind
    )

    if 'compress' in function_kind:
        simulation_directory = '.'
        if len(sys.argv) > 2:
            simulation_directory = str(sys.argv[2])

        snapshot_index_limits = [0, 600]
        if len(sys.argv) > 3:
            snapshot_index_limits[0] = int(sys.argv[3])
            if len(sys.argv) > 4:
                snapshot_index_limits[1] = int(sys.argv[4])

        snapshot_indices = np.arange(snapshot_index_limits[0], snapshot_index_limits[1] + 1)

        Compress.test_compression(simulation_directory, snapshot_indices=snapshot_indices)

    elif 'clean' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])
        clean_directories(directory)

        if 'archive' in function_kind:
            archive_directories(directory)

    elif 'archive' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])
        archive_directories(directory)

    elif 'delete' in function_kind:
        simulation_directory = '.'
        if len(sys.argv) > 2:
            simulation_directory = str(sys.argv[2])

        snapshot_index_limits = None
        if len(sys.argv) > 3:
            snapshot_index_limits = [int(sys.argv[3]), int(sys.argv[4])]

        delete_snapshots(simulation_directory, snapshot_index_limits=snapshot_index_limits)

    elif 'globus' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])
        Globus.submit_transfer(directory)

    elif 'rsync' in function_kind:
        if len(sys.argv) < 5:
            raise OSError('imports: machine_from directory_from directory_to')

        machine_from = str(sys.argv[2])
        directory_from = str(sys.argv[3])
        directory_to = str(sys.argv[4])

        Rsync.rsync_simulation_files(
            machine_from, directory_from, directory_to, include_snapshot600=True,
        )
        # Rsync.rsync_snapshot_files(machine_from, directory_from, directory_to)
