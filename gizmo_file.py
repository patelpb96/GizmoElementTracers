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
# clean simulation directory
# --------------------------------------------------------------------------------------------------
def clean_directory(
    simulation_directory='.',
    gizmo_directory='gizmo',
    snapshot_directory='output',
    gizmo_out_file='gizmo.out',
    gizmo_err_file='gizmo.err',
    snapshot_scalefactor_file='snapshot_scale-factors.txt',
    restart_directory='restartfiles',
):
    '''
    Clean Gizmo simulation directory. Run this after a simulation finishes.

    Parameters
    ----------
    '''
    gizmo_config_file = 'gizmo_config.h'  # file to save used config settings and gizmo version

    # move to this directory
    cwd = os.getcwd()
    os.system(f'cd {simulation_directory}')

    # clean gizmo source code
    os.system(f'mv {gizmo_directory}/GIZMO_config.h {gizmo_config_file}')  # save used config
    os.chdir(f'{gizmo_directory}')
    os.system('make clean')
    os.system(f'echo "\n# git version of gizmo" >> ../{gizmo_config_file}')  # save git version
    os.system(f'git log -n 1 >> ../{gizmo_config_file}')
    os.chdir('..')
    os.system(f'mv ewald_spc_table_64_dbl.dat spcool_tables TREECOOL -t {gizmo_directory}/')
    os.system(f'rm -f {snapshot_scalefactor_file}')
    os.system(f'rm -f {gizmo_err_file}')

    # clean output files
    os.system(f'head -1000 {gizmo_out_file} > {gizmo_out_file}.txt')
    os.system(f'rm -f {gizmo_out_file}')
    os.system(f'rm -rf {snapshot_directory}/{restart_directory}')
    os.system(f'rm -f {snapshot_directory}/HIIheating.txt')
    os.system(f'rm -f {snapshot_directory}/MomWinds.txt')
    os.system(f'rm -f {snapshot_directory}/sfr.txt')
    os.system(f'rm -f {snapshot_directory}/SNeIIheating.txt')

    # clean backup files
    os.system('rm -f *~ .#* ._* /#*#')

    # move back to original directory
    os.system(f'cd {cwd}')


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
        snapshot_index=600
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
# transfer files via globus
# --------------------------------------------------------------------------------------------------
class GlobusClass(ut.io.SayClass):
    '''
    .
    '''

    def submit_transfer(
        self,
        simulation_path_directory='.',
        snapshot_directory='output',
        batch_file_name='globus_batch.txt',
        machine_name='peloton',
    ):
        '''
        Submit globus transfer of simulation files.
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
        Write batch file that sets files to transfer via globus.

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
def rsync_snapshots(
    machine_name,
    simulation_directory_from='',
    simulation_directory_to='.',
    snapshot_indices=snapshot_indices_keep,
):
    '''
    Use rsync to copy snapshot file[s].

    Parameters
    ----------
    machine_name : str : 'pfe', 'stampede', 'bw', 'peloton'
    directory_from : str : directory to copy from
    directory_to : str : local directory to put snapshots
    snapshot_indices : int or list : index[s] of snapshots to transfer
    '''
    snapshot_name_base = 'snap*_{:03d}*'

    directory_from = ut.io.get_path(simulation_directory_from) + 'output/'
    directory_to = ut.io.get_path(simulation_directory_to) + 'output/.'

    if np.isscalar(snapshot_indices):
        snapshot_indices = [snapshot_indices]

    snapshot_path_names = ''
    for snapshot_index in snapshot_indices:
        snapshot_path_names += directory_from + snapshot_name_base.format(snapshot_index) + ' '

    command = 'rsync -ahvP --size-only '
    command += f'{machine_name}:"{snapshot_path_names}" {directory_to}'
    print(f'\n* executing:\n{command}\n')
    os.system(command)


def rsync_simulation_files(
    machine_name, directory_from='/oldscratch/projects/xsede/GalaxiesOnFIRE', directory_to='.'
):
    '''
    Use rsync to copy simulation files.

    Parameters
    ----------
    machine_name : str : 'pfe', 'stampede', 'bw', 'peloton'
    directory_from : str : directory to copy from
    directory_to : str : directory to put files
    '''
    excludes = [
        'output/',
        'restartfiles/',
        'ewald_spc_table_64_dbl.dat',
        'spcool_tables/',
        'TREECOOL',
        'energy.txt',
        'balance.txt',
        'GasReturn.txt',
        'HIIheating.txt',
        'MomWinds.txt',
        'SNeIIheating.txt',
        '*.ics',
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

    directory_from = machine_name + ':' + ut.io.get_path(directory_from)
    directory_to = ut.io.get_path(directory_to)

    command = 'rsync -ahvP --size-only '

    arguments = ''
    for exclude in excludes:
        arguments += f'--exclude="{exclude}" '

    command += arguments + directory_from + ' ' + directory_to + '.'
    print(f'\n* executing:\n{command}\n')
    os.system(command)


# --------------------------------------------------------------------------------------------------
# delete files
# --------------------------------------------------------------------------------------------------
def delete_snapshots(
    simulation_directory='.',
    snapshot_directory='output',
    snapshot_index_limits=[1, 599],
    delete_halos=False,
):
    '''
    Delete all snapshots in given directory within snapshot_index_limits,
    except for those in snapshot_indices_keep list.

    Parameters
    ----------
    simulation_directory : str : directory of simulation
    snapshot_directory : str : directory of snapshots
    snapshot_index_limits : list : min and max snapshot indices to delete
    delete_halos : bool : whether to delete halo catalog files at same snapshot times
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
# running from command line
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function to run: clean, compress, globus, rsync, delete')

    function_kind = str(sys.argv[1])

    assert (
        'clean' in function_kind
        or 'compress' in function_kind
        or 'rsync' in function_kind
        or 'globus' in function_kind
        or 'delete' in function_kind
    )

    if 'clean' in function_kind:
        simulation_directory = '.'
        if len(sys.argv) > 2:
            simulation_directory = str(sys.argv[2])
        clean_directory(simulation_directory)

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

    elif 'globus' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])
        Globus.submit_transfer(directory)

    elif 'rsync' in function_kind:
        if len(sys.argv) < 5:
            raise OSError('imports: machine_name simulation_directory_from simulation_directory_to')

        machine_name = str(sys.argv[2])
        simulation_directory_from = str(sys.argv[3])
        simulation_directory_to = str(sys.argv[4])

        rsync_simulation_files(machine_name, simulation_directory_from, simulation_directory_to)
        rsync_snapshots(machine_name, simulation_directory_from, simulation_directory_to)

    elif 'delete' in function_kind:
        simulation_directory = '.'
        if len(sys.argv) > 2:
            simulation_directory = str(sys.argv[2])

        snapshot_index_limits = None
        if len(sys.argv) > 3:
            snapshot_index_limits = [int(sys.argv[3]), int(sys.argv[4])]

        delete_snapshots(simulation_directory, snapshot_index_limits=snapshot_index_limits)
