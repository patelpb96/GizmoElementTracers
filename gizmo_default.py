#!/usr/bin/env python3

'''
Default names and values for files and directories used throughout the gizmo_analysis package.
If you prefer a different default, change it here and it should propagate througout the package.
Some names have wildcards, such as '*', or '!', these represent name bases, generally for finding
any/all such files in a directory via glob.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# simulation ----------
# base directory of a simulation
simulation_directory = '.'


# snapshots ----------
# directory of snapshot files and other Gizmo output files (such as cpu.txt)
snapshot_directory = 'output/'

# name base of snapshot files/directories, to input to glob to find all files/directories
snapshot_name_base = 'snap*[!txt]'

# default snapshot index (typically z = 0)
snapshot_index = 600

# name of file that lists (only) snapshot scale-factors
snapshot_scalefactor_file_name = 'snapshot_scale-factors.txt'

# name of file that lists snapshot full time information
snapshot_time_file_name = 'snapshot_times.txt'

# directory within snapshot_directory that stores restart files
restart_directory = 'restartfiles/'

# name (base) of files for Gizmo restart
restart_file_name = 'restart.*'


# Gizmo----------
# directory of Gizmo source code
gizmo_directory = 'gizmo/'

# name (base) of file that Gizmo write main run-time information
gizmo_out_file_name = 'gizmo.out*'

# name of file that Gizmo write errors to
gizmo_err_file_name = 'gizmo.err'

# name of file that stores Gizmo CPU wall-times
gizmo_cpu_file_name = 'cpu.txt'

# directory where keep slurm/pbs job files
gizmo_job_directory = 'gizmo_jobs/'


# particle tracking ----------
# directory of particle tracking files, including stored hosts coordinates across all snapshots
track_directory = 'track/'

# file that contains hostst coordinates, rotation tensors, and axis ratios across all snapshots
# and also stores star particle formation coordinates
hosts_coordinates_file_name = 'hosts_coordinates.hdf5'


# initial condition ----------
# directory of initial condition files
ic_directory = 'initial_condition/'

# name (base) for MUSIC config file - read (via glob) to get all cosmological parameters
music_config_file_name = '*.conf'
