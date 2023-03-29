'''
Default names and values for files and directories used throughout this package.
If you prefer a different default, change it here, and it will propagate througout this package.
Some names have wildcards, such as '*', or '!': these represent name bases, generally for finding
any/all such files in a directory via glob.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# simulation ----------
# base directory of a simulation
# this packages assumes that all files/directories below are within simulation_directory
# setting to '.' assumes that you are running analysis from within a simulation directory
simulation_directory = '.'


# snapshots ----------
# directory of snapshot files and other Gizmo output files (such as cpu.txt)
snapshot_directory = 'output/'

# name of text file that lists (only) snapshot scale-factors
snapshot_scalefactor_file_name = 'snapshot_scalefactors.txt'

# name of text file that lists full time information of snapshots
snapshot_time_file_name = 'snapshot_times.txt'

# directory within snapshot_directory that stores restart files
restart_directory = 'restartfiles/'

# name (base) of restart files for Gizmo
restart_file_name = 'restart.*'


# Gizmo----------
# directory of Gizmo source code
gizmo_directory = 'gizmo/'

# name (base) of file to which Gizmo writes main run-time information
gizmo_out_file_name = 'gizmo.out*'

# name of file to which Gizmo write errors
gizmo_err_file_name = 'gizmo.err'

# name of file that stores CPU wall-times
gizmo_cpu_file_name = 'cpu.txt'

# directory to keep slurm/pbs job files
gizmo_job_directory = 'gizmo_jobs/'


# initial condition ----------
# directory of initial condition files
ic_directory = 'initial_condition/'

# name (base) for MUSIC config file - read (via glob) to get all cosmological parameters
music_config_file_name = '*.conf'


# particle tracking ----------
# directory of particle tracking files, including stored coordinates of hosts across all snapshots
track_directory = 'track/'

# file that contains for each primary host its coordinates, rotation tensors, and axis ratios
# across all snapshots, and also stores star particle formation coordinates
hosts_coordinates_file_name = 'host_coordinates.hdf5'


# FoF groups ----------
# directory of group files
group_directory = 'group/'
