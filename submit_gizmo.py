#!/home1/02769/arwetzel/local/anaconda/bin/python
##!/usr/bin/env python

## job name
#PBS -N <name>
## queue type
##PBS -q <queue>
## total CPU resource allocation
## each node on Zwicky has 12 processors & 24 GB
#PBS -l nodes=8:ppn=12
## wall time allocation
#PBS -l walltime=02:00:00:00
## combine stderr & stdout into one file
#PBS -j oe
## where to put run time information
#PBS -o $PBS_JOBNAME.$PBS_JOBID
## email results: a = aborted, b = begin, e = end
#PBS -M arwetzel@gmail.com
#PBS -m ae
## import terminal environmental variables
#PBS -V

'''
Submit Gizmo run to queue.

@author: Andrew Wetzel
'''

from __future__ import absolute_import, division, print_function
import os
import time
import datetime


# directories & file names
executable = '~/simulation/gizmo/GIZMO'    # executable
parameter_file_name = 'gizmo_parameters.txt'    # name of parameter file
restart_from_restart = False
restart_from_snapshot = False

# parallel parameters
mpi_num_per_node = 12    # number of MPI tasks per node
node_num = 8    # number of nodes
mpi_num = mpi_num_per_node * node_num    # total number of MPI tasks across all nodes
omp_num = 4    # number of OpenMP threads per MPI process


os.chdir(os.environ['PBS_O_WORKDIR'])    # move to directory am in when submit this job

os.system('date')    # print time
time_ini = time.time()

# execute
#exec_command = 'mpirun -npernode %d ./%s %s' % (mpi_num_per_node, executable, parameter_file_name)
exec_command = 'mpirun -np %d ./%s %s' % (mpi_num, executable, parameter_file_name)

if restart_from_restart:
    exec_command += ' 1'
    #if not os.path.isfile(snapshot_directory + 'restartfiles'):
elif restart_from_snapshot:
    exec_command += ' 2'

exec_command += ' 1> gizmo.out 2> gizmo.err'

os.system(exec_command)


# print runtime information
time_dif = time.time() - time_ini
time_dif_str = str(datetime.timedelta(seconds=time_dif))
print('wall time: %.1f sec = %s' % (time_dif, time_dif_str.split('.')[0]))
os.sys.stdout.flush()
os.system('date')    # print time
