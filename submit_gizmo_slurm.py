#!/home1/02769/arwetzel/local/anaconda/bin/python
##!/usr/bin/env python

## job name
#SBATCH -J m12i_ref12_dark
## queue type
#SBATCH -p normal
## total CPU resource allocation
## each node on Stampede has 16 processors & 32 GB
#SBATCH -N 16 -n 256
## wall time allocation
#SBATCH -t 2:00:00:00
## email notification
#SBATCH --mail-user=arwetzel@gmail.com
#SBATCH --mail-type=end
# allocation to charge
#SBATCH -A TG-AST130039

'''
Submit Gizmo run to queue.

@author: Andrew Wetzel
'''

from __future__ import absolute_import, division, print_function
import os
import time
import datetime


# directories & file names
executable = './gizmo/GIZMO'    # executable
parameter_file_name = 'gizmo_parameters.txt'    # name of parameter file
restart_from_restart = False
restart_from_snapshot = False

# parallel parameters
omp_num = 16    # number of OpenMP threads per MPI process
os.system('export OMP_NUM_THREADS=%d' % omp_num)


os.system('date')    # print time
time_ini = time.time()


# execute
exec_command = 'ibrun tacc_affinity ./%s %s' % (executable, parameter_file_name)

if restart_from_restart:
    exec_command += ' 1'
    #if not os.path.isfile(snapshot_directory + 'restartfiles'):
elif restart_from_snapshot:
    exec_command += ' 2'

os.system(exec_command)


# print wall time information
time_dif = time.time() - time_ini
time_dif_str = str(datetime.timedelta(seconds=time_dif))
print('wall time: %.1f sec = %s' % (time_dif, time_dif_str.split('.')[0]))
os.sys.stdout.flush()
os.system('date')    # print time
