#!/home1/02769/arwetzel/local/anaconda/bin/python
##!/usr/bin/env python

## job name
#SBATCH -J music
## queue type
#SBATCH -p normal
## total CPU resource allocation
## each node on Stampede has 16 processors & 32 GB
#SBATCH -N 1 -n 16
## runtime allocation
#SBATCH -t 2:00:00:00
## email notification
#SBATCH --mail-user=arwetzel@gmail.com
#SBATCH --mail-type=end
# allocation to charge
#SBATCH -A TG-AST130039


'''
Submit MUSIC run to queue.

@author: Andrew Wetzel
'''

from __future__ import absolute_import, division, print_function
import os
import time
import datetime


# directories & file names
executable = '%s/simulation/music/MUSIC' % os.environ['HOME']    # executable
parameter_file_name = 'ic_*.conf'    # name of parameter file

# parallel parameters
omp_num = 16    # number of OpenMP threads per MPI process
os.system('export OMP_NUM_THREADS=%d' % omp_num)


os.system('date')    # print time
time_ini = time.time()


# execute
os.system('ibrun tacc_affinity ./%s %s' % (executable, parameter_file_name))


# print wall time information
time_dif = time.time() - time_ini
time_dif_str = str(datetime.timedelta(seconds=time_dif))
print('wall time: %.1f sec = %s' % (time_dif, time_dif_str.split('.')[0]))
os.sys.stdout.flush()
os.system('date')    # print time
