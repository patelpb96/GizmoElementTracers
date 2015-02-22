#!/home1/02769/arwetzel/local/anaconda/bin/python
##!/usr/bin/env python

## job name
#PBS -N music
## queue type
##PBS -q longJobsQ
## total CPU resource allocation
## each node on Zwicky has 12 processors & 24 GB
#PBS -l nodes=1:ppn=12
## wall time allocation
#PBS -l walltime=00:08:00:00
## combine stderr & stdout into one file
#PBS -j oe
## where to put run time information
#PBS -o $PBS_JOBNAME_job$PBS_JOBID.txt
## email results: a = aborted, b = begin, e = end
#PBS -M arwetzel@gmail.com
#PBS -m a
## import terminal environmental variables
#PBS -V

'''
Submit MUSIC run to queue.

@author: Andrew Wetzel
'''

from __future__ import absolute_import, division, print_function
import os
import time
import datetime


# set directories & file names
executable = '%s/simulation/music/MUSIC' % os.environ['HOME']    # executable
parameter_file_name = 'ic_*.conf'    # name of parameter file

# set parallel parameters
omp_num = 12    # number of OpenMP threads per MPI process
os.system('export OMP_NUM_THREADS=%d' % omp_num)


os.chdir(os.environ['PBS_O_WORKDIR'])    # move to directory am in when submit this job

os.system('date')    # print time
time_ini = time.time()


# execute
os.system('%s %s' % (executable, parameter_file_name))


# print runtime information
time_dif = time.time() - time_ini
time_dif_str = str(datetime.timedelta(seconds=time_dif))
print('wall time: %.1f sec = %s' % (time_dif, time_dif_str.split('.')[0]))
os.sys.stdout.flush()
os.system('date')    # print time
