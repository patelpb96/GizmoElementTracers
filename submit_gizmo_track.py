#!/usr/bin/env python

#SBATCH --job-name=gizmo_track
#SBATCH --partition=serial
##SBATCH --partition=development
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=2:00:00
#SBATCH --output=track/gizmo_track%j.txt
#SBATCH --mail-user=arwetzel@gmail.com
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140064


'''
Submit gizmo particle tracking to queue.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
import time
import datetime
# local ----
from gizmo import gizmo_track


# print job information
os.system('date')
print('')
print('job name = ' + os.environ['SLURM_JOB_NAME'])
print('job id = ' + os.environ['SLURM_JOB_ID'])
print('using {} nodes, {} MPI tasks per node, {} MPI tasks total'.format(
      os.environ['SLURM_NNODES'], os.environ['SLURM_TASKS_PER_NODE'], os.environ['SLURM_NTASKS']))
if int(os.environ['SLURM_CPUS_PER_TASK']) > 1:
    os.environ['OMP_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
    print('using {} OpenMP threads per MPI task'.format(os.environ['OMP_NUM_THREADS']))
cpu_number = int(os.environ['SLURM_NTASKS']) * int(os.environ['OMP_NUM_THREADS'])
print('using {} CPUs total'.format(cpu_number))

os.sys.stdout.flush()


# execute
time_ini = time.time()

gizmo_track.write_particle_index_pointer(
    species='star', match_prop_name='id.child', test_prop_name='form.scalefactor')


# print run time information
time_dif = time.time() - time_ini
time_dif_str = str(datetime.timedelta(seconds=time_dif))
print('wall time = {:.0f} sec = {:.2f} day = {}'.format(
      time_dif, time_dif / 3600 / 24, time_dif_str.split('.')[0]))
print('cpu time = {:.1f} hr\n'.format(time_dif * cpu_number / 3600))
os.sys.stdout.flush()
os.system('date')
