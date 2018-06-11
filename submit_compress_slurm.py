#!/usr/bin/env python3

#SBATCH --job-name=gizmo_compress
#SBATCH --partition=skx-normal    # SKX node: 48 cores, 4 GB per core, 192 GB total
##SBATCH --partition=normal    ## KNL node: 64 cores x 2 FP threads, 1.6 GB per core, 96 GB total
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=24:00:00
#SBATCH --output=gizmo_compress_job_%j.txt
#SBATCH --mail-user=arwetzel@gmail.com
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140064

'''
Submit compression of gizmo snapshot files to queue.
Submit this script from within the primary directory of the simulation.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
# local ----
from utilities.basic import io as ut_io
from gizmo_analysis import gizmo_file

# print run-time and CPU information
ScriptPrint = ut_io.ScriptPrintClass('slurm')
ScriptPrint.print_initial()

# check if any input arguments
snapshot_index_max = 600
if len(os.sys.argv) > 1:
    snapshot_index_max = str(os.sys.argv[1])
snapshot_index_limits = [0, 600]

# execute
gizmo_file.compress_snapshots(snapshot_index_limits=snapshot_index_limits)

# print run-time information
ScriptPrint.print_final()
