#!/usr/bin/env python3

#SBATCH --job-name=gizmo_track
#SBATCH --partition=normal
## Stampede node has 16 cores and 32 GB
## Stampede2 node has 64 (useable) cores, each with 2 FP threads, so 128 total, and 96 GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=24:00:00
#SBATCH --output=track/gizmo_track_%j.txt
#SBATCH --mail-user=arwetzel@gmail.com
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140064


'''
Submit gizmo particle tracking to queue.
Submit this script from within the primary directory of the simulation.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
# local ----
from utilities.basic import io as ut_io
from gizmo import gizmo_track


species_name = 'star'  # which particle species to track


ScriptPrint = ut_io.ScriptPrintClass('slurm')

# print run-time and CPU information
ScriptPrint.print_initial()

# check if any input arguments
if len(os.sys.argv) > 1:
    function_kind = str(os.sys.argv[1])
    assert ['indices' in function_kind or 'coordinates' in function_kind]
else:
    function_kind = 'indices'  # default is to assign only index pointers
print('executing function[s]: {}'.format(function_kind))
os.sys.stdout.flush()

# execute
if 'indices' in function_kind:
    IndexPointer = gizmo_track.IndexPointerClass(species_name)
    IndexPointer.write_index_pointer()

if 'coordinates' in function_kind:
    HostCoordinate = gizmo_track.HostCoordinateClass(species_name)
    HostCoordinate.write_formation_coordinates()

ScriptPrint.print_final()
