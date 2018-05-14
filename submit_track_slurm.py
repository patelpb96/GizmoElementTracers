#!/usr/bin/env python3

#SBATCH --job-name=gizmo_track
#SBATCH --partition=skx-normal    # SKX node: 48 cores, 4 GB per core, 192 GB total
##SBATCH --partition=normal    ## KNL node: 64 cores x 2 FP threads, 1.6 GB per core, 96 GB total
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=48:00:00
#SBATCH --output=track/gizmo_track_job_%j.txt
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
from gizmo_analysis import gizmo_track

species_name = 'star'  # which particle species to track

# print run-time and CPU information
ScriptPrint = ut_io.ScriptPrintClass('slurm')
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
    ParticleIndexPointer = gizmo_track.ParticleIndexPointerClass(species_name)
    ParticleIndexPointer.write_index_pointers_to_snapshots(thread_number=ScriptPrint.omp_number)

if 'coordinates' in function_kind:
    ParticleCoordinate = gizmo_track.ParticleCoordinateClass(species_name)
    ParticleCoordinate.write_formation_coordinates()

# print run-time information
ScriptPrint.print_final()
