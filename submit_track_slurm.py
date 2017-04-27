#!/usr/bin/env python

#SBATCH --job-name=gizmo_track
#SBATCH --partition=serial
##SBATCH --partition=development
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=12:00:00
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


species = 'star'  # which particle species to track


ScriptPrint = ut_io.ScriptPrintClass('slurm')
ScriptPrint.print_initial()

# check if any input arguments
if len(os.sys.argv) > 1:
    function_kind = str(os.sys.argv[1])
else:
    function_kind = 'indices'  # default is to assign just index pointers
print('executing function[s]: {}'.format(function_kind))
os.sys.stdout.flush()

# execute
if 'indices' in function_kind:
    IndexPointer = gizmo_track.IndexPointerClass(species)
    IndexPointer.write_index_pointer(match_prop_name='id.child', test_prop_name='form.scalefactor')

if 'distances' in function_kind:
    HostDistance = gizmo_track.HostDistanceClass(species)
    HostDistance.write_form_host_distance()

ScriptPrint.print_final()
