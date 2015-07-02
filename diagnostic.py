#!/usr/bin/env python


'''
Get run time (average per MPI taks) at given scale factors from cpu.txt for Gizmo run.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
# local ----
from utilities import utility as ut


def print_run_times(directory='.', print_lines=False):
    '''
    .
    '''
    scale_factors = np.array([
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.666, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    run_times = []

    file_name = 'cpu.txt'

    file_path_name = ut.io.get_path(directory) + file_name

    file_in = open(file_path_name, 'r')

    a_i = 0
    print_next_line = False
    for line in file_in:
        if 'Time: %.3f' % scale_factors[a_i] in line or 'Time: 1,' in line:
            if print_lines:
                print(line, end='')
            print_next_line = True
        elif print_next_line:
            if print_lines:
                print(line)
            run_times.append(float(line.split()[1]))
            print_next_line = False
            a_i += 1
            if a_i >= len(scale_factors):
                break

    run_times = np.array(run_times)

    print('# scale-factor redshift run-time-percent run-time[hr]')
    for a_i in xrange(len(run_times)):
        print('%.2f %5.2f  %5.1f  %7.1f' %
              (scale_factors[a_i], 1 / scale_factors[a_i] - 1,
               100 * run_times[a_i] / run_times.max(), run_times[a_i] / 3600))

    #for scale_factor in scale_factors:
    #    os.system('grep "Time: %.2f" %s --after-context=1 --max-count=2' %
    #              (scale_factor, file_path_name))


#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = str(sys.argv[1])
    else:
        directory = '.'

    print_run_times(directory)
