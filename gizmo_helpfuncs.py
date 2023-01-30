'''
A set of helper functions designed to make everything easier (for me)
----------
@author: Preet Patel <pbpa@ucdavis.edu>
----------
'''

import os

class Batch:
    '''
    Idea: submit a python File to batch  
    '''
    def __init__(self,
        sbatch_file = "batch_script"):

        '''setting initial vars'''
        self.sbatch_filename = sbatch_file


    def conv2unix(self):
        print("Converting commands to UNIX...")
        os.system("dos2unix" + self.sbatch_filename)

    def submit(self):
        self.conv2unix()
        os.system("sbatch " + self.sbatch_filename)