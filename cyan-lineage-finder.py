#################################################################################
#                                                                               #
#         cyan-lineage-finder.py heavily based on yt/merger_tree.py             #
#                                                                               #
#                     written by Ji-hoon Kim, 08/2012                           #
#                                                                               #
#################################################################################

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.figure
import matplotlib.backends.backend_agg
import pylab
from numpy import pi
from matplotlib.colors import LogNorm, Normalize
from yt.mods import *
import yt.utilities.amr_utils as au
import yt.utilities.pydot as pydot
from yt.analysis_modules.halo_merger_tree.api import *
from yt.analysis_modules.halo_finding.api import *
import os.path
import os, sys
import numpy
import math
import optparse
import operator

#################################################################################
# --------------------------------- M  A  I  N -------------------------------- #
#################################################################################

def main():
    # Database to look at, dummy needed at the end
    # (Requirement: the halo list of the last redshift in file 1 should be the same as the halo list of the first redshift in file 2, i.e. made by the same halofinder)
    haloDB_names = ["/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf036.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf048.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf060.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf072.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf084.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf096.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf108.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf120.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf132.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf144.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf156.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf168.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf180.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf192.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf204.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf216.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf228.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf240.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf252.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf264.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf276.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf288.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf300.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf312.txt",
                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_0.7_lev5_mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf320.txt",
                    "/Dummy/Dummy.txt"]
#     haloDB_names = ["/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf036.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf048.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf060.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf072.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf084.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf096.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf108.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf120.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf132.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf144.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf156.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf168.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf180.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf192.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf204.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf216.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf228.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf240.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf252.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf264.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf276.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf288.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf300.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf312.txt",
#                     "/Users/ji-hoonkim/Documents2/Analysis_Pictures_Notes/workspace_yt/running-080112_CHaRGe/0.7-lev5-mdfr=4/MergerTreeDB_080112_CHaRGe_hmtf320.txt",
#                     "/Dummy/Dummy.txt"]
    # haloDB_names = ["/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf30.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf76.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf90.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf100.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf110.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf120.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf130.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf140.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf150.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf160.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf170.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf180.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf190.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf200.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf210.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf220.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf230.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf240.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf250.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf260.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf270.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf280.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf290.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf300.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf310.txt",
    #                 "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/MergerTreeDB_080112_CHaRGe_hmtf320.txt",
    #                 "/Dummy/Dummy.txt"]
#    haloDB_names = ["/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf190.txt",
#                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf200.txt",
#                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf210.txt",
#                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf220.txt",
#                    "/Dummy/Dummy.txt"]
#    haloDB_names = ["/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf30.txt",
#                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf76.txt",
#                    "/lustre/ki/pfs/mornkr/080112_CHaRGe/ICS_DM_NP512_nozoom/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf90.txt",
#                    "/Dummy/Dummy.txt"]
#    haloDB_names = ["/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf190.txt",
#                    "/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf200.txt",
#                    "/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf210.txt",
#                    "/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo/MergerTreeDB_080112_CHaRGe_hmtf220.txt",
#                    "/Dummy/Dummy.txt"]
#    haloDB_names = ["/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf30.txt",
#                    "/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf76.txt",
#                    "/Users/ji-hoonkim/Documents/Codes/enzo-runs/080112_CHaRGe/ic.enzo_tip/test/MergerTreeDB_080112_CHaRGe_hmtf90.txt",
#                    "/Dummy/Dummy.txt"]

    # The starting redshift in each file
    haloDB_starting_redshifts = get_starting_redshifts_cyan(haloDB_names)
    # haloDB_starting_redshifts = [10.65762, 4.696225, 2.055899, 1.720717, 1.528641,
    #                              1.365004, 1.223393, 1.099233, 0.9891612, 0.8906462,
    #                              0.8017412, 0.7209297, 0.6470048, 0.5789968, 0.5161162,
    #                              0.4577136, 0.4032496, 0.3522722, 0.3043999, 0.2593082,
    #                              0.2167193, 0.1763938, 0.1381247, 0.1017315, 0.06705640,
    #                              0.03396061, 0.000000]
#    haloDB_starting_redshifts = [0.6470048, 0.5789968, 0.5161162, 0.4577136, 0.000000]
#    haloDB_starting_redshifts = [10.65762, 4.696225, 2.055899, 0.000000]

    # Pick a halo of interest
    initial = 0    # Initial snapshot in which you want to find the Lagrangian volume
    final = 320    # Final snapshot you have run your halo finder
    accuracy = 10    # Find Lagrangian volume only using every Nth of particles in the final redshift
    safety = 100    # Find Lagrangian volume only within a sphere of radius = safety*halo_MaxRad in the initial redshift
    initial_file = "./DD%04d/data%04d" % (initial, initial)
    final_file = "./DD%04d/data%04d" % (final, final)

#    halo_snapIDs = [9, 15, 107, 118, 350, 372]
#    halo_snapIDs = range(300,650,1)
#    halo_snapIDs = [20]
#    halo_snapIDs = [146, 147, 148, 149, 150]
#    halo_snapIDs = [414, 415, 417, 438, 439, 457]
    halo_snapIDs = [466, 468, 473, 497, 499, 503]
#    halo_snapIDs = [1150, 1151, 1152, 1153, 1154]
#    halo_snapIDs = [7052, 7053, 7054, 7055, 7056]
    halo_redshifts = [0.002321708]
#    halo_redshifts = [1.720717]
#    halo_redshifts = [1.950505]
#    halo_redshifts = [2.002110] # no longer exists!
#    halo_redshifts = [2.055896]
#    halo_redshifts = [4.243385]
#    halo_redshifts = [4.458208]
#    halo_redshifts = [4.696225]

    for halo_redshift in halo_redshifts:
        for halo_snapID in halo_snapIDs:
            my_halo = get_GlobalHaloID_cyan(halo_snapID, halo_redshift, haloDB_starting_redshifts, haloDB_names)
            print "Analyzing halo of GlobalHaloID %d in file %s..." % (my_halo[0], haloDB_names[my_halo[1]])

            # Basic functions
            my_parents = get_halo_parents_cyan(my_halo, 10, haloDB_starting_redshifts, haloDB_names)
            print "my_parents = ", my_parents
#            my_parent = get_direct_parent_cyan(my_halo, haloDB_starting_redshifts, haloDB_names)
#            print my_parent

            # Create a lineage
#            my_lineage = [[my_halo, 0.0, get_halo_info_cyan(my_halo, haloDB_names)[4], 1.0/(1.0+get_halo_info_cyan(my_halo, haloDB_names)[2])]] # GlobalHaloID_and_file, HaloMass, scale factor
#            major_merger_events = []
#            get_lineage_cyan(my_halo, my_lineage, major_merger_events, haloDB_starting_redshifts, haloDB_names)
#            print "Lineage found : scale factor   = ", numpy.array(zip(*my_lineage)[3][::-1])
#            print "Lineage found : mass evolution = ", numpy.array(zip(*my_lineage)[2][::-1])
#            print "major_merger_events = ", major_merger_events
#            draw_lineage_mass_cyan(my_lineage, major_merger_events, 'test/HaloEvolution_halo%d_z=%.6f.pdf' % (halo_snapID, halo_redshift))

            # Draw a lineage tree
#            LineageTreeDotOutput_cyan(my_halo, 'test/LineageTree_halo%d_z=%.6f.pdf' % (halo_snapID, halo_redshift), haloDB_starting_redshifts, haloDB_names)
#            MergerTreeDotOutput_cyan(my_halo, 'test/MergerTree_halo%d_z=%.6f.pdf' % (halo_snapID, halo_redshift), 0.9, 5, haloDB_starting_redshifts, haloDB_names)

            # Find a Lagrangian region
#            vLag_finder_cyan(my_halo, initial_file, final_file, accuracy, safety, haloDB_starting_redshifts, haloDB_names)

            # Draw a projection around the target halo
            draw_projection(my_halo, halo_snapID, final, final_file, haloDB_starting_redshifts, haloDB_names)





#################################################################################
# ----------------------------- F U N C T I O N S ----------------------------- #
#################################################################################

# Thanks to http://code.activestate.com/recipes/502260-parseline-break-a-text-line-into-formatted-regions/
def parseline(line, format):
    """\
    FORMAT
# GlobalHaloID   SnapCurrentTimeIdentifier SnapZ          SnapHaloID     HaloMass       NumPart        CenMassX       CenMassY       CenMassZ       BulkVelX       BulkVelY       BulkVelZ       MaxRad         ChildHaloID0   ChildHaloFrac0 ChildHaloID1   ChildHaloFrac1 ChildHaloID2   ChildHaloFrac2 ChildHaloID3   ChildHaloFrac3 ChildHaloID4   ChildHaloFrac4
#  1              1343432345                4.696225e+00   0              2.897968e+12   16746          3.025045e-01   7.283326e-01   8.406055e-01   5.296852e+06   4.779544e+06   1.807349e+05   9.156460e-03   7324           9.708587e-01   15790          0.000000e+00   13864          0.000000e+00   12764          0.000000e+00   12216          0.000000e+00
    ---
    Given a line (a string actually) and a short string telling how to format it, return a list of python objects that result.
    The format string maps words (as split by line.split()) into python code:
    x   ->    Nothing; skip this word
    s   ->    Return this word as a string
    i   ->    Return this word as an int
    d   ->    Return this word as an int
    f   ->    Return this word as a float
    Basic parsing of strings:
    >>> parseline('Hello, World','ss')
    ['Hello,', 'World']
    >>> parseline('C1   0.0  0.0 0.0','sfff')
    ['C1', 0.0, 0.0, 0.0]
    """
    xlat = {'x':None, 's':str, 'f':float, 'd':int, 'i':int}
    result = []
    words = line.split()
    for i in range(len(format)):
        f = format[i]
        trans = xlat.get(f)
        if trans: result.append(trans(words[i]))
    if len(result) == 0: return None
    if len(result) == 1: return result[0]
    return result

# Mimics get_GlobalHaloID() in merger_tree.py
def get_GlobalHaloID_cyan(haloID, redshift, haloDB_starting_redshifts, haloDB_names):
    count = 0
    if redshift > haloDB_starting_redshifts[0]:
        print 'WARNING: Something is wrong!  Check your redshift of a halo to be inspected.'
    for i in range(0, len(haloDB_starting_redshifts)):
        if redshift >= haloDB_starting_redshifts[i]:
            file_number_to_open = max(0, i - 1)    # when the halo exits in two files, choose the earlier one
            break
    fyle = open(haloDB_names[file_number_to_open], 'r')
    for line in fyle:
        if count != 0:
            line_parsed = parseline(line, 'ddfd')
            if line_parsed[2] == redshift and line_parsed[3] == haloID:
                break
        count += 1
    fyle.close()
    return [line_parsed[0], file_number_to_open]    # GlobalHaloID, file_number_to_open

# Get the number of halos found at that redshift
def get_number_of_halos_cyan(redshift, haloDB_starting_redshifts, haloDB_names):
    count = 0
    number_of_halos = 0
    if redshift > haloDB_starting_redshifts[0]:
        print 'WARNING: Something is wrong!  Check your redshift of a halo to be inspected.'
    for i in range(0, len(haloDB_starting_redshifts)):
        if redshift >= haloDB_starting_redshifts[i]:
            file_number_to_open = max(0, i - 1)
            break
    fyle = open(haloDB_names[file_number_to_open], 'r')
    for line in fyle:
        if count != 0:
            line_parsed = parseline(line, 'ddf')
            if line_parsed[2] == redshift:
                number_of_halos += 1
        count += 1
    fyle.close()
    return number_of_halos

# Get the array of distinct redshifts in the entire haloDB's
def get_distinct_redshifts_cyan(haloDB_names):
    redshifts = []
    for haloDB_name in haloDB_names[:-1]:
        count = 0
        fyle = open(haloDB_name, 'r')
        for line in fyle:
            if count != 0:
                line_parsed = parseline(line, 'ddf')
                if line_parsed[2] in redshifts:
                    continue
                else:
                    redshifts.append(line_parsed[2])
            count += 1
        fyle.close()
    print "redshifts included in haloDB's: ", redshifts
    return redshifts

# Get the array of first redshifts in the each of haloDB's, this will be used to delineate the haloDB
def get_starting_redshifts_cyan(haloDB_names):
    starting_redshifts = []
    for haloDB_name in haloDB_names[:-1]:
        count = 0
        fyle = open(haloDB_name, 'r')
        for line in fyle:
            if count != 0:
                line_parsed = parseline(line, 'ddf')
                if line_parsed[2] in starting_redshifts:
                    print "WARNING: Duplicative starting_redshifts in more than two haloDB's.  Something is terribly wrong!  The rest of the code will not work properly."
                    break
                else:
                    starting_redshifts.append(line_parsed[2])
                    break
            count += 1
        fyle.close()
    starting_redshifts.append(0.000000)    # for practical reasons
    print "starting_redshifts = ", starting_redshifts
    return starting_redshifts

# Mimics get_halo_info_cyan() in merger_tree.py
def get_halo_info_cyan(GlobalHaloID_and_file, haloDB_names):
    count = 0
    fyle = open(haloDB_names[GlobalHaloID_and_file[1]], 'r')
    for line in fyle:
        if count != 0:
            line_parsed = parseline(line, 'ddfdfdfffffffdfdfdfdfdf')
            if line_parsed[0] == GlobalHaloID_and_file[0]:
                break
        count += 1
    fyle.close()
    return parseline(line, 'ddfdfdfffffffdfdfdfdfdf')

# Mimics but DIFFERENT from get_halo_parents() in merger_tree.py, see below
def get_halo_parents_cyan(GlobalHaloID_and_file, max_parents, haloDB_starting_redshifts, haloDB_names):
    halo_mass_thres = 5.0e10
    parents = []
    find_halo_parents_in_the_prior_file = 0
    count = 0
#    print "check = ", GlobalHaloID_and_file
    fyle = open(haloDB_names[GlobalHaloID_and_file[1]], 'r')
    for line in fyle:
        if count != 0:
            line_parsed = parseline(line, 'ddfdfdfffffffdf')
            if line_parsed[13] == GlobalHaloID_and_file[0]:    # a halo is considered as a parent if the most fraction of mass is given to the specified child halo; DIFFERENT from merger_tree.py
                if line_parsed[2] >= haloDB_starting_redshifts[GlobalHaloID_and_file[1]]:    # if your parent is at the top of the file in this haloDB_names[], find your parent again in the prior file
                    parents.append([get_GlobalHaloID_cyan(line_parsed[3], line_parsed[2], haloDB_starting_redshifts, haloDB_names),
                                    line_parsed[14], line_parsed[4], 1.0 / (1.0 + line_parsed[2])])
                else:
                    parents.append([[line_parsed[0], GlobalHaloID_and_file[1]], line_parsed[14], line_parsed[4], 1.0 / (1.0 + line_parsed[2])])    # GlobalHaloID_and_file, ChildHaloFrac0, HaloMass, scale factor
            if len(parents) >= max_parents or (len(parents) >= 1 and parents[-1][2] < halo_mass_thres):    # the order in parents is from more massive to less massive, and we cut out the less massive ones
                parents.pop()
                break
        count += 1
    fyle.close()
    return parents

# Get up to two most contributing direct parents, mimicking get_direct_parent_cyan()
def get_two_direct_parents_cyan(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names):
    major_merger_ratio_thres = 3.0    #4.0
    parents = []
    parents = get_halo_parents_cyan(GlobalHaloID_and_file, 10, haloDB_starting_redshifts, haloDB_names)
    parents.sort(compare_parents)
    parents = filter(filter_out_low_contributors, parents)
#    print "check = ", parents
    if len(parents) >= 2:    # the second parent is listed only when the merger ratio is above major_merger_ratio_thres
        if parents[0][2] < major_merger_ratio_thres * parents[1][2]:
            return parents[:2]    # GlobalHaloID_and_file, ChildHaloFrac0, HaloMass, scale factor
    return parents[:1]

# Mimics get_direct_parent() in merger_tree.py
def get_direct_parent_cyan(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names):
    parents = []
    parents = get_halo_parents_cyan(GlobalHaloID_and_file, 10, haloDB_starting_redshifts, haloDB_names)
    parents.sort(compare_parents)
    parents = filter(filter_out_low_contributors, parents)
    return parents[:1]    # GlobalHaloID_and_file, ChildHaloFrac0, HaloMass, scale factor

# OLD WAY
def get_direct_parent_cyan_old(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names):
    ID_and_file = None
    mass = 0
    scale = 0
    hfrac = 0
    parents = get_halo_parents_cyan(GlobalHaloID_and_file, 10, haloDB_starting_redshifts, haloDB_names)
    for parent in parents:
        if parent[1] < 0.5: continue    # the most massive parent halo that contributed at least 50% of its mass
        if parent[2] > mass:
            ID_and_file = parent[0]
            hfrac = parent[1]
            mass = parent[2]
            scale = parent[3]
    return [ID_and_file, hfrac, mass, scale]    # GlobalHaloID_and_file, ChildHaloFrac0, HaloMass, scale factor

# Sort key on descending HaloMass, descending ChildHaloFrac0
def compare_parents(a, b):
    return cmp(b[2], a[2]) or cmp(b[1], a[1])

# Filter key on removing the low contributors (to be a parent halo, you at least have to contribute 50% of its mass to the child halo)
def filter_out_low_contributors(a):
    return a[1] > 0.5

# Trace back the direct_parent back in time
def get_lineage_cyan(GlobalHaloID_and_file, lineage, major_merger_events, haloDB_starting_redshifts, haloDB_names):
#    result = get_direct_parent_cyan(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names)
    result = get_two_direct_parents_cyan(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names)
    if len(result) >= 2:
        major_merger_events.append([result[0][3], result[0][2], result[0][2] / result[1][2]])    # scale factor, more masive halo mass, merger ratio
    print 'Finding lineage for GlobalHaloID_and_file ', GlobalHaloID_and_file, ', and the result is = ', result
    if len(result) == 0:
        print 'Lineage end found!'
        return 1
    else:
        lineage.append(result[0])    # append only the most massive direct_parent
#    if len(lineage) > 2: return 1 # loop breaker, not too far back in time
    get_lineage_cyan(result[0][0], lineage, major_merger_events, haloDB_starting_redshifts, haloDB_names)    # recursively append the result going back in time

# Draw a lineage on the plane of scale factor and mass (halos mass evolution)
def draw_lineage_mass_cyan(lineage, major_merger_events, outputfile_name):
    lines = plt.plot(numpy.array(zip(*lineage)[3]), numpy.array(zip(*lineage)[2]))
    l1 = lines
    plt.setp(l1, linewidth=2, color='blue', linestyle='-', marker='o')
    plt.semilogy()
    plt.xlim(0, 1)
    plt.ylim(1e10, 1e14)
    #ax = plt.gca()
    #ax.set_xlim(ax.get_xlim()[::-1]) # reversing the x-axis
    plt.xlabel("Scale Factor")
    plt.ylabel("Halo Mass (Msun)")
    plt.title('Scale Factor vs. Halo Mass')
    plt.grid(True)
#    major_merger_events = [[0.15154072975347657, 909055000000.0, 3.3890321301071005], [0.143121877438439, 388507100000.0, 1.0103509013283596]] # for a quick test
    for major_merger_event in major_merger_events:
        plt.annotate('%.1f : 1 merger' % major_merger_event[2], (major_merger_event[0] + 0.005, 0.95 * major_merger_event[1]),
                     xytext=(major_merger_event[0] + 0.08, 0.3 * major_merger_event[1]),
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1.2, headwidth=4.0, shrink=0.005),
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'),
                     horizontalalignment='left', verticalalignment='top')
    plt.savefig(outputfile_name)
    plt.clf()

# Mimics _MergerTreeDotOutput in merger_tree.py
def MergerTreeDotOutput_cyan(GlobalHaloID_and_file, dotfile_name, link_min, max_parents, haloDB_starting_redshifts, haloDB_names):
    my_graph = pydot.Dot(graph_type='digraph')
    my_subgs = {}
    GlobalHaloIDs_and_files = [GlobalHaloID_and_file]
    redshifts = get_distinct_redshifts_cyan(haloDB_names)
    for redshift in redshifts:    # set up subgraphs for distinct redshifts
        my_subgs[redshift] = pydot.Subgraph('', rank='same')
        my_graph.add_subgraph(my_subgs[redshift])
    add_nodes_cyan(GlobalHaloIDs_and_files, my_graph, my_subgs, haloDB_starting_redshifts, haloDB_names)    # add initial nodes
    while len(GlobalHaloIDs_and_files) > 0:
        print "Finding parents for %d children." % len(GlobalHaloIDs_and_files)
        GlobalHaloIDs_and_files = link_parents_cyan(GlobalHaloIDs_and_files, my_graph, link_min, max_parents, haloDB_starting_redshifts, haloDB_names)    # add edges and find parents for next nodes
        add_nodes_cyan(GlobalHaloIDs_and_files, my_graph, my_subgs, haloDB_starting_redshifts, haloDB_names)    # add next nodes, go back to recursive calculation
#        break # loop breaker, for a test
    write_dotfile_cyan(my_graph, dotfile_name)

# Make a lineage tree mimicking MergerTreeDotOutput,
# a reduced version of MergerTree that follows up to two most massive halos that contributed the most; the second parent is shown when the merger ratio is above major_merger_ratio_thres
def LineageTreeDotOutput_cyan(GlobalHaloID_and_file, dotfile_name, haloDB_starting_redshifts, haloDB_names):
    my_graph = pydot.Dot(graph_type='digraph')
    my_subgs = {}
    GlobalHaloIDs_and_files = [GlobalHaloID_and_file]
    redshifts = get_distinct_redshifts_cyan(haloDB_names)
    for redshift in redshifts:    # set up subgraphs for distinct redshifts
        my_subgs[redshift] = pydot.Subgraph('', rank='same')
        my_graph.add_subgraph(my_subgs[redshift])
    add_nodes_cyan(GlobalHaloIDs_and_files, my_graph, my_subgs, haloDB_starting_redshifts, haloDB_names)    # add initial nodes
    while len(GlobalHaloIDs_and_files) > 0:
        print "Finding parents for %d children." % len(GlobalHaloIDs_and_files)
        GlobalHaloIDs_and_files = link_lineage_cyan(GlobalHaloIDs_and_files, my_graph, haloDB_starting_redshifts, haloDB_names)    # add edges and find parents for next nodes
        add_nodes_cyan(GlobalHaloIDs_and_files, my_graph, my_subgs, haloDB_starting_redshifts, haloDB_names)    # add next nodes
        if len(GlobalHaloIDs_and_files) >= 2:
            GlobalHaloIDs_and_files = GlobalHaloIDs_and_files[:1]    # pass only the most massive halo to the next recursive level, to make a "lineage" tree
#        break # loop breaker, for a test
    write_dotfile_cyan(my_graph, dotfile_name)

# Mimics _add_nodes() in merger_tree.py
def add_nodes_cyan(GlobalHaloIDs_and_files, graph, subgs, haloDB_starting_redshifts, haloDB_names):
    for GlobalHaloID_and_file in GlobalHaloIDs_and_files:
        halo_info = get_halo_info_cyan(GlobalHaloID_and_file, haloDB_names)
        maxID = get_number_of_halos_cyan(halo_info[2], haloDB_starting_redshifts, haloDB_names)
        color_float = 1. - float(halo_info[3]) / (maxID + 1)
        print "Adding nodes for GlobalHaloID_and_file = ", GlobalHaloID_and_file
        graph.add_node(pydot.Node(get_unique_node_id_cyan(GlobalHaloID_and_file),
                                  label="{%1.3e\\n(%1.3f,%1.3f,%1.3f)}" % \
                                      (halo_info[4], halo_info[6], halo_info[7], halo_info[8]),
                                  shape="record",
                                  color="%0.3f 1. %0.3f" % (color_float, color_float)))
        subgs[halo_info[2]].add_node(pydot.Node(get_unique_node_id_cyan(GlobalHaloID_and_file)))
        if len(subgs[halo_info[2]].get_node_list()) == 1:    # add redshift box
            subgs[halo_info[2]].add_node(pydot.Node("%1.5f" % halo_info[2],
                                                    label="%1.5f" % halo_info[2],
                                                    shape="record", color="green"))

# Mimics _find_parents() in merger_tree.py
def link_parents_cyan(GlobalHaloIDs_and_files, graph, link_min, max_parents, haloDB_starting_redshifts, haloDB_names):
    new_GlobalHaloIDs_and_files = []
    for GlobalHaloID_and_file in GlobalHaloIDs_and_files:
        parents_temp = get_halo_parents_cyan(GlobalHaloID_and_file, max_parents, haloDB_starting_redshifts, haloDB_names)
#        print "check 1 =", parents_temp
        for parent_temp in parents_temp:
            if parent_temp[1] <= link_min:    # draw only important contributors, parent_temp format = [GlobalHaloID_and_file, ChildHaloFrac0, HaloMass, scale factor]
                continue
            else:
                print "Adding edges for GlobalHaloID_and_file = ", GlobalHaloID_and_file
#                print "check 2 =", parent_temp
                graph.add_edge(pydot.Edge(get_unique_node_id_cyan(parent_temp[0]), get_unique_node_id_cyan(GlobalHaloID_and_file),
                                          label="%3.2f%% " % float(parent_temp[1] * 100),
                                          color="blue",
                                          fontsize="10"))
                new_GlobalHaloIDs_and_files.append(parent_temp[0])
    return new_GlobalHaloIDs_and_files

# Link a lineage mimicking link_parents_cyan()
def link_lineage_cyan(GlobalHaloIDs_and_files, graph, haloDB_starting_redshifts, haloDB_names):
    new_GlobalHaloIDs_and_files = []
    for GlobalHaloID_and_file in GlobalHaloIDs_and_files:
        two_direct_parents_temp = get_two_direct_parents_cyan(GlobalHaloID_and_file, haloDB_starting_redshifts, haloDB_names)
#        print "check 1 =", two_direct_parents_temp
        if len(two_direct_parents_temp) != 0:
            for two_direct_parent_temp in two_direct_parents_temp:
                print "Adding edges for GlobalHaloID_and_file = ", GlobalHaloID_and_file
#                print "check 2 =", two_direct_parent_temp
                if len(two_direct_parents_temp) == 1:
                    graph.add_edge(pydot.Edge(get_unique_node_id_cyan(two_direct_parent_temp[0]), get_unique_node_id_cyan(GlobalHaloID_and_file),
                                              label="%3.2f%%" % float(two_direct_parent_temp[1] * 100),
                                              color="blue",
                                              fontsize="10"))
                if len(two_direct_parents_temp) == 2:
                    graph.add_edge(pydot.Edge(get_unique_node_id_cyan(two_direct_parent_temp[0]), get_unique_node_id_cyan(GlobalHaloID_and_file),
                                              label="%3.2f%%" % float(two_direct_parent_temp[1] * 100),
                                              color="red",
                                              fontsize="10"))
                new_GlobalHaloIDs_and_files.append(two_direct_parent_temp[0])
    return new_GlobalHaloIDs_and_files

# Get the unique node ID
def get_unique_node_id_cyan(GlobalHaloID_and_file):
#    print "check = ", GlobalHaloID_and_file
    return str(GlobalHaloID_and_file[0] + GlobalHaloID_and_file[1] * 1e7)

# Mimics _write_dotfile in merger_tree.py
def write_dotfile_cyan(graph, dotfile_name):
    suffix = dotfile_name.split(".")[-1]
    if suffix == "gv": suffix = "raw"
    print "Writing %s format %s to disk." % (suffix, dotfile_name)
    graph.write("%s" % dotfile_name, format=suffix)

# Find the Lagrangian region in the initial DD0000 for a specified halo at DD####
def vLag_finder_cyan(GlobalHaloID_and_file, initial_file, final_file, accuracy, safety, haloDB_starting_redshifts, haloDB_names):
    # Halo information
    halo_info = get_halo_info_cyan(GlobalHaloID_and_file, haloDB_names)
    halo_CenMass = [halo_info[6], halo_info[7], halo_info[8]]
    halo_BulkVel = [halo_info[9] / 1e5, halo_info[10] / 1e5, halo_info[11] / 1e5]    # km/s
    halo_MaxRad = halo_info[12]
    # Final Snapshot
    a = EnzoStaticOutput(final_file)
#    print "check 1 = ", final_file, a, halo_CenMass, halo_MaxRad
    if abs(a.current_redshift - halo_info[2]) < 0.0001:
        print "Final redshift on which you have run the halo finder = ", a.current_redshift
    else:
        print "The redshifts do not match!  Check your parameters..."
        return 1
    my_sphere = a.h.sphere(halo_CenMass, 2 * halo_MaxRad)
    DM_particles = my_sphere["particle_type"] == 1
    particle_indices_in_Lagrangian_volume = my_sphere["particle_index"][DM_particles]
    print "The number of particles inside 2*halo_MaxRad = %f is %d" % (2 * halo_MaxRad, len(particle_indices_in_Lagrangian_volume))    # this typically has much more particles than halo_NumPart
    # Initial Snapshot
    a0 = EnzoStaticOutput(initial_file)
#    print "check 2 = ", a0, halo_CenMass, safety*halo_MaxRad
    print "Initial redshift in which you want to find the Lagrangian volume  = ", a0.current_redshift
    my_sphere0 = a0.h.sphere(halo_CenMass, safety * halo_MaxRad)
    # Now find where the particles are initially
    print "Now looking for %d particles in the sphere of %f in the initial snapshot..." % (int(len(particle_indices_in_Lagrangian_volume) / accuracy), safety * halo_MaxRad)
    particles_in_Lagrangian_volume = my_sphere0["particle_index"] == particle_indices_in_Lagrangian_volume[0]
    for i in range(1, len(particle_indices_in_Lagrangian_volume), accuracy):
	particles_in_Lagrangian_volume += my_sphere0["particle_index"] == particle_indices_in_Lagrangian_volume[i]
#	print "i: index = ", i, particle_indices_in_Lagrangian_volume[i]
    # Now find min and max of initial particle_position_[xyz] in Lagrangian volume
    print "Now finding the locations in the initial snapshot..."
#    print "check = ", my_sphere0["particle_position_x"][particles_in_Lagrangian_volume]
    xmin = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume].min()
    xmax = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume].max()
    ymin = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume].min()
    ymax = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume].max()
    zmin = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume].min()
    zmax = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume].max()
    # Caution that particles may have come from beyond the boundary (thanks to periodic boundary condition),
    if halo_CenMass[0] + safety * halo_MaxRad > 1.0:
	particles_came_from_beyond_plus_x = halo_CenMass[0] - my_sphere0["particle_position_x"][particles_in_Lagrangian_volume] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_x"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_x]) > 0:
            xmax = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_x].max() + 1.0
            xmin = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume][~particles_came_from_beyond_plus_x].min()
    if halo_CenMass[1] + safety * halo_MaxRad > 1.0:
	particles_came_from_beyond_plus_y = halo_CenMass[1] - my_sphere0["particle_position_y"][particles_in_Lagrangian_volume] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_y]) > 0:
            ymax = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_y].max() + 1.0
            ymin = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][~particles_came_from_beyond_plus_y].min()
    if halo_CenMass[2] + safety * halo_MaxRad > 1.0:
	particles_came_from_beyond_plus_z = halo_CenMass[2] - my_sphere0["particle_position_z"][particles_in_Lagrangian_volume] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_z]) > 0:
            zmax = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][particles_came_from_beyond_plus_z].max() + 1.0
            zmin = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][~particles_came_from_beyond_plus_z].min()
    if halo_CenMass[0] - safety * halo_MaxRad < 0.0:
	particles_came_from_beyond_minus_x = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume] - halo_CenMass[0] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_x]) > 0:
            xmin = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_x].min() - 1.0
            xmax = my_sphere0["particle_position_x"][particles_in_Lagrangian_volume][~particles_came_from_beyond_minus_x].max()
    if halo_CenMass[1] - safety * halo_MaxRad < 0.0:
	particles_came_from_beyond_minus_y = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume] - halo_CenMass[1] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_y]) > 0:
            ymin = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_y].min() - 1.0
            ymax = my_sphere0["particle_position_y"][particles_in_Lagrangian_volume][~particles_came_from_beyond_minus_y].max()
    if halo_CenMass[2] - safety * halo_MaxRad < 0.0:
	particles_came_from_beyond_minus_z = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume] - halo_CenMass[2] > 1.0 - 0.99 * safety * halo_MaxRad
	if len(my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_z]) > 0:
            zmin = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][particles_came_from_beyond_minus_z].min() - 1.0
            zmax = my_sphere0["particle_position_z"][particles_in_Lagrangian_volume][~particles_came_from_beyond_minus_z].max()
    # Finally dump the results
    print "The halo position at z=%f is (%f, %f, %f) moving at (%f, %f, %f) km/s" % (a.current_redshift, halo_CenMass[0], halo_CenMass[1], halo_CenMass[2],
                                                                                     halo_BulkVel[0], halo_BulkVel[1], halo_BulkVel[2])
    print "The ranges of the Lagrangian volume initially at z=%f is [x,y,z] = [[%f, %f], [%f, %f], [%f, %f]]" % (a0.current_redshift, xmin, xmax, ymin, ymax, zmin, zmax)
    print "Done!"

# Draw a particle_density projection around the target halo
def draw_projection(GlobalHaloID_and_file, halo_snapID, final, final_file, haloDB_starting_redshifts, haloDB_names):
    # Halo information
    halo_info = get_halo_info_cyan(GlobalHaloID_and_file, haloDB_names)
    halo_CenMass = [halo_info[6], halo_info[7], halo_info[8]]
    halo_MaxRad = halo_info[12]
    # Final Snapshot
    pf2 = load(final_file)
    center2 = halo_CenMass
    width = 0.1
    proj_axis = 1
    region2 = pf2.h.region(center2, [center2[0] - 0.5 * width, center2[1] - 0.5 * width, center2[2] - 0.5 * width],
                           [center2[0] + 0.5 * width, center2[1] + 0.5 * width, center2[2] + 0.5 * width])
    print region2
    if ((center2[0] - 0.5 * width) < pf2.parameters['RefineRegionLeftEdge'][0] < (center2[0] + 0.5 * width)) or ((center2[1] - 0.5 * width) < pf2.parameters['RefineRegionLeftEdge'][1] < (center2[1] + 0.5 * width)) or ((center2[2] - 0.5 * width) < pf2.parameters['RefineRegionLeftEdge'][2] < (center2[2] + 0.5 * width)):
        return 1

    pc2 = PlotCollection(pf2, center2)
    p2 = pc2.add_projection("particle_density", proj_axis, data_source=region2)
#    p2 = pc2.add_projection("particle_density", proj_axis, data_source=region2, weight_field="particle_density")
    p2.set_zlim(1e-6, 1e-1)
#    p2.set_zlim(1e-25, 1e-32)
    p2.set_width(width, '1')

    if os.path.exists(final_file):
        halos = LoadHaloes(pf2, "./DD%04d/MergerHalos" % final)
        for j in reversed(range(len(halos))):
            if ((center2[proj_axis] - 0.5 * width) < halos[j].center_of_mass()[proj_axis] < (center2[proj_axis] + 0.5 * width)):
                continue
            else:
                halos._groups.remove(halos[j])    # by removing halo entries from backwards, we don't disturb the halo index later
        p2.modify["hop_circles"](halos, max_number=1000, min_size=0, annotate=True)
	p2.modify["text"]([0.82, 0.1], "z=%.2f" % pf2.current_redshift, text_args={'color':'r', 'fontsize':'x-large'})
	p2.modify["sphere"](center2, halo_MaxRad,
			    circle_args={'color':'r', 'linestyle':'dashed', 'linewidth':2.0, 'alpha':0.6, 'fill':False},
			    text=None, text_args={'color':'r', 'fontsize':'x-large'})    # to use this, slightly modified /a/sulky38/g.ki.ki12/mornkr/yt-x86_64/src/yt-hg/yt/visualization/plot_modifications.py
	pc2.save("./test/path_halo%d_w=%.2f" % (halo_snapID, width))

#    p2 = pc2.add_projection("particle_density", 1, data_source=region2)
#    p2.set_zlim(1e-6, 1e-1)
#    p2.set_width(width, '1')








#################################################################################
# --------------------------- Idiomatic  M  A  I  N --------------------------- #
#################################################################################

# This is only to have main() above all the other functions()
if __name__ == "__main__":
    main()



