'''
Generate initial condition files for MUSIC.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf, int32, float32
# local ----
import yt.mods    #@UnresolvedImport
import yt.analysis_modules.halo_finding.api as halo_io    #@UnresolvedImport
#import yt.analysis_modules.halo_merger_tree.api as tree_io    #@UnresolvedImport
from utilities import utility as ut

AGORA_DIRECTORY = '/Users/awetzel/work/research/simulation/AGORA/18hMpc_level-10/'
SNAPSHOT_FINAL_DIRECTORY = AGORA_DIRECTORY + 'DD0320/'
SNAPSHOT_INITIAL_DIRECTORY = AGORA_DIRECTORY + 'DD0000/'


# relatively isolated halos with M_vir ~ 2e11 M_sun
# AGORA uses 473
halo_ids = np.array([414, 415, 417, 438, 439, 457, 466, 468, 473, 497, 499, 503])

# pc = yt.mods.SlicePlot(pf, 'y', ('deposit', 'all_density'),
# center=[0.72157766, 0.5858333, 0.65605193], width=0.002160379324*6)


class InitialConditionClass(ut.io.SayClass):
    '''
    '''
    def __init__(self):
        self.snapshot = []
        self.snapshot.append(yt.mods.load(SNAPSHOT_FINAL_DIRECTORY + 'data0320'))
        self.snapshot.append(yt.mods.load(SNAPSHOT_INITIAL_DIRECTORY + 'data0000'))

        self.cube = []
        self.cube.append(self.snapshot[0].h.all_data())
        self.cube.append(self.snapshot[1].h.all_data())

        # physical
        self.cube_length_kpc = np.float32(self.snapshot[0].length_unit.in_units('kpc'))
        self.cube_length_Mpc = np.float32(self.snapshot[0].length_unit.in_units('Mpc'))

        self.read_halo_catalog()

    def read_halo_catalog(self):
        '''
        .
        '''
        self.halo = halo_io.LoadHaloes(self.snapshot[-1], SNAPSHOT_FINAL_DIRECTORY + 'MergerHalos')

        self.hal = ut.array.DictClass()
        self.hal.snap = {}
        self.hal.snap['scale.factor'] = 1 / (1 + self.snapshot[0].current_redshift)
        self.hal.info = {}
        self.hal.info['box.length'] = self.cube_length_Mpc / self.hal.snap['scale.factor']

        self.hal['position'] = np.zeros((len(self.halo), 3), dtype=float32)    # Mpc comoving
        self.hal['mass'] = np.zeros(len(self.halo), dtype=float32)    # log M_sun
        self.hal['radius'] = np.zeros(len(self.halo), dtype=float32)    # Mpc comoving
        self.hal['particle.num'] = np.zeros(len(self.halo), dtype=int32)

        for hi in xrange(len(self.halo)):
            self.hal['mass'][hi] = self.halo[hi].total_mass()
            self.hal['radius'][hi] = self.halo[hi].maximum_radius()
            self.hal['position'][hi] = self.halo[hi].center_of_mass()
            self.hal['particle.num'][hi] = self.halo[hi].get_size()
        self.hal['mass'] = log10(self.hal['mass'])
        self.hal['radius'] *= self.hal.info['box.length']
        self.hal['position'] *= self.hal.info['box.length']

        NearestNeig = ut.catalog.NearestNeigClass()
        NearestNeig.assign(self.hal, 'mass', [1, Inf], [1, Inf], 200, 8, 'comoving', 'virial')
        NearestNeig.assign_to_catalog(self.hal)

    def read_particles(self):
        '''
        '''
        dimen_names = ['x', 'y', 'z']
        zis = [1, 0]

        self.part = []
        for _ in zis:
            self.part.append({'id': [], 'id-to-index': [], 'mass': [], 'position': []})

        for zi in zis:
            self.part[zi]['id'] = np.array(self.cube[zi]['particle_index'], dtype=int32)
            if np.unique(self.part[zi]['id']).size != self.part[zi]['id'].size:
                raise ValueError('partice IDs not unique')

            ut.catalog.assign_id_to_index(self.part[zi], 'id', 0)

            self.part[zi]['mass'] = np.array(self.cube[zi]['particle_mass'].in_units('Msun'),
                                             dtype=float32)

            self.part[zi]['position'] = np.zeros(
                (self.part[zi]['id'].size, len(dimen_names)), dtype=np.float32)
            for dimen_i, dimen_name in enumerate(dimen_names):
                self.part[zi]['position'][:, dimen_i] = \
                    self.cube[zi]['particle_position_' + dimen_name]

    def get_particle_ids(self, zi=0):
        '''
        .
        '''
        return np.array(self.cube[zi]['particle_index'], dtype=int32)

    def get_particles_around_halo(self, hi, rad_max, scale_vir=False):
        '''
        rad_max in kpc or virial radius units.
        '''
        # convert rad_max to simulation units [0, 1)
        if scale_vir:
            rad_max *= self.halo[hi].maximum_radius()
        else:
            rad_max /= self.cube_length_kpc
        sp = self.snapshot[0].h.sphere(self.halo[hi].center_of_mass(), rad_max)

        return np.array(sp['particle_index'], dtype=int32)

    def get_halos_around_halo(self, hi, dist_max):
        '''
        '''
        Neighbor = ut.neighbor.NeighborClass()

        dist_max *= self.hal['radius'][hi]
        mass_min = self.hal['mass'] - log10(2)
        his_m = ut.array.elements(self.hal['mass'], [mass_min, Inf])
        distances, his_neig = Neighbor.get_neighbors(
            self.hal['position'][[hi]], self.hal['position'][his_m], 200, [1e-6, dist_max],
            self.hal.info['box.length'], neig_ids=his_m)
        distances /= self.hal['radius'][hi]

        return distances, his_neig


class InitialConditionTestClass(ut.io.SayClass):
    '''
    '''
    def __init__(self):
        self.snapshot = []
        self.snapshot.append(yt.mods.load(SNAPSHOT_FINAL_DIRECTORY + 'data0320'))
        self.snapshot.append(yt.mods.load(SNAPSHOT_INITIAL_DIRECTORY + 'data0000'))

        self.cube = []
        self.cube.append(self.snapshot[0].h.all_data())
        self.cube.append(self.snapshot[1].h.all_data())

        # physical
        self.cube_length_kpc = np.float32(self.snapshot[0].length_unit.in_units('kpc'))
        self.cube_length_Mpc = np.float32(self.snapshot[0].length_unit.in_units('Mpc'))

        self.read_halo_catalog()

    def read_halo_catalog(self):
        '''
        .
        '''
        self.halo = halo_io.LoadHaloes(self.snapshot[-1], SNAPSHOT_FINAL_DIRECTORY + 'MergerHalos')

        self.hal = ut.array.DictClass()
        self.hal.snap = {}
        self.hal.snap['scale.factor'] = 1 / (1 + self.snapshot[0].current_redshift)
        self.hal.info = {}
        self.hal.info['box.length'] = self.cube_length_Mpc / self.hal.snap['scale.factor']

        self.hal['position'] = np.zeros((len(self.halo), 3), dtype=float32)    # Mpc comoving
        self.hal['mass'] = np.zeros(len(self.halo), dtype=float32)    # log M_sun
        self.hal['radius'] = np.zeros(len(self.halo), dtype=float32)    # Mpc comoving
        self.hal['particle.num'] = np.zeros(len(self.halo), dtype=int32)

        for hi in xrange(len(self.halo)):
            self.hal['mass'][hi] = self.halo[hi].total_mass()
            self.hal['radius'][hi] = self.halo[hi].maximum_radius()
            self.hal['position'][hi] = self.halo[hi].center_of_mass()
            self.hal['particle.num'][hi] = self.halo[hi].get_size()
        self.hal['mass'] = log10(self.hal['mass'])
        self.hal['radius'] *= self.hal.info['box.length']
        self.hal['position'] *= self.hal.info['box.length']

        NearestNeig = ut.catalog.NearestNeigClass()
        NearestNeig.assign(self.hal, 'mass', [1, Inf], [1, Inf], 200, 8, 'comoving', 'virial')
        NearestNeig.assign_to_catalog(self.hal)

    def read_particles(self):
        '''
        '''
        dimen_names = ['x', 'y', 'z']
        zis = [1, 0]

        self.part = []
        for _ in zis:
            self.part.append({'id': [], 'id-to-index': [], 'mass': [], 'position': []})

        for zi in zis:
            self.part[zi]['id'] = np.array(self.cube[zi]['particle_index'], dtype=int32)
            if np.unique(self.part[zi]['id']).size != self.part[zi]['id'].size:
                raise ValueError('partice IDs not unique')

            ut.catalog.assign_id_to_index(self.part[zi], 'id', 0)

            self.part[zi]['mass'] = np.array(self.cube[zi]['particle_mass'].in_units('Msun'),
                                             dtype=float32)

            self.part[zi]['position'] = np.zeros(
                (self.part[zi]['id'].size, len(dimen_names)), dtype=np.float32)
            for dimen_i, dimen_name in enumerate(dimen_names):
                self.part[zi]['position'][:, dimen_i] = \
                    self.cube[zi]['particle_position_' + dimen_name]

    def get_particle_ids(self, zi=0):
        '''
        .
        '''
        return np.array(self.cube[zi]['particle_index'], dtype=int32)

    def get_particles_around_halo(self, hi, rad_max, scale_vir=False):
        '''
        rad_max in kpc or virial radius units.
        '''
        # convert rad_max to simulation units [0, 1)
        if scale_vir:
            rad_max *= self.halo[hi].maximum_radius()
        else:
            rad_max /= self.cube_length_kpc
        sp = self.snapshot[0].h.sphere(self.halo[hi].center_of_mass(), rad_max)

        return np.array(sp['particle_index'], dtype=int32)

    def get_halos_around_halo(self, hi, dist_max):
        '''
        '''
        Neighbor = ut.neighbor.NeighborClass()

        dist_max *= self.hal['radius'][hi]
        mass_min = self.hal['mass'] - log10(2)
        his_m = ut.array.elements(self.hal['mass'], [mass_min, Inf])
        distances, his_neig = Neighbor.get_neighbors(
            self.hal['position'][[hi]], self.hal['position'][his_m], 200, [1e-6, dist_max],
            self.hal.info['box.length'], neig_ids=his_m)
        distances /= self.hal['radius'][hi]

        return distances, his_neig