'''
Generate initial conditions for MUSIC using AGORA.

Halo masses in log {M_sun}, particle masses in {M_sun}, positions in {kpc comoving}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf, int32, float32
# local ----
import yt    #@UnresolvedImport
import yt.analysis_modules.halo_finding.api as halo_io    #@UnresolvedImport
import yt.analysis_modules.halo_analysis.api as halo_analysis    #@UnresolvedImport
#import yt.analysis_modules.halo_merger_tree.api as tree_io    #@UnresolvedImport
from utilities import utility as ut

# relatively isolated halos with M_vir ~ 2e11 M_sun
# AGORA uses 473
halo_ids = np.array([414, 415, 417, 438, 439, 457, 466, 468, 473, 497, 499, 503])

# pc = yt.SlicePlot(pf, 'y', ('deposit', 'all_density'),
# center=[0.72157766, 0.5858333, 0.65605193], width=0.002160379324*6)


class AgoraClass(ut.array.DictClass, ut.io.SayClass):
    '''
    Read in particles and halos from AGORA snapshots at z_initial and z = 0.
    Use for generating initial conditions.
    '''
    def __init__(
        self, agora_dir='/Users/awetzel/work/research/simulation/agora/refinement-09_60Mpc/',
        snapshot_final_dir='DD0320/', snapshot_initial_dir='DD0000/', read_halo=True):
        '''
        Parameters
        ----------
        directory of AGORA simulation: string
        directory of final shapshot: string
        directory of initial snapshot: string
        whether to read halo catalog: boolean
        '''
        self.agora_directory = agora_dir
        self.snapshot_final_directory = self.agora_directory + snapshot_final_dir
        self.snapshot_initial_directory = self.agora_directory + snapshot_initial_dir

        self.snapshot = []
        self.snapshot.append(yt.load(self.snapshot_final_directory + 'data0320'))
        self.snapshot.append(yt.load(self.snapshot_initial_directory + 'data0000'))

        self['box.length.phys'] = np.float32(self.snapshot[0].length_unit.in_units('kpc'))
        self['box.length'] = self['box.length.phys'] * (1 + self.snapshot[0].current_redshift)
        self['redshifts'] = np.array([self.snapshot[0].current_redshift,
                                      self.snapshot[1].current_redshift])

        self.data = []
        self.data.append(self.snapshot[0].h.all_data())
        self.data.append(self.snapshot[1].h.all_data())

        if read_halo:
            self.read_halo_catalog()

    def read_halo_catalog(self):
        '''
        Read HOP halo catalog, save as dictionary to self.

        first need to run finder via:
        hc = halo_analysis.HaloCatalog(data_ds=self.snapshot[0], finder_method='hop',
                                       output_dir='halo_catalog')
        hc.create()
        '''
        self.halo = halo_io.LoadHaloes(self.snapshot[0],
                                       self.snapshot_final_directory + 'MergerHalos')
        #self.halo = halo_analysis.HaloCatalog(data_ds=self.snapshot[0],
        #                                      self.snapshot_final_directory + 'MergerHalos')

        self.hal = ut.array.DictClass()
        self.hal.snap = {}
        self.hal.snap['scale.factor'] = 1 / (1 + self['redshifts'][0])
        self.hal.info = {}
        self.hal.info['box.length'] = self['box.length']

        self.hal['position'] = np.zeros((len(self.halo), 3), dtype=float32)
        self.hal['mass'] = np.zeros(len(self.halo), dtype=float32)
        self.hal['radius'] = np.zeros(len(self.halo), dtype=float32)
        self.hal['particle.num'] = np.zeros(len(self.halo), dtype=int32)

        for hi in xrange(len(self.halo)):
            self.hal['mass'][hi] = self.halo[hi].total_mass()
            self.hal['radius'][hi] = self.halo[hi].maximum_radius()
            self.hal['position'][hi] = self.halo[hi].center_of_mass()
            self.hal['particle.num'][hi] = self.halo[hi].get_size()
        self.hal['mass'] = log10(self.hal['mass'])    # log {M_sun}
        self.hal['radius'] *= self.hal.info['box.length']    # {kpc comoving}
        self.hal['position'] *= self.hal.info['box.length']    # {kpc comoving}

        NearestNeig = ut.catalog.NearestNeigClass()
        NearestNeig.assign(self.hal, 'mass', [1, Inf], [1, Inf], 200, 8, 'comoving', 'virial')
        NearestNeig.assign_to_catalog(self.hal)

    def read_particles(self, zis=[1, 0]):
        '''
        Read particles, save as dictionary to self.

        Parameters
        ----------
        snapshot indices to read: int or list
        '''
        dimen_names = ['x', 'y', 'z']
        if np.isscalar(zis):
            zis = [zis]

        self.part = ut.array.ListClass()
        for _ in zis:
            part_z = ut.array.DictClass()
            part_z['id'] = []
            part_z['id-to-index'] = []
            part_z['mass'] = []
            part_z['position'] = []
            part_z.info = {}
            part_z.info['box.length'] = self['box.length']
            self.part.append(part_z)
        self.part.info = {}
        self.part.info['box.length'] = self['box.length']

        for zi in zis:
            # in yt, particle_index = id
            self.part[zi]['id'] = np.array(self.data[zi]['particle_index'], dtype=int32)
            if np.unique(self.part[zi]['id']).size != self.part[zi]['id'].size:
                raise ValueError('partice ids are not unique')
            del(self.data[zi]['particle_index'])

            ut.catalog.assign_id_to_index(self.part[zi], 'id', 0)

            self.part[zi]['mass'] = np.array(self.data[zi]['particle_mass'].in_units('Msun'),
                                             dtype=float32)    # {M_sun}
            del(self.data[zi]['particle_mass'])

            self.part[zi]['position'] = np.zeros(
                (self.part[zi]['id'].size, len(dimen_names)), dtype=np.float32)
            for dimen_i, dimen_name in enumerate(dimen_names):
                # {kpc comoving}
                self.part[zi]['position'][:, dimen_i] = \
                    self.data[zi]['particle_position_' + dimen_name] * self.hal.info['box.length']
                del(self.data[zi]['particle_position_' + dimen_name])

    def get_particles_around_halo(self, halo_index, distance_max, scale_vir=True):
        '''
        Get ids of particles that are within distance_max of center of given halo.

        Parameters
        ----------
        halo index: int
        maximum distance {kpc comoving or in units of virial radius}: float
        whether to scale distance_max by virial radius
        '''
        if scale_vir:
            distance_max *= self.hal['radius'][halo_index]
        # convert distance_max to simulation units [0, 1)
        distance_max /= self['box.length']
        sp = self.snapshot[0].h.sphere(self.halo[halo_index].center_of_mass(), distance_max)

        return np.array(sp['particle_index'], dtype=int32)

    def get_halos_around_halo(self, halo_index, distance_max, neig_mass_frac_min=0.5):
        '''
        Get neig_distances & indices of halos that are within distance_max of center of given halo.

        Parameters
        ----------
        halo index: int
        maximum distance {kpc physical}: float
        minimum fraction of input mass to keep neighboring halos: float
        '''
        Neighbor = ut.neighbor.NeighborClass()

        distance_max *= self.hal['radius'][halo_index]
        mass_min = self.hal['mass'] + log10(neig_mass_frac_min)
        his_m = ut.array.elements(self.hal['mass'], [mass_min, Inf])
        neig_distances, neig_indices = Neighbor.get_neighbors(
            self.hal['position'][[halo_index]], self.hal['position'][his_m], 200,
            [1e-6, distance_max], self.hal.info['box.length'], neig_ids=his_m)
        neig_distances /= self.hal['radius'][halo_index]

        return neig_distances, neig_indices

    def print_ic_zoom_region_for_halo(
        self, halo_index, refinement_num=1, distance_max=None, geometry='cube'):
        '''
        Print extent of lagrangian region at z_initial around given halo at z = 0.
        Use rules of thumb from Onorbe et al.

        Parameters
        ----------
        halo index: int
        number of refinement levels beyond current level for zoom-in region: int
        maximum distance want to be uncontaminated {kpc comoving}: float
            if None, use R_vir
        geometry of zoom-in lagrangian regon in initial conditions: string
            options: cube, ellipsoid
        '''
        if not distance_max:
            distance_max = self.hal['radius'][halo_index] * 1.2

        if geometry == 'cube':
            distance_max = (1.5 * refinement_num + 1) * distance_max
        elif geometry == 'ellipsoid':
            distance_max = (1.5 * refinement_num + 7) * distance_max

        pids = self.get_particles_around_halo(halo_index, distance_max, scale_vir=False)
        pis = self.part[1]['id-to-index'][pids]
        poss = self.part[1]['position'][pis]
        lims = np.zeros((poss.shape[1], 2))
        wids = np.zeros(poss.shape[1])
        for dimen_i in xrange(poss.shape[1]):
            lims[dimen_i] = np.array(ut.array.get_limits(poss[:, dimen_i]))
            wids[dimen_i] = lims[[dimen_i]].max() - lims[[dimen_i]].min()
            self.say('dimension-%d: %s (%.3f) kpc, %s (%.8f) box length' %
                     (dimen_i, ut.array.get_limits(lims[[dimen_i]], digit_num=3), wids[dimen_i],
                      ut.array.get_limits(lims[[dimen_i]] / self['box.length'], digit_num=8),
                      wids[dimen_i] / self['box.length']))
        lims /= self['box.length']
        wids /= self['box.length']
        self.say('for MUSIC config file:')
        self.say('  ref_offset = %.8f, %.8f, %.8f' % (lims[0, 0], lims[1, 0], lims[2, 0]))
        self.say('  ref_extent = %.8f, %.8f, %.8f' % (wids[0], wids[1], wids[2]))


class TestClass(ut.io.SayClass):
    '''
    '''
    def __init__(self):
        pass

    def print_particle_contamination(self, part, cen_position=None, distance_lim=None,
                                     distance_bin_num=20, scaling='lin', geometry='cube'):
        '''
        Test lower resolution particle contamination around center.

        Parameters
        ----------
        particle catalog: dict
        3-d position of center: array
        maximum distance from center to check: float
        region geometry: string
            options: cube, sphere
        '''
        if distance_lim is None:
            distance_lim = [0, 0.5 * (1 - 1e-5) * part.info['box.length']]

        DistBin = ut.bin.DistanceBinClass(scaling, distance_lim, distance_bin_num)

        masses_unique = np.unique(part['mass'])
        pis_all = ut.array.arange_length(part['mass'])
        pis_hires = pis_all[part['mass'] == masses_unique.min()]
        pis_contam = pis_all[part['mass'] != masses_unique.min()]

        if cen_position is None:
            cen_position = np.zeros(part['position'].shape[1])
            for dimen_i in xrange(part['position'].shape[1]):
                cen_position[dimen_i] = np.median(part['position'][pis_hires, dimen_i])

        print('center position = %s' % cen_position)

        if geometry == 'sphere':
            distances, neig_pis = ut.neighbor.Neighbor.get_neighbors(
                cen_position, part['position'], part['mass'].size,
                distance_lim, part.info['box.length'], neig_ids=pis_all)
            distances = distances[0]
            neig_pis = neig_pis[0]
        elif geometry == 'cube':
            distance_vector = np.abs(ut.coord.distance(
                'vector', part['position'], cen_position, part.info['box.length']))

        for di in xrange(DistBin.num):
            dist_bin_lim = DistBin.get_bin_limit(scaling, di)

            if geometry == 'sphere':
                pis_all_d = neig_pis[ut.array.elements(distances, dist_bin_lim)]
            elif geometry == 'cube':
                pis_all_d = np.array(pis_all)
                for dimen_i in xrange(part['position'].shape[1]):
                    pis_all_d = ut.array.elements(distance_vector[:, dimen_i], dist_bin_lim,
                                                  pis_all_d)

            pis_contam_d = np.intersect1d(pis_all_d, pis_contam)
            frac = ut.math.Fraction.fraction(pis_contam_d.size, pis_all_d.size)
            self.say('distance = [%.3f, %.3f], fraction = %.5f' %
                     (dist_bin_lim[0], dist_bin_lim[1], frac))
            if frac >= 1.0:
                break

Test = TestClass()
