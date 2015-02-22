'''
Generate initial conditions for MUSIC using AGORA.

Halo masses in {log M_sun}, particle masses in {M_sun}, positions in {kpc comoving}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import log10, Inf, int32, float32
# local ----
import yt
import yt.analysis_modules.halo_finding.api as halo_io
#import yt.analysis_modules.halo_analysis.api as halo_analysis
#import yt.analysis_modules.halo_merger_tree.api as tree_io
from utilities import utility as ut

Fraction = ut.math.FractionClass()

# relatively isolated halos with M_vir ~ 2e11 M_sun
# AGORA uses 473
halo_ids = np.array([414, 415, 417, 438, 439, 457, 466, 468, 473, 497, 499, 503])

# pc = yt.SlicePlot(pf, 'y', ('deposit', 'all_density'),
# center=[0.72157766, 0.5858333, 0.65605193], width=0.002160379324*6)


#===================================================================================================
# read AGORA
#===================================================================================================
class IOClass(ut.array.DictClass, ut.io.SayClass):
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
            hc = halo_analysis.HaloCatalog(
                data_ds=self.snapshot[0], finder_method='hop', output_dir='halo_catalog')
            hc.create()
        '''
        self.hal_yt = halo_io.LoadHaloes(
            self.snapshot[0], self.snapshot_final_directory + 'MergerHalos')
        #self.hal_yt = halo_analysis.HaloCatalog(
        #    data_ds=self.snapshot[0], self.snapshot_final_directory + 'MergerHalos')

        self.hal = ut.array.DictClass()
        self.hal.snap = {}
        self.hal.snap['scale.factor'] = 1 / (1 + self['redshifts'][0])
        self.hal.info = {}
        self.hal.info['box.length'] = self['box.length']

        self.hal['position'] = np.zeros((len(self.hal_yt), 3), dtype=float32)
        self.hal['mass'] = np.zeros(len(self.hal_yt), dtype=float32)
        self.hal['radius'] = np.zeros(len(self.hal_yt), dtype=float32)
        self.hal['particle.num'] = np.zeros(len(self.hal_yt), dtype=int32)

        for hi in xrange(len(self.hal_yt)):
            self.hal['mass'][hi] = self.hal_yt[hi].total_mass()
            self.hal['radius'][hi] = self.hal_yt[hi].maximum_radius()
            self.hal['position'][hi] = self.hal_yt[hi].center_of_mass()
            self.hal['particle.num'][hi] = self.hal_yt[hi].get_size()
        self.hal['mass'] = log10(self.hal['mass'])    # {log M_sun}
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
        dimension_names = ['x', 'y', 'z']
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
                (self.part[zi]['id'].size, len(dimension_names)), dtype=np.float32)
            for dimension_i, dimension_name in enumerate(dimension_names):
                # {kpc comoving}
                self.part[zi]['position'][:, dimension_i] = (
                    self.data[zi]['particle_position_' + dimension_name] *
                    self.hal.info['box.length'])
                del(self.data[zi]['particle_position_' + dimension_name])

            self.part[zi].info['mass.unique'] = np.unique(self.part[zi]['mass'])


#===================================================================================================
# analysis
#===================================================================================================
def get_halos_around_halo(hal, halo_index, distance_max, neig_mass_frac_min=0.5):
    '''
    Get neig_distances & indices of halos that are within distance_max of center of given halo.

    Parameters
    ----------
    halo index: int
    maximum distance {kpc physical}: float
    minimum fraction of input mass to keep neighboring halos: float
    '''
    Neighbor = ut.neighbor.NeighborClass()

    distance_max *= hal['radius'][halo_index]
    mass_min = hal['mass'] + log10(neig_mass_frac_min)
    his_m = ut.array.elements(hal['mass'], [mass_min, Inf])
    neig_distances, neig_indices = Neighbor.get_neighbors(
        hal['position'][[halo_index]], hal['position'][his_m], 200, [1e-6, distance_max],
        hal.info['box.length'], neig_ids=his_m)
    neig_distances /= hal['radius'][halo_index]

    return neig_distances, neig_indices


def get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_vir=True):
    '''
    Get ids of particles that are within distance_max of center of given halo.

    Parameters
    ----------
    halo index: int
    maximum distance {kpc comoving or in units of virial radius}: float
    whether to scale distance by virial radius: boolean
    '''
    if scale_vir:
        distance_max *= Agora.hal['radius'][halo_index]
    # convert distance_max to simulation units [0, 1)
    distance_max /= Agora['box.length']
    sp = Agora.snapshot[0].h.sphere(Agora.hal_yt[halo_index].center_of_mass(), distance_max)

    return np.array(sp['particle_index'], dtype=int32)


def print_contamination_around_halo(
    Agora, halo_index, distance_max, distance_bin_wid=0.5, scale_vir=True):
    '''
    Test lower resolution particle contamination around halo as a function of distance.

    Parameters
    ----------
    Agora data: class
    halo index: int
    maximum distance from center to check: float
    distance bin width: float
    whether to scale distances by virial radius: boolean
    '''
    distance_scaling = 'lin'
    zi = 0

    Say = ut.io.SayClass(print_contamination_around_halo)

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, [0, distance_max], width=distance_bin_wid)

    pids = get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_vir)
    Say.say('read %d particles around halo' % pids.size)

    pis = Agora.part[zi]['id-to-index'][pids]
    distances = ut.coord.distance(
        'scalar', Agora.part[zi]['position'][pis], Agora.hal['position'][halo_index],
        Agora['box.length'])
    if scale_vir:
        distances /= Agora.hal['radius'][halo_index]
        Say.say('halo radius = %.3f kpc comoving' % Agora.hal['radius'][halo_index])

    pis_contam = pis[Agora.part[zi]['mass'][pis] != Agora.part[zi].info['mass.unique'].min()]
    if pis_contam.size == 0:
        Say.say('yay! no contaminating particles out to distance_max = %.3f' % distance_max)
        return

    for di in xrange(DistanceBin.num):
        dist_bin_lim = DistanceBin.get_bin_limit(distance_scaling, di)
        pis_d = pis[ut.array.elements(distances, dist_bin_lim)]
        pis_contam_d = np.intersect1d(pis_d, pis_contam)
        num_frac = Fraction.get_fraction(pis_contam_d.size, pis_d.size)
        mass_frac = Fraction.get_fraction(
            np.sum(Agora.part[zi]['mass'][pis_contam_d]), np.sum(Agora.part[zi]['mass'][pis_d]))
        Say.say('distance = [%.3f, %.3f]: fraction by number = %.5f, by mass = %.5f' %
                (dist_bin_lim[0], dist_bin_lim[1], num_frac, mass_frac))
        if num_frac >= 1.0:
            break


def print_contamination_in_box(
    part, center_pos=None, distance_lim=None, distance_bin_num=20, scaling='lin',
    geometry='cube'):
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
    Say = ut.io.SayClass(print_contamination_in_box)

    Neighbor = ut.neighbor.NeighborClass()

    if distance_lim is None:
        distance_lim = [0, 0.5 * (1 - 1e-5) * part.info['box.length']]

    if center_pos is None:
        center_pos = np.zeros(part['position'].shape[1])
        for dimension_i in xrange(part['position'].shape[1]):
            center_pos[dimension_i] = 0.5 * part.info['box.length']
    print('center position = %s' % center_pos)

    DistanceBin = ut.bin.DistanceBinClass(scaling, distance_lim, distance_bin_num)

    masses_unique = np.unique(part['mass'])
    pis_all = ut.array.arange_length(part['mass'])
    pis_contam = pis_all[part['mass'] != masses_unique.min()]

    if geometry == 'sphere':
        distances, neig_pis = Neighbor.get_neighbors(
            center_pos, part['position'], part['mass'].size,
            distance_lim, part.info['box.length'], neig_ids=pis_all)
        distances = distances[0]
        neig_pis = neig_pis[0]
    elif geometry == 'cube':
        distance_vector = np.abs(ut.coord.distance(
            'vector', part['position'], center_pos, part.info['box.length']))

    for di in xrange(DistanceBin.num):
        dist_bin_lim = DistanceBin.get_bin_limit(scaling, di)

        if geometry == 'sphere':
            pis_all_d = neig_pis[ut.array.elements(distances, dist_bin_lim)]
        elif geometry == 'cube':
            pis_all_d = np.array(pis_all)
            for dimension_i in xrange(part['position'].shape[1]):
                pis_all_d = ut.array.elements(
                    distance_vector[:, dimension_i], dist_bin_lim, pis_all_d)

        pis_contam_d = np.intersect1d(pis_all_d, pis_contam)
        frac = Fraction.get_fraction(pis_contam_d.size, pis_all_d.size)
        Say.say('distance = [%.3f, %.3f], fraction = %.5f' %
                (dist_bin_lim[0], dist_bin_lim[1], frac))
        if frac >= 1.0:
            break


def print_ic_zoom_region_for_halo(
    Agora, halo_index, refinement_num=1, distance_max=None, geometry='cube'):
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
        distance_max = Agora.hal['radius'][halo_index] * 1.2

    if geometry == 'cube':
        distance_max = (1.5 * refinement_num + 1) * distance_max
    elif geometry == 'ellipsoid':
        distance_max = (1.5 * refinement_num + 7) * distance_max

    pids = Agora.get_particle_ids_around_halo(halo_index, distance_max, scale_vir=False)

    pis = Agora.part[1]['id-to-index'][pids]
    poss = Agora.part[1]['position'][pis]
    lims = np.zeros((poss.shape[1], 2))
    wids = np.zeros(poss.shape[1])
    for dimen_i in xrange(poss.shape[1]):
        lims[dimen_i] = np.array(ut.array.get_limits(poss[:, dimen_i]))
        wids[dimen_i] = lims[[dimen_i]].max() - lims[[dimen_i]].min()
        Agora.say('dimension-%d: %s (%.3f) kpc, %s (%.8f) box length' %
                  (dimen_i, ut.array.get_limits(lims[[dimen_i]], digit_num=3), wids[dimen_i],
                   ut.array.get_limits(lims[[dimen_i]] / Agora['box.length'], digit_num=8),
                   wids[dimen_i] / Agora['box.length']))
    lims /= Agora['box.length']
    wids /= Agora['box.length']
    Agora.say('for MUSIC config file:')
    Agora.say('  ref_offset = %.8f, %.8f, %.8f' % (lims[0, 0], lims[1, 0], lims[2, 0]))
    Agora.say('  ref_extent = %.8f, %.8f, %.8f' % (wids[0], wids[1], wids[2]))
