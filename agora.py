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
from utilities import cosmology

Fraction = ut.math.FractionClass()

# isolated halos with M_vir ~ 2e11 M_sun in agora_ref9 (AGORA uses 473)
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
        self, agora_dir='.', snapshot_final_dir='DD0320/', snapshot_initial_dir='DD0000/',
        read_halo=True):
        '''
        Parameters
        ----------
        directory of AGORA simulation: string
        directory of final shapshot: string
        directory of initial snapshot: string
        whether to read halo catalog: boolean
        '''
        self.agora_directory = ut.io.get_path(agora_dir)
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

        self.Cosmo = cosmology.CosmologyClass(source='agora')

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
        self.hal.info = {}
        self.hal.info['box.length'] = self['box.length']
        self.hal.snap = {}
        self.hal.snap['redshift'] = self['redshifts'][0]
        self.hal.snap['scale.factor'] = 1 / (1 + self['redshifts'][0])
        self.hal.Cosmo = self.Cosmo

        self.hal['position'] = np.zeros((len(self.hal_yt), 3), dtype=float32)
        self.hal['mass'] = np.zeros(len(self.hal_yt), dtype=float32)
        self.hal['radius'] = np.zeros(len(self.hal_yt), dtype=float32)
        self.hal['particle.num'] = np.zeros(len(self.hal_yt), dtype=int32)

        for hi in xrange(len(self.hal_yt)):
            self.hal['mass'][hi] = self.hal_yt[hi].total_mass()
            self.hal['radius'][hi] = self.hal_yt[hi].maximum_radius()
            self.hal['position'][hi] = self.hal_yt[hi].center_of_mass()
            self.hal['particle.num'][hi] = self.hal_yt[hi].get_size()
        self.hal['mass'] = log10(self.hal['mass'])  # {log M_sun}
        self.hal['radius'] *= self.hal.info['box.length']  # {kpc comoving}
        self.hal['position'] *= self.hal.info['box.length']  # {kpc comoving}

        NearestNeig = ut.catalog.NearestNeighborClass()
        NearestNeig.assign_to_self(
            self.hal, 'mass', [1, Inf], [1, Inf], 200, 8000, 'comoving', 'virial')
        NearestNeig.assign_to_catalog(self.hal)

    def read_particles(self, zis=[0, 1], divvy_by_mass=False):
        '''
        Read particles, save as dictionary to self.

        Parameters
        ----------
        zis : int or list : snapshot indices to read
        divvy_by_mass : boolean : whether to divvy particles by species
        '''
        dimension_names = ['x', 'y', 'z']
        if np.isscalar(zis):
            zis = [zis]

        self.part = ut.array.ListClass()
        for zi in zis:
            self.part.append(ut.array.DictClass())
        for zi in zis:
            self.part[zi]
            self.part[zi]['id'] = []
            self.part[zi]['id-to-index'] = []
            self.part[zi]['mass'] = []
            self.part[zi]['position'] = []
            self.part[zi].info = {}
            self.part[zi].info['box.length'] = self['box.length']
            self.part[zi].info['has.baryons'] = False
            self.part[zi].snap = {}
            self.part[zi].snap['redshift'] = self['redshifts'][zi]
            self.part[zi].snap['scale.factor'] = 1 / (1 + self['redshifts'][zi])
            self.part[zi].Cosmo = self.Cosmo

        self.part.info = {}
        self.part.info['box.length'] = self['box.length']
        self.part.info['has.baryons'] = False
        self.part.Cosmo = self.Cosmo

        for zi in zis:
            # in yt, particle_index = id
            self.part[zi]['id'] = np.array(self.data[zi]['particle_index'], dtype=int32)
            if np.unique(self.part[zi]['id']).size != self.part[zi]['id'].size:
                raise ValueError('partice ids are not unique')
            del(self.data[zi]['particle_index'])

            ut.catalog.assign_id_to_index(self.part[zi], 'id', 0)

            self.part[zi]['mass'] = np.array(
                self.data[zi]['particle_mass'].in_units('Msun'), dtype=float32)  # {M_sun}
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

        if divvy_by_mass:
            self.divvy_particles_by_mass(zis)

        self.order_particles_by_id(zis)

    def divvy_particles_by_mass(self, zis=[0, 1]):
        '''
        Divvy particles into separate dictionaries by mass.

        Parameters
        ----------
        zis : int or list : snapshot indices to read
        '''
        self.say('separating particle species by mass')

        for zi in zis:
            self.say('  zi = %d' % (zi))
            spec_masses = np.unique(self.part[zi]['mass'])

            props = tuple(self.part[zi].keys())

            for spec_i, spec_mass in enumerate(spec_masses):
                spec_indices = np.where(self.part[zi]['mass'] == spec_mass)[0]

                if spec_i == 0:
                    spec_name = 'dark'
                else:
                    spec_name = 'dark.%d' % (spec_i + 1)

                self.part[zi][spec_name] = {}
                for prop in props:
                    if prop != 'id-to-index':
                        self.part[zi][spec_name][prop] = self.part[zi][prop][spec_indices]
                self.say('  %s: %d particles' % (spec_name, spec_indices.size))

            for prop in props:
                del(self.part[zi][prop])

    def order_particles_by_id(self, zis=[0, 1]):
        '''
        Order particles in catalog by id.

        Parameters
        ----------
        zis : int or list : snapshot indices to read
        '''
        self.say('ordering particles by id')

        if 'dark' in self.part[0]:
            for zi in zis:
                spec_names = [spec_name for spec_name in self.part[zi] if 'dark' in spec_name]
                for spec_name in spec_names:
                    indices_sorted_by_id = np.argsort(self.part[zi][spec_name]['id'])
                    for prop in self.part[zi][spec_name]:
                        self.part[zi][spec_name][prop] = \
                            self.part[zi][spec_name][prop][indices_sorted_by_id]
        else:
            for zi in zis:
                indices_sorted_by_id = np.argsort(self.part[zi]['id'])
                for prop in self.part[zi]:
                    self.part[zi][prop] = self.part[zi][prop][indices_sorted_by_id]


#===================================================================================================
# analysis
#===================================================================================================
def get_halos_around_halo(hal, halo_index, distance_max, scale_virial=True, neig_mass_frac_min=0.5):
    '''
    Get distances & indices of halos that are within distance_max of center of given halo.

    Parameters
    ----------
    hal : dict : catalog of halos
    halo_index : int : index of halo to select
    distance_max : float : maximum distance {kpc physical or R_vir}
    distance_kind : string : kind for maximum distance
        options: physical, virial
    neig_mass_frac_min : float : minimum fraction of input mass to select neighboring halos
    '''
    Neighbor = ut.neighbor.NeighborClass()

    if scale_virial:
        distance_max *= hal['radius'][halo_index]
    mass_min = hal['mass'][halo_index] + log10(neig_mass_frac_min)
    his_m = ut.array.elements(hal['mass'], [mass_min, Inf])
    neig_distances, neig_indices = Neighbor.get_neighbors(
        hal['position'][[halo_index]], hal['position'][his_m], 200, [1e-6, distance_max],
        hal.info['box.length'], neig_ids=his_m)
    if scale_virial:
        neig_distances /= hal['radius'][halo_index]

    return neig_distances, neig_indices


def get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_virial=True):
    '''
    Get ids of particles that are within distance_max of center of given halo.

    Parameters
    ----------
    halo index: int
    maximum distance {kpc comoving or in units of virial radius}: float
    whether to scale distance by virial radius: boolean
    '''
    if scale_virial:
        distance_max *= Agora.hal['radius'][halo_index]
    # convert distance_max to simulation units [0, 1)
    distance_max /= Agora['box.length']
    sp = Agora.snapshot[0].h.sphere(Agora.hal_yt[halo_index].center_of_mass(), distance_max)

    return np.array(sp['particle_index'], dtype=int32)


def print_contamination_around_halo(
    Agora, halo_index, distance_max, distance_bin_wid=0.5, scale_virial=True):
    '''
    Test lower resolution particle contamination around halo as a function of distance.

    Parameters
    ----------
    Agora : class : Agora data class
    halo_index: int : index of halo
    distance_max : float : maximum distance from halo center to check
    distance_bin_wid : float : width of distance bin for printing
    scale_virial : boolean : whether to scale distances by virial radius
    '''
    distance_scaling = 'lin'
    zi = 0

    Say = ut.io.SayClass(print_contamination_around_halo)

    DistanceBin = ut.bin.DistanceBinClass(
        distance_scaling, [0, distance_max], width=distance_bin_wid)

    pids = get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_virial)
    Say.say('read %d particles around halo' % pids.size)

    pis = Agora.part[zi]['id-to-index'][pids]
    distances = ut.coord.distance(
        'scalar', Agora.part[zi]['position'][pis], Agora.hal['position'][halo_index],
        Agora['box.length'])
    if scale_virial:
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
