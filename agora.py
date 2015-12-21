'''
Generate initial conditions for MUSIC using AGORA.

Masses in {M_sun}, positions in {kpc comoving}, distances in {kpc physical}.

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import Inf
# local ----
import yt
import yt.analysis_modules.halo_finding.api as halo_io
#import yt.analysis_modules.halo_analysis.api as halo_analysis
#import yt.analysis_modules.halo_merger_tree.api as tree_io
import utilities as ut


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
        read_catalog=True):
        '''
        Parameters
        ----------
        agora_dir : string : directory of AGORA simulation
        snapshot_final_dir : string : directory of final shapshot
        snapshot_initial_dir : string : directory of initial snapshot
        read_catalog : boolean : whether to read halo catalog
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

        self.Cosmology = ut.cosmology.CosmologyClass(source='agora')

        if read_catalog:
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
        self.hal.snapshot = {}
        self.hal.snapshot['redshift'] = self['redshifts'][0]
        self.hal.snapshot['scalefactor'] /= (1 + self['redshifts'][0])
        self.hal.Cosmology = self.Cosmology

        self.hal['position'] = np.zeros((len(self.hal_yt), 3), dtype=np.float32)
        self.hal['mass'] = np.zeros(len(self.hal_yt), dtype=np.float32)
        self.hal['radius'] = np.zeros(len(self.hal_yt), dtype=np.float32)
        self.hal['particle.number'] = np.zeros(len(self.hal_yt), dtype=np.int32)

        for hi in range(len(self.hal_yt)):
            self.hal['mass'][hi] = self.hal_yt[hi].total_mass()
            self.hal['radius'][hi] = self.hal_yt[hi].maximum_radius()
            self.hal['position'][hi] = self.hal_yt[hi].center_of_mass()
            self.hal['particle.number'][hi] = self.hal_yt[hi].get_size()
        self.hal['mass'] = self.hal['mass']  # {M_sun}
        # {kpc physical}
        self.hal['radius'] *= self.hal.info['box.length'] * self.hal.snapshot['scalefactor']
        self.hal['position'] *= self.hal.info['box.length']  # {kpc comoving}

        NearestNeig = ut.catalog.NearestNeighborClass()
        NearestNeig.assign_to_self(
            self.hal, 'mass', [1, Inf], [1, Inf], 200, 8000, 'comoving', 'halo')
        NearestNeig.assign_to_catalog(self.hal)

    def read_particles(self, tis=[0, 1], divvy_by_mass=False):
        '''
        Read particles, save as dictionary to self.

        Parameters
        ----------
        tis : int or list : indices of snapshot  to read
        divvy_by_mass : boolean : whether to divvy particles by species
        '''
        dimension_names = ['x', 'y', 'z']
        if np.isscalar(tis):
            tis = [tis]

        self.part = ut.array.ListClass()
        for ti in tis:
            self.part.append(ut.array.DictClass())
        for ti in tis:
            self.part[ti]
            self.part[ti]['id'] = []
            self.part[ti]['id-to-index'] = []
            self.part[ti]['mass'] = []
            self.part[ti]['position'] = []
            self.part[ti].info = {}
            self.part[ti].info['box.length'] = self['box.length']
            self.part[ti].info['has.baryons'] = False
            self.part[ti].snapshot = {}
            self.part[ti].snapshot['redshift'] = self['redshifts'][ti]
            self.part[ti].snapshot['scalefactor'] /= (1 + self['redshifts'][ti])
            self.part[ti].Cosmology = self.Cosmology

        self.part.info = {}
        self.part.info['box.length'] = self['box.length']
        self.part.info['has.baryons'] = False
        self.part.Cosmology = self.Cosmology

        for ti in tis:
            # in yt, particle_index = id
            self.part[ti]['id'] = np.array(self.data[ti]['particle_index'], dtype=np.int32)
            if np.unique(self.part[ti]['id']).size != self.part[ti]['id'].size:
                raise ValueError('partice ids are not unique')
            del(self.data[ti]['particle_index'])

            ut.catalog.assign_id_to_index_species(self.part[ti], 'id', 0)

            self.part[ti]['mass'] = np.array(
                self.data[ti]['particle_mass'].in_units('Msun'), dtype=np.float32)  # {M_sun}
            del(self.data[ti]['particle_mass'])

            self.part[ti]['position'] = np.zeros(
                (self.part[ti]['id'].size, len(dimension_names)), dtype=np.float32)
            for dimension_i, dimension_name in enumerate(dimension_names):
                # {kpc comoving}
                self.part[ti]['position'][:, dimension_i] = (
                    self.data[ti]['particle_position_' + dimension_name] *
                    self.hal.info['box.length'])
                del(self.data[ti]['particle_position_' + dimension_name])

            self.part[ti].info['mass.unique'] = np.unique(self.part[ti]['mass'])

        if divvy_by_mass:
            self.divvy_particles_by_mass(tis)

        self.order_particles_by_id(tis)

    def divvy_particles_by_mass(self, tis=[0, 1]):
        '''
        Divvy particles into separate dictionaries by mass.

        Parameters
        ----------
        tis : int or list : snapshot indices to read
        '''
        self.say('separating particle species by mass')

        for ti in tis:
            self.say('  ti = %d' % (ti))
            spec_masses = np.unique(self.part[ti]['mass'])

            props = tuple(self.part[ti].keys())

            for spec_i, spec_mass in enumerate(spec_masses):
                spec_indices = np.where(self.part[ti]['mass'] == spec_mass)[0]

                if spec_i == 0:
                    spec_name = 'dark'
                else:
                    spec_name = 'dark.%d' % (spec_i + 1)

                self.part[ti][spec_name] = {}
                for prop in props:
                    if prop != 'id-to-index':
                        self.part[ti][spec_name][prop] = self.part[ti][prop][spec_indices]
                self.say('  %s: %d particles' % (spec_name, spec_indices.size))

            for prop in props:
                del(self.part[ti][prop])

    def order_particles_by_id(self, tis=[0, 1]):
        '''
        Order particles in catalog by id.

        Parameters
        ----------
        tis : int or list : snapshot indices to read
        '''
        self.say('ordering particles by id')

        if 'dark' in self.part[0]:
            for ti in tis:
                spec_names = [spec_name for spec_name in self.part[ti] if 'dark' in spec_name]
                for spec_name in spec_names:
                    indices_sorted_by_id = np.argsort(self.part[ti][spec_name]['id'])
                    for prop in self.part[ti][spec_name]:
                        self.part[ti][spec_name][prop] = \
                            self.part[ti][spec_name][prop][indices_sorted_by_id]
        else:
            for ti in tis:
                indices_sorted_by_id = np.argsort(self.part[ti]['id'])
                for prop in self.part[ti]:
                    self.part[ti][prop] = self.part[ti][prop][indices_sorted_by_id]


#===================================================================================================
# analysis
#===================================================================================================
def get_halos_around_halo(hal, halo_index, distance_max, scale_virial=True, neig_mass_frac_min=0.5):
    '''
    Get distances {kpc physical} and indices of halos that are within distance_max of center of
    given halo.

    Parameters
    ----------
    hal : dict : catalog of halos
    halo_index : int : index of halo to select
    distance_max : float : maximum distance {kpc physical or R_halo}
    neig_mass_frac_min : float : minimum fraction of input mass to select neighboring halos
    '''
    Neighbor = ut.neighbor.NeighborClass()

    if scale_virial:
        distance_max *= hal['radius'][halo_index]

    mass_min = neig_mass_frac_min * hal['mass'][halo_index]
    his_m = ut.array.get_indices(hal['mass'], [mass_min, Inf])

    neig_distances, neig_indices = Neighbor.get_neighbors(
        hal['position'][[halo_index]], hal['position'][his_m], 200, [1e-6, distance_max],
        hal.info['box.length'], hal.snapshot['scalefactor'], neig_ids=his_m)

    if scale_virial:
        neig_distances /= hal['radius'][halo_index]

    return neig_distances, neig_indices


def get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_virial=True):
    '''
    Get ids of particles that are within distance_max of center of given halo.

    Parameters
    ----------
    Agora : class : Agora data class
    halo_index: int : index of halo
    distance_max : float : maximum distance {kpc comoving or in units of R_halo}
    scale_virial : boolean : whether to scale distance by virial radius
    '''
    if scale_virial:
        distance_max *= Agora.hal['radius'][halo_index]

    # convert distance_max to simulation units [0, 1)
    distance_max /= Agora['box.length']
    sp = Agora.snapshot[0].h.sphere(Agora.hal_yt[halo_index].center_of_mass(), distance_max)

    return np.array(sp['particle_index'], dtype=np.int32)


def print_contamination_around_halo(
    Agora, halo_index, distance_max, distance_bin_width=0.5, scale_virial=True):
    '''
    Test lower resolution particle contamination around halo as a function of distance.

    Parameters
    ----------
    Agora : class : Agora data class
    halo_index: int : index of halo
    distance_max : float : maximum distance from halo center to check
    distance_bin_width : float : width of distance bin for printing
    scale_virial : boolean : whether to scale distances by virial radius
    '''
    distance_scaling = 'lin'
    ti = 0

    Say = ut.io.SayClass(print_contamination_around_halo)

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, [0, distance_max], width=distance_bin_width)

    pids = get_particle_ids_around_halo(Agora, halo_index, distance_max, scale_virial)
    Say.say('read %d particles around halo' % pids.size)

    pis = Agora.part[ti]['id-to-index'][pids]
    distances = ut.coordinate.get_distances(
        'scalar', Agora.part[ti]['position'][pis], Agora.hal['position'][halo_index],
        Agora['box.length'])
    if scale_virial:
        distances /= Agora.hal['radius'][halo_index]
        Say.say('halo radius = %.3f kpc comoving' % Agora.hal['radius'][halo_index])

    pis_contam = pis[Agora.part[ti]['mass'][pis] != Agora.part[ti].info['mass.unique'].min()]
    if pis_contam.size == 0:
        Say.say('yay! no contaminating particles out to distance_max = %.3f' % distance_max)
        return

    for dist_i in range(DistanceBin.number):
        distance_bin_limits = DistanceBin.get_bin_limits(distance_scaling, dist_i)
        pis_d = pis[ut.array.get_indices(distances, distance_bin_limits)]
        pis_contam_d = np.intersect1d(pis_d, pis_contam)
        num_frac = Fraction.get_fraction(pis_contam_d.size, pis_d.size)
        mass_frac = Fraction.get_fraction(
            np.sum(Agora.part[ti]['mass'][pis_contam_d]), np.sum(Agora.part[ti]['mass'][pis_d]))
        Say.say('distance = [%.3f, %.3f]: fraction by number = %.5f, by mass = %.5f' %
                (distance_bin_limits[0], distance_bin_limits[1], num_frac, mass_frac))
        if num_frac >= 1.0:
            break


def print_contamination_in_box(
    part, center_position=None, distance_limits=None, distance_bin_number=20, scaling='lin',
    geometry='cube'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    part : dict : catalog of particles
    center_position : array : 3-d position of center {kpc comoving}
    distance_limits : float : maximum distance from center to check {kpc physical}
    distance_bin_number : int : number of distance bins
    scaling : string : 'log', 'lin'
    geometry : string : geometry of region: 'cube', 'sphere'
    '''
    Say = ut.io.SayClass(print_contamination_in_box)

    Neighbor = ut.neighbor.NeighborClass()

    if distance_limits is None:
        distance_limits = [0, 0.5 * (1 - 1e-5) * part.info['box.length']]

    if center_position is None:
        center_position = np.zeros(part['position'].shape[1])
        for dimension_i in range(part['position'].shape[1]):
            center_position[dimension_i] = 0.5 * part.info['box.length']
    print('center position = %s' % center_position)

    DistanceBin = ut.binning.DistanceBinClass(scaling, distance_limits, number=distance_bin_number)

    masses_unique = np.unique(part['mass'])
    pis_all = ut.array.arange_length(part['mass'])
    pis_contam = pis_all[part['mass'] != masses_unique.min()]

    if geometry == 'sphere':
        distances, neig_pis = Neighbor.get_neighbors(
            center_position, part['position'], part['mass'].size,
            distance_limits, part.info['box.length'], neig_ids=pis_all)
        distances = distances[0]
        neig_pis = neig_pis[0]
    elif geometry == 'cube':
        distance_vector = np.abs(ut.coordinate.get_distances(
            'vector', part['position'], center_position, part.info['box.length']))

    for dist_i in range(DistanceBin.number):
        distance_bin_limits = DistanceBin.get_bin_limits(scaling, dist_i)

        if geometry == 'sphere':
            pis_all_d = neig_pis[ut.array.get_indices(distances, distance_bin_limits)]
        elif geometry == 'cube':
            pis_all_d = np.array(pis_all)
            for dimension_i in range(part['position'].shape[1]):
                pis_all_d = ut.array.get_indices(
                    distance_vector[:, dimension_i], distance_bin_limits, pis_all_d)

        pis_contam_d = np.intersect1d(pis_all_d, pis_contam)
        frac = Fraction.get_fraction(pis_contam_d.size, pis_all_d.size)
        Say.say('distance = [%.3f, %.3f], fraction = %.5f' %
                (distance_bin_limits[0], distance_bin_limits[1], frac))
        if frac >= 1.0:
            break


def print_ic_zoom_region_for_halo(
    Agora, halo_index, refinement_number=1, distance_max=None, geometry='cube'):
    '''
    Print extent of lagrangian region at z_initial around given halo at z = 0.
    Use rules of thumb from Onorbe et al.

    Parameters
    ----------
    halo_index : int : index of halo
    refinement_number : int : number of refinement levels beyond current level for zoom-in region
    distance_max : float : maximum distance want to be uncontaminated {kpc comoving}
        if None, use R_vir
    geometry : string : geometry of zoom-in lagrangian regon in initial conditions:
        'cube', 'ellipsoid'
    '''
    if not distance_max:
        distance_max = Agora.hal['radius'][halo_index] * 1.2

    if geometry == 'cube':
        distance_max = (1.5 * refinement_number + 1) * distance_max
    elif geometry == 'ellipsoid':
        distance_max = (1.5 * refinement_number + 7) * distance_max

    pids = Agora.get_particle_ids_around_halo(halo_index, distance_max, scale_vir=False)

    pis = Agora.part[1]['id-to-index'][pids]
    positions = Agora.part[1]['position'][pis]
    limits = np.zeros((positions.shape[1], 2))
    widths = np.zeros(positions.shape[1])
    for dimen_i in range(positions.shape[1]):
        limits[dimen_i] = np.array(ut.array.get_limits(positions[:, dimen_i]))
        widths[dimen_i] = limits[[dimen_i]].max() - limits[[dimen_i]].min()
        Agora.say('dimension-%d: %s (%.3f) kpc, %s (%.8f) box length' %
                  (dimen_i, ut.array.get_limits(limits[[dimen_i]], digit_number=3), widths[dimen_i],
                   ut.array.get_limits(limits[[dimen_i]] / Agora['box.length'], digit_number=8),
                   widths[dimen_i] / Agora['box.length']))
    limits /= Agora['box.length']
    widths /= Agora['box.length']
    Agora.say('for MUSIC config file:')
    Agora.say('  ref_offset = %.8f, %.8f, %.8f' % (limits[0, 0], limits[1, 0], limits[2, 0]))
    Agora.say('  ref_extent = %.8f, %.8f, %.8f' % (widths[0], widths[1], widths[2]))
