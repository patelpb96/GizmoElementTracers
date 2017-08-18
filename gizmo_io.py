'''
Read Gizmo snapshots.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatibility
import collections
import h5py
import numpy as np
from numpy import log10, Inf  # @UnusedImport
# local ----
import utilities as ut


#===================================================================================================
# particle dictionary class
#===================================================================================================
class ParticleDictionaryClass(dict):
    '''
    Dictionary class to store particle data.
    Allows greater flexibility for producing derived quantities.
    '''
    def __init__(self):
        # use to translate between element name and index in element table
        self.element_dict = collections.OrderedDict()
        self.element_dict['metals'] = self.element_dict['total'] = 0
        self.element_dict['helium'] = self.element_dict['he'] = 1
        self.element_dict['carbon'] = self.element_dict['c'] = 2
        self.element_dict['nitrogen'] = self.element_dict['n'] = 3
        self.element_dict['oxygen'] = self.element_dict['o'] = 4
        self.element_dict['neon'] = self.element_dict['ne'] = 5
        self.element_dict['magnesium'] = self.element_dict['mg'] = 6
        self.element_dict['silicon'] = self.element_dict['si'] = 7
        self.element_dict['sulphur'] = self.element_dict['s'] = 8
        self.element_dict['calcium'] = self.element_dict['ca'] = 9
        self.element_dict['iron'] = self.element_dict['fe'] = 10

        # to use if read only subset of elements
        self.element_pointer = np.arange(len(self.element_dict) / 2, dtype=np.int32)

    def prop(self, property_name='', indices=None):
        '''
        Get property, either from self dictionary or derive.
        If several properties, need to provide mathematical relationship, for example:
        'log temperature', 'temperature / density', 'abs position'

        Parameters
        ----------
        property_name : string : name of property
        indices : array : indices to select on

        Returns
        -------
        values : float or array : convertes values as float (for scalar) or numpy array
        '''
        ## parsing general to all catalogs ----------
        property_name = property_name.strip()  # strip white space

        # if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]

        # math relation, combining more than one property
        if ('/' in property_name or '*' in property_name or '+' in property_name or
                '-' in property_name):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise ValueError(
                    'property = {} is not valid input to {}'.format(property_name, self.__class__))

            # make copy so not change values in input catalog
            prop_values = np.array(self.prop(prop_names[0], indices))

            for prop_name in prop_names[1:]:
                if '/' in property_name:
                    if np.isscalar(prop_values):
                        if self.prop(prop_name, indices) == 0:
                            prop_values = np.nan
                        else:
                            prop_values = prop_values / self.prop(prop_name, indices)
                    else:
                        masks = self.prop(prop_name, indices) != 0
                        prop_values[masks] = (prop_values[masks] /
                                              self.prop(prop_name, indices)[masks])
                        masks = self.prop(prop_name, indices) == 0
                        prop_values[masks] = np.nan
                if '*' in property_name:
                    prop_values = prop_values * self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values = prop_values + self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values = prop_values - self.prop(prop_name, indices)

            if prop_values.size == 1:
                prop_values = np.float(prop_values)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        ## parsing specific to this catalog ----------
        if 'form.' in property_name or property_name == 'age':
            if property_name == 'age' or ('time' in property_name and 'lookback' in property_name):
                # look-back time (stellar age) to formation
                values = self.snapshot['time'] - self.prop('form.time', indices)
            elif 'time' in property_name:
                # time (age of universe) of formation
                values = self.Cosmology.get_time(
                    self.prop('form.scalefactor', indices), 'scalefactor')
            elif 'redshift' in property_name:
                # redshift of formation
                values = 1 / self.prop('form.scalefactor', indices) - 1
            elif 'snapshot' in property_name:
                # snapshot index immediately after formation
                # increase formation scale-factor slightly for safety, because scale-factors of
                # written snapshots do not exactly coincide with input scale-factors
                padding_factor = (1 + 1e-7)
                values = self.Snapshot.get_snapshot_indices(
                    'scalefactor', np.clip(self['form.scalefactor'] * padding_factor, 0, 1),
                    round_kind='up')

            return values

        if 'number.density' in property_name:
            values = (self.prop('density', indices) * ut.const.proton_per_sun *
                      ut.const.kpc_per_cm ** 3)

            if '.hydrogen' in property_name:
                # number density of hydrogen, using actual hydrogen mass of each particle [cm ^ -3]
                values = values * self.prop('massfraction.hydrogen', indices)
            else:
                # number density of 'hydrogen', assuming solar metallicity for particles [cm ^ -3]
                values = values * ut.const.sun_hydrogen_mass_fraction

            return values

        if 'kernel.length' in property_name:
            # gaussian standard-deviation length (for cubic kernel) = inter-particle spacing [pc]
            return 1000 * (self.prop('mass', indices) / self.prop('density', indices)) ** (1 / 3)

        if 'mass.' in property_name:
            # mass of individual element
            values = (self.prop('mass', indices) *
                      self.prop(property_name.replace('mass.', 'massfraction.'), indices))

            if property_name == 'mass.hydrogen.neutral':
                # mass of neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices)

            return values

        # elemental abundance
        if 'massfraction.' in property_name or 'metallicity.' in property_name:
            # special cases
            if 'massfraction.hydrogen' in property_name:
                # special case: mass fraction of hydrogen (excluding helium and metals)
                values = (1 - self.prop('massfraction', indices)[:, 0] -
                          self.prop('massfraction', indices)[:, 1])

                if property_name == 'massfraction.hydrogen.neutral':
                    # mass fraction of neutral hydrogen (excluding helium, metals, and ionized)
                    values = values * self.prop('hydrogen.neutral.fraction', indices)

                return values

            elif 'alpha' in property_name:
                return np.mean(
                    [self.prop('metallicity.o', indices),
                     self.prop('metallicity.mg', indices),
                     self.prop('metallicity.si', indices),
                     self.prop('metallicity.ca', indices),
                     ], 0)

            # normal cases
            element_index = None
            for prop_name in property_name.split('.'):
                if prop_name in self.element_dict:
                    element_index = self.element_pointer[self.element_dict[prop_name]]
                    element_name = prop_name
                    break

            if element_index is None:
                raise ValueError(
                    'property = {} is not valid input to {}'.format(property_name, self.__class__))

            if indices is None:
                values = self['massfraction'][:, element_index]
            else:
                values = self['massfraction'][indices, element_index]

            if 'metallicity.' in property_name:
                values = ut.math.get_log(
                    values / ut.const.sun_composition[element_name]['massfraction'])

            return values

        # distance wrt galaxy/halo center
        if 'distance' in property_name:
            # 3-D distance vector
            values = ut.coordinate.get_distances(
                'vector', self.prop('position', indices), self.center_position,
                self.info['box.length']) * self.snapshot['scalefactor']  # [kpc physical]

            if 'principal' in property_name:
                # align with principal axes
                values = ut.coordinate.get_coordinates_rotated(values, self.principal_axes_vectors)

                if '2d' in property_name:
                    # compute distances along major axes and minor axis (R and Z)
                    values = ut.coordinate.get_distances_major_minor(values)

            return values

        # should not get this far without a return
        raise ValueError(
            'property = {} is not valid input to {}'.format(property_name, self.__class__))


#===================================================================================================
# read
#===================================================================================================
class ReadClass(ut.io.SayClass):
    '''
    Read Gizmo snapshot.
    '''
    def __init__(self, snapshot_name_base='snap*'):
        '''
        Set properties for snapshot file names.

        snapshot_name_base : string : name base of snapshot file/directory
        '''
        self.snapshot_name_base = snapshot_name_base
        self.file_extension = '.hdf5'

        self.gas_eos = 5 / 3  # gas equation of state

        # create ordered dictionary to convert particle species name to its id,
        # set all possible species, and set the order in which to read species
        self.species_dict = collections.OrderedDict()
        # dark-matter species
        self.species_dict['dark'] = 1  # dark matter at highest resolution
        self.species_dict['dark.2'] = 2  # lower-resolution dark matter across all resolutions
        #self.species_dict['dark.3'] = 3
        #self.species_dict['dark.4'] = 5
        # baryon species
        self.species_dict['gas'] = 0
        self.species_dict['star'] = 4
        # other - these ids overlap with above, so have to comment in if using them
        self.species_dict['blackhole'] = 5
        #self.species_dict['bulge'] = 2
        #self.species_dict['disk'] = 3

        self.species_all = tuple(self.species_dict.keys())
        self.species_read = list(self.species_all)

    def read_snapshots(
        self, species='all',
        snapshot_value_kind='index', snapshot_values=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        properties='all', element_indices=None, particle_subsample_factor=0,
        separate_dark_lowres=True, sort_dark_by_id=False, force_float32=False,
        assign_center=True, assign_principal_axes=False, assign_orbit=False,
        assign_form_host_distance=False,
        check_properties=True):
        '''
        Read given properties for given particle species from simulation snapshot file[s].
        Return as dictionary class.

        Parameters
        ----------
        species : string or list : name[s] of particle species:
            'all' = all species in file
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution
            'star' = stars
            'blackhole' = black holes, if run contains them
            'bulge' or 'disk' = stars for non-cosmological run
        snapshot_value_kind : string :
            input snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : string : directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_dict
        element_indices : int or list : indices of elements to keep
            note: 0 = total metals, 1 = helium, 10 = iron, None or 'all' = read all elements
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        separate_dark_lowres : boolean :
            whether to separate low-resolution dark matter into separate dicts according to mass
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        force_float32 : boolean : whether to force all floats to 32-bit, to save memory
        assign_center : boolean : whether to assign center position and velocity of galaxy/halo
        assign_principal_axes : boolean : whether to assign principal axes (moment of intertia)
        assign_orbit : booelan : whether to assign derived orbital properties wrt galaxy/halo center
        assign_form_host_distance : boolean :
            whether to assign distance from host galaxy at formation to stars
        check_properties : boolean : whether to check sanity of particle properties after read in

        Returns
        -------
        parts : dictionary or list thereof :
            if single snapshot, return as dictionary, else if multiple snapshots, return as list
        '''
        # parse input species to read
        if species == 'all' or species == ['all'] or not species:
            # read all species in snapshot
            species = self.species_all
        else:
            # read subsample of species in snapshot
            if np.isscalar(species):
                species = [species]  # ensure is list
            # check if input species names are valid
            for spec in list(species):
                if spec not in self.species_dict:
                    species.remove(spec)
                    self.say('! not recognize input species = {}'.format(spec))
        self.species_read = list(species)

        # read information about snapshot times
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)

        snapshot_values = ut.array.arrayize(snapshot_values)

        parts = []  # list to store particle dictionaries

        # read all input snapshots
        for snapshot_value in snapshot_values:
            snapshot_index = Snapshot.parse_snapshot_value(snapshot_value_kind, snapshot_value)

            # read header from snapshot file
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory, simulation_name)

            # read particles from snapshot file[s]
            part = self.read_particles(
                'index', snapshot_index, simulation_directory, snapshot_directory, properties,
                element_indices, force_float32, header)

            # read/get (additional) cosmological parameters
            if header['is.cosmological']:
                Cosmology = self.get_cosmology(
                    simulation_directory, header['omega_lambda'], header['omega_matter'],
                    hubble=header['hubble'])

            # adjust properties for each species
            self.adjust_particle_properties(
                part, header, particle_subsample_factor, separate_dark_lowres, sort_dark_by_id)

            # check sanity of particle properties read in
            if check_properties:
                self.check_properties(part)

            # assign auxilliary information to particle dictionary class
            # store header dictionary
            part.info = header
            for spec in part:
                part[spec].info = part.info

            # store cosmology class
            part.Cosmology = Cosmology
            for spec in part:
                part[spec].Cosmology = part.Cosmology

            # store information about snapshot time
            time = Cosmology.get_time(header['redshift'])
            part.snapshot = {
                'index': snapshot_index,
                'redshift': header['redshift'],
                'scalefactor': header['scalefactor'],
                'time': time,
                'time.lookback': Cosmology.get_time(0) - time,
                'time.hubble': ut.const.Gyr_per_sec / Cosmology.get_hubble_parameter(0),
            }
            for spec in part:
                part[spec].snapshot = part.snapshot

            # store information on all snapshot times - may or may not be initialized
            part.Snapshot = Snapshot
            for spec in part:
                part[spec].Snapshot = part.Snapshot

            # initialize arrays to store center position and velocity
            part.center_position = []
            part.center_velocity = []
            for spec in part:
                part[spec].center_position = []
                part[spec].center_velocity = []
            if assign_center:
                self.assign_center(part)

            # initialize arrays to store rotation vectors that define principal axes
            part.principal_axes_vectors = []
            for spec in part:
                part[spec].principal_axes_vectors = []
            if assign_center and assign_principal_axes:
                self.assign_principal_axes(part)

            # store derived orbital properties wrt center of galaxy/halo
            if (assign_orbit and 'star' in species and
                    ('velocity' in properties or properties is 'all')):
                self.assign_orbit(part, 'star')

            # assign distance from host galaxy at formation to stars
            if assign_form_host_distance and 'star' in species:
                from . import gizmo_track
                HostDistance = gizmo_track.HostDistanceClass(
                    'star', simulation_directory + 'track/')
                HostDistance.io_form_host_distance(part)

            # if read only 1 snapshot, return as particle dictionary instead of list
            if len(snapshot_values) == 1:
                parts = part
            else:
                parts.append(part)
                print()

        return parts

    def read_snapshots_simulations(
        self, simulation_directories=[], species='all', redshift=0,
        properties='all', element_indices=[0, 1, 6, 10], force_float32=True,
        assign_principal_axes=False):
        '''
        Read snapshots at the same redshift from different simulations.
        Return as list of dictionaries.

        Parameters
        ----------
        directories : list or list of lists :
            list of simulation directories, or list of pairs of directory + simulation name
        species : string or list : name[s] of particle species to read
        redshift : float
        properties : string or list : name[s] of properties to read
        element_indices : int or list : indices of elements to read
        force_float32 : boolean : whether to force positions to be 32-bit
        assign_principal_axes : boolean : whether to assign principal axes (moment of intertia)

        Returns
        -------
        parts : list of dictionaries
        '''
        # parse list of directories
        if np.ndim(simulation_directories) == 0:
            raise ValueError('input simulation_directories = {} but need to input list'.format(
                             simulation_directories))
        elif np.ndim(simulation_directories) == 1:
            # assign null names
            simulation_directories = list(
                zip(simulation_directories, ['' for _ in simulation_directories]))
        elif np.ndim(simulation_directories) == 2:
            pass
        elif np.ndim(simulation_directories) >= 3:
            raise ValueError('not sure how to parse simulation_directories = {}'.format(
                             simulation_directories))

        # first pass, read only header, to check that can read all simulations
        bad_snapshot_value = 0
        for directory, simulation_name in simulation_directories:
            try:
                _header = self.read_header(
                    'redshift', redshift, directory, simulation_name=simulation_name)
            except Exception:
                self.say('! could not read snapshot header at z = {:.3f} in {}'.format(
                         redshift, directory))
                bad_snapshot_value += 1

        if bad_snapshot_value:
            self.say('\n! could not read {} snapshots'.format(bad_snapshot_value))
            return

        parts = []
        directories_read = []
        for directory, simulation_name in simulation_directories:
            try:
                part = self.read_snapshots(
                    species, 'redshift', redshift, directory, simulation_name=simulation_name,
                    properties=properties, element_indices=element_indices,
                    force_float32=force_float32, assign_principal_axes=assign_principal_axes)

                if 'velocity' in properties:
                    self.assign_orbit(part, 'gas')

                parts.append(part)
                directories_read.append(directory)

            except Exception:
                self.say('! could not read snapshot at z = {:.3f} in {}'.format(
                         redshift, directory))

        if not len(parts):
            self.say('! could not read any snapshots at z = {:.3f}'.format(redshift))
            return

        if 'mass' in properties and 'star' in part:
            for part, directory in zip(parts, directories_read):
                print('{}: star.mass = {:.3e}'.format(directory, part['star']['mass'].sum()))

        return parts

    def read_header(
        self, snapshot_value_kind='index', snapshot_value=600, simulation_directory='.',
        snapshot_directory='output/', simulation_name=''):
        '''
        Read header from snapshot file.

        Parameters
        ----------
        snapshot_value_kind : string : input snapshot number kind: index, redshift
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification

        Returns
        -------
        header : dict : header dictionary
        '''
        # convert name in snapshot's header dictionary to custom name preference
        header_dict = {
            # 6-element array of number of particles of each type in file
            'NumPart_ThisFile': 'particle.numbers.in.file',
            # 6-element array of total number of particles of each type (across all files)
            'NumPart_Total': 'particle.numbers.total',
            'NumPart_Total_HighWord': 'particle.numbers.total.high.word',
            # mass of each particle species, if all particles are same
            # (= 0 if they are different, which is usually true)
            'MassTable': 'particle.masses',
            'Time': 'time',  # [Gyr/h]
            'BoxSize': 'box.length',  # [kpc/h comoving]
            'Redshift': 'redshift',
            # number of output files per snapshot
            'NumFilesPerSnapshot': 'file.number.per.snapshot',
            'Omega0': 'omega_matter',
            'OmegaLambda': 'omega_lambda',
            'HubbleParam': 'hubble',
            'Flag_Sfr': 'has.star.formation',
            'Flag_Cooling': 'has.cooling',
            'Flag_StellarAge': 'has.star.age',
            'Flag_Metals': 'has.metals',
            'Flag_Feedback': 'has.feedback',
            'Flag_DoublePrecision': 'has.double.precision',
            'Flag_IC_Info': 'has.ic.info',
        }

        header = {}  # dictionary to store header information

        # parse input values
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        if snapshot_value_kind != 'index':
            Snapshot = self.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_value(snapshot_value_kind, snapshot_value)
        else:
            snapshot_index = snapshot_value

        file_name = self.get_snapshot_file_name(snapshot_directory, snapshot_index)

        self.say('* read header from: {}'.format(file_name.replace('./', '')), end='\n')

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            header_in = file_in['Header'].attrs  # load header dictionary

            for prop_in in header_in:
                prop = header_dict[prop_in]
                header[prop] = header_in[prop_in]  # transfer to custom header dict

        # determine whether simulation is cosmological
        if (0 < header['hubble'] < 1 and 0 < header['omega_matter'] <= 1 and
                0 < header['omega_lambda'] <= 1):
            header['is.cosmological'] = True
        else:
            header['is.cosmological'] = False
            self.say('assuming that simulation is not cosmological')
            self.say('read h = {:.3f}, omega_matter_0 = {:.3f}, omega_lambda_0 = {:.3f}'.format(
                     header['hubble'], header['omega_matter'], header['omega_lambda']))

        # convert header quantities
        if header['is.cosmological']:
            header['scalefactor'] = float(header['time'])
            del(header['time'])
            header['box.length/h'] = float(header['box.length'])
            header['box.length'] /= header['hubble']  # convert to [kpc comoving]
        else:
            header['time'] /= header['hubble']  # convert to [Gyr]

        self.say('snapshot contains the following number of particles:')
        # keep only species that have any particles
        read_particle_number = 0
        for spec in ut.array.get_list_combined(self.species_all, self.species_read):
            spec_id = self.species_dict[spec]
            self.say('{:9s} (id = {}): {} particles'.format(
                     spec, spec_id, header['particle.numbers.total'][spec_id]))

            if header['particle.numbers.total'][spec_id] > 0:
                read_particle_number += header['particle.numbers.total'][spec_id]
            elif spec in self.species_read:
                self.species_read.remove(spec)

        if read_particle_number <= 0:
            raise ValueError('! snapshot file[s] contain no particles of species = {}'.format(
                             self.species_read))

        # check if simulation contains baryons
        header['has.baryons'] = False
        for spec in ut.array.get_list_combined(self.species_all, ['gas', 'star', 'disk', 'bulge']):
            spec_id = self.species_dict[spec]
            if header['particle.numbers.total'][spec_id] > 0:
                header['has.baryons'] = True
                break

        # assign simulation name
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.split('/')[-2]
            simulation_name = simulation_name.replace('_', ' ')
            simulation_name = simulation_name.replace('res', 'r')
        header['simulation.name'] = simulation_name

        header['catalog.kind'] = 'particle'

        print()

        return header

    def read_particles(
        self, snapshot_value_kind='index', snapshot_value=600, simulation_directory='.',
        snapshot_directory='output/', properties='all', element_indices=None,
        force_float32=False, header=None):
        '''
        Read particles from snapshot file[s].

        Parameters
        ----------
        snapshot_value_kind : string : input snapshot number kind: index, redshift
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_dict
        element_indices : int or list : indices of elements to keep
            note: 0 = total metals, 1 = helium, 10 = iron, None or 'all' = read all elements
        force_float32 : boolean : whether to force all floats to 32-bit, to save memory

        Returns
        -------
        part : dict : catalog of particles
        '''
        # convert name in snapshot's particle dictionary to custon name preference
        # if comment out any prop, will not read it
        property_dict = {
            ## all particles ----------
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',
            ## particles with adaptive smoothing
            #'AGS-Softening': 'smooth.length',  # for gas, this is same as SmoothingLength

            ## gas particles ----------
            'InternalEnergy': 'temperature',
            'Density': 'density',
            # stored in snapshot file as maximum distance to neighbor (radius of compact support)
            # but here convert to Plummer-equivalent length (for consistency with force softening)
            'SmoothingLength': 'smooth.length',
            #'ArtificialViscosity': 'artificial.viscosity',
            # average free-electron number per proton, averaged over mass of gas particle
            'ElectronAbundance': 'electron.fraction',
            # fraction of hydrogen that is neutral (not ionized)
            'NeutralHydrogenAbundance': 'hydrogen.neutral.fraction',
            'StarFormationRate': 'sfr',  # [M_sun / yr]

            ## star/gas particles ----------
            ## id.generation and id.child initialized to 0 for all gas particles
            ## each time a gas particle splits into two:
            ##   'self' particle retains id.child, other particle gets id.child += 2 ^ id.generation
            ##   both particles get id.generation += 1
            ## allows maximum of 30 generations, then restarts at 0
            ##   thus, particles with id.child > 2^30 are not unique anymore
            'ParticleChildIDsNumber': 'id.child',
            'ParticleIDGenerationNumber': 'id.generation',

            ## mass fraction of individual elements ----------
            ## 0 = all metals (everything not H, He)
            ## 1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
            'Metallicity': 'massfraction',

            ## star particles ----------
            ## 'time' when star particle formed
            ## for cosmological runs, = scale-factor; for non-cosmological runs, = time [Gyr/h]
            'StellarFormationTime': 'form.scalefactor',

            ## black hole particles ----------
            'BH_Mass': 'bh.mass',
            'BH_Mdot': 'form.rate'
        }

        part = ut.array.DictClass()  # dictionary class to store properties for particle species

        # parse input list of properties to read
        if properties == 'all' or properties == ['all'] or not properties:
            properties = list(property_dict.keys())
        else:
            if np.isscalar(properties):
                properties = [properties]  # ensure is list
            # make safe list of properties to read
            properties_temp = []
            for prop in list(properties):
                prop = str.lower(prop)
                if 'massfraction' in prop or 'metallicity' in prop:
                    prop = 'massfraction'  # this has several aliases, so ensure default name
                for prop_in in property_dict:
                    if prop in [str.lower(prop_in), str.lower(property_dict[prop_in])]:
                        properties_temp.append(prop_in)
            properties = properties_temp
            del(properties_temp)

        if 'InternalEnergy' in properties:
            # need helium mass fraction and electron fraction to compute temperature
            for prop in np.setdiff1d(['ElectronAbundance', 'Metallicity'], properties):
                properties.append(prop)

        # parse other input values
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        if snapshot_value_kind != 'index':
            Snapshot = self.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_value(snapshot_value_kind, snapshot_value)
        else:
            snapshot_index = snapshot_value

        if not header:
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory)

        file_name = self.get_snapshot_file_name(snapshot_directory, snapshot_index)

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            if header['file.number.per.snapshot'] == 1:
                self.say('* read particles from: {}'.format(file_name.strip('./')))
            else:
                self.say('* read particles')

            # initialize arrays to store each prop for each species
            for spec in self.species_read:
                spec_id = self.species_dict[spec]
                part_number_tot = header['particle.numbers.total'][spec_id]

                # add species to particle dictionary
                part[spec] = ParticleDictionaryClass()

                # set element pointers if reading only subset of elements
                if (element_indices is not None and len(element_indices) and
                        element_indices != 'all'):
                    if np.isscalar(element_indices):
                        element_indices = [element_indices]
                    for element_i, element_index in enumerate(element_indices):
                        part[spec].element_pointer[element_index] = element_i

                # check if snapshot file happens not to have particles of this species
                if part_numbers_in_file[spec_id] <= 0:
                    # this scenario should occur only for multi-file snapshot
                    if header['file.number.per.snapshot'] == 1:
                        raise ValueError('! no {} particles in single-file snapshot'.format(spec))

                    # need to read in other snapshot files until find one with particles of species
                    for file_i in range(1, header['file.number.per.snapshot']):
                        file_name = file_name.replace('.0.', '.{}.'.format(file_i))
                        # try each snapshot file
                        with h5py.File(file_name, 'r') as file_in_i:
                            part_numbers_in_file_i = file_in_i['Header'].attrs['NumPart_ThisFile']
                            if part_numbers_in_file_i[spec_id] > 0:
                                # found one!
                                part_in = file_in_i['PartType' + str(spec_id)]
                                break
                    else:
                        # tried all files and still did not find particles of species
                        raise ValueError('! no {} particles in any snapshot files'.format(spec))
                else:
                    part_in = file_in['PartType' + str(spec_id)]

                props_print = []
                ignore_flag = False  # whether ignored any properties in the file
                for prop_in in part_in.keys():
                    if prop_in in properties:
                        prop = property_dict[prop_in]

                        # determine shape of prop array
                        if len(part_in[prop_in].shape) == 1:
                            prop_shape = part_number_tot
                        elif len(part_in[prop_in].shape) == 2:
                            prop_shape = [part_number_tot, part_in[prop_in].shape[1]]
                            if (prop_in == 'Metallicity' and element_indices is not None and
                                    element_indices != 'all'):
                                prop_shape = [part_number_tot, len(element_indices)]

                        # determine data type to store
                        prop_in_dtype = part_in[prop_in].dtype
                        if force_float32 and prop_in_dtype == 'float64':
                            prop_in_dtype = np.float32

                        # initialize to -1's
                        part[spec][prop] = np.zeros(prop_shape, prop_in_dtype) - 1

                        if prop == 'id':
                            # initialize so calling an un-itialized value leads to error
                            part[spec][prop] -= part_number_tot

                        if prop_in in property_dict:
                            props_print.append(property_dict[prop_in])
                        else:
                            props_print.append(prop_in)
                    else:
                        ignore_flag = True

                if ignore_flag:
                    props_print.sort()
                    self.say('read {:6}: {}'.format(spec, props_print))

                # special case: particle mass is fixed and given in mass array in header
                if 'Masses' in properties and 'Masses' not in part_in:
                    prop = property_dict['Masses']
                    part[spec][prop] = np.zeros(part_number_tot, dtype=np.float32)

        ## read properties for each species ----------
        # initial particle indices to assign to each species from each file
        part_indices_lo = np.zeros(len(self.species_read), dtype=np.int64)

        # loop over all files at given snapshot
        for file_i in range(header['file.number.per.snapshot']):
            # open i'th of multiple files for snapshot
            file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))

            # open snapshot file
            with h5py.File(file_name_i, 'r') as file_in:
                if header['file.number.per.snapshot'] > 1:
                    self.say('from: ' + file_name_i.split('/')[-1])

                part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

                # read particle properties
                for spec_i, spec in enumerate(self.species_read):
                    spec_id = self.species_dict[spec]
                    if part_numbers_in_file[spec_id] > 0:
                        part_in = file_in['PartType' + str(spec_id)]

                        part_index_lo = part_indices_lo[spec_i]
                        part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                        # check if mass of species is fixed, according to header mass array
                        if 'Masses' in properties and header['particle.masses'][spec_id] > 0:
                            prop = property_dict['Masses']
                            part[spec][prop][
                                part_index_lo:part_index_hi] = header['particle.masses'][spec_id]

                        for prop_in in part_in.keys():
                            if prop_in in properties:
                                prop = property_dict[prop_in]
                                if len(part_in[prop_in].shape) == 1:
                                    part[spec][prop][part_index_lo:part_index_hi] = part_in[prop_in]
                                elif len(part_in[prop_in].shape) == 2:
                                    if (prop_in == 'Metallicity' and element_indices is not None and
                                            element_indices != 'all'):
                                        prop_in = part_in[prop_in][:, element_indices]
                                    else:
                                        prop_in = part_in[prop_in]

                                    part[spec][prop][part_index_lo:part_index_hi, :] = prop_in

                        part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        print()

        return part

    def adjust_particle_properties(
        self, part, header, particle_subsample_factor=None, separate_dark_lowres=True,
        sort_dark_by_id=False):
        '''
        Adjust properties for each species, including unit conversions, separating dark species by
        mass, sorting by id, and subsampling.

        Parameters
        ----------
        part : dict : particle dictionary class
        header : dict : header dictionary
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        separate_dark_lowres : boolean :
            whether to separate low-resolution dark matter into separate dicts according to mass
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        '''
        # if dark.2 contains different masses (refinements), split into separate dicts
        species_name = 'dark.2'

        if species_name in part and 'mass' in part[species_name]:
            dark_lowres_masses = np.unique(part[species_name]['mass'])
            if dark_lowres_masses.size > 9:
                self.say('! warning: {} different masses of low-resolution dark matter'.format(
                         dark_lowres_masses.size))

            if separate_dark_lowres and dark_lowres_masses.size > 1:
                self.say('* separating low-res dark-matter by mass into separate dictionaries')
                dark_lowres = {}
                for prop in part[species_name]:
                    dark_lowres[prop] = np.array(part[species_name][prop])

                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec = 'dark.{}'.format(dark_i + 2)

                    part[spec] = ParticleDictionaryClass()

                    for prop in dark_lowres:
                        part[spec][prop] = dark_lowres[prop][spec_indices]
                    self.say('{}: {} particles'.format(spec, spec_indices.size))

                del(spec_indices)
                print()

        if sort_dark_by_id:
            # order dark-matter particles by id - should be conserved across snapshots
            self.say('* sorting the following dark particles by id:')
            for spec in part:
                if 'dark' in spec and 'id' in part[spec]:
                    indices_sorted = np.argsort(part[spec]['id'])
                    self.say('{}: {} particles'.format(spec, indices_sorted.size))
                    for prop in part[spec]:
                        part[spec][prop] = part[spec][prop][indices_sorted]
            del(indices_sorted)
            print()

        # apply unit conversions
        for spec in part:
            if 'position' in part[spec]:
                # convert to [kpc comoving]
                part[spec]['position'] /= header['hubble']

            if 'mass' in part[spec]:
                # convert to [M_sun]
                part[spec]['mass'] *= 1e10 / header['hubble']

            if 'bh.mass' in part[spec]:
                # convert to [M_sun]
                part[spec]['bh.mass'] *= 1e10 / header['hubble']

            if 'velocity' in part[spec]:
                # convert to [km / s physical]
                part[spec]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'density' in part[spec]:
                # convert to [M_sun / kpc^3 physical]
                part[spec]['density'] *= (
                    1e10 / header['hubble'] / (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec]:
                # convert to [pc physical]
                part[spec]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                # convert to Plummer softening - 2.8 is valid for cubic spline
                # alternately, to convert to Gaussian scale length, divide by 2
                part[spec]['smooth.length'] /= 2.8

            if 'form.scalefactor' in part[spec]:
                if header['is.cosmological']:
                    pass
                else:
                    part[spec]['form.scalefactor'] /= header['hubble']  # convert to [Gyr]

            if 'temperature' in part[spec]:
                # convert from [(km / s) ^ 2] to [Kelvin]
                # ignore small corrections from elements beyond He
                helium_mass_fracs = part[spec]['massfraction'][:, 1]
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + part[spec]['electron.fraction'])
                molecular_weights = mus * ut.const.proton_mass
                part[spec]['temperature'] *= (
                    ut.const.centi_per_kilo ** 2 * (self.gas_eos - 1) * molecular_weights /
                    ut.const.boltzmann)
                del(helium_mass_fracs, ys_helium, mus, molecular_weights)

            if 'potential' in part[spec]:
                # convert from [km / s^2 comoving] to [km / s^2 physical]
                part[spec]['potential'] = part[spec]['potential'] / header['scalefactor']

        # renormalize so potential max = 0
        renormalize_potential = False
        if renormalize_potential:
            potential_max = 0
            for spec in part:
                if part[spec]['potential'].max() > potential_max:
                    potential_max = part[spec]['potential'].max()
            for spec in part:
                part[spec]['potential'] -= potential_max

        # sub-sample particles, for smaller memory
        if particle_subsample_factor > 1:
            self.say('* periodically subsampling all particles by factor = {}'.format(
                     particle_subsample_factor), end='\n\n')
            for spec in part:
                for prop in part[spec]:
                    part[spec][prop] = part[spec][prop][::particle_subsample_factor]

    def get_snapshot_file_name(self, directory, snapshot_index):
        '''
        Get name (with relative path) of file to read in.
        If multiple files per snapshot, get name of 0th one.

        Parameters
        ----------
        directory: string : directory to check for files
        snapshot_index : int : index of snapshot

        Returns
        -------
        file name (with relative path): string
        '''
        directory = ut.io.get_path(directory)

        path_names, file_indices = ut.io.get_file_names(
            directory + self.snapshot_name_base, (int, float))

        if snapshot_index < 0:
            snapshot_index = file_indices[snapshot_index]  # allow negative indexing of snapshots
        elif snapshot_index not in file_indices:
            raise ValueError('cannot find snapshot index = {} in: {}'.format(
                             snapshot_index, path_names))

        path_name = path_names[np.where(file_indices == snapshot_index)[0][0]]

        if self.file_extension in path_name:
            # got actual file, so good to go
            path_file_name = path_name
        else:
            # got snapshot directory with multiple files, return only 0th one
            path_file_names = ut.io.get_file_names(path_name + '/' + self.snapshot_name_base)
            if len(path_file_names) and '.0.' in path_file_names[0]:
                path_file_name = path_file_names[0]
            else:
                raise ValueError('cannot find 0th snapshot file in ' + path_file_names)

        return path_file_name

    def read_snapshot_times(self, directory='.'):
        '''
        Read snapshot file that contains scale-factors[, redshifts, times, time spacings].
        Return as dictionary.

        Parameters
        ----------
        directory : string : directory of snapshot time file

        Returns
        -------
        Snapshot : dictionary of snapshot information
        '''
        directory = ut.io.get_path(directory)

        Snapshot = ut.simulation.SnapshotClass()

        try:
            try:
                Snapshot.read_snapshots('snapshot_times.txt', directory)
            except IOError:
                Snapshot.read_snapshots('snapshot_scale-factors.txt', directory)
        except Exception:
            raise IOError('cannot find file of snapshot times in {}'.format(directory))

        self.is_first_print = True

        return Snapshot

    def get_cosmology(
        self, directory='.', omega_lambda=None, omega_matter=None, omega_baryon=None, hubble=None,
        sigma_8=None, n_s=None):
        '''
        Get cosmological parameters, stored in Cosmology class.
        Read cosmological parameters from MUSIC initial condition config file.
        If cannot find file, assume AGORA cosmology as default.

        Parameters
        ----------
        directory : string : directory of simulation (where directory of initial conditions is)

        Returns
        -------
        Cosmology : cosmology class, which also stores cosmological parameters
        '''
        def get_check_value(line, value_test=None):
            frac_dif_max = 0.01
            value = float(line.split('=')[-1].strip())
            if 'h0' in line:
                value /= 100
            if value_test is not None:
                frac_dif = np.abs((value - value_test) / value)
                if frac_dif > frac_dif_max:
                    print('! read {}, but previously assigned = {}'.format(line, value_test))
            return value

        if directory:
            # find MUSIC file, assuming named *.conf
            try:
                file_name_find = ut.io.get_path(directory) + '*/*.conf'
                file_name = ut.io.get_file_names(file_name_find)[0]
                self.say('* read cosmological parameters from: {}\n'.format(
                    file_name.strip('./')))
                # read cosmological parameters
                with open(file_name, 'r') as file_in:
                    for line in file_in:
                        line = line.lower().strip().strip('\n')  # ensure lowercase for safety
                        if 'omega_l' in line:
                            omega_lambda = get_check_value(line, omega_lambda)
                        elif 'omega_m' in line:
                            omega_matter = get_check_value(line, omega_matter)
                        elif 'omega_b' in line:
                            omega_baryon = get_check_value(line, omega_baryon)
                        elif 'h0' in line:
                            hubble = get_check_value(line, hubble)
                        elif 'sigma_8' in line:
                            sigma_8 = get_check_value(line, sigma_8)
                        elif 'nspec' in line:
                            n_s = get_check_value(line, n_s)

            except ValueError:
                self.say('! cannot find MUSIC config file: {}'.format(file_name_find.strip('./')))

        # AGORA box (use as default, if cannot find MUSIC config file)
        if omega_baryon is None or sigma_8 is None or n_s is None:
            self.say('! missing cosmological parameters, assuming values from AGORA box:')
            if omega_baryon is None:
                omega_baryon = 0.0455
                self.say('assuming omega_baryon = {}'.format(omega_baryon))
            if sigma_8 is None:
                sigma_8 = 0.807
                self.say('assuming sigma_8 = {}'.format(sigma_8))
            if n_s is None:
                n_s = 0.961
                self.say('assuming n_s = {}'.format(n_s))
            self.say('')

        Cosmology = ut.cosmology.CosmologyClass(
            omega_lambda, omega_matter, omega_baryon, hubble, sigma_8, n_s)

        return Cosmology

    def check_properties(self, part):
        '''
        Checks sanity of particle properties, print warning if they are outside given limits.

        Parameters
        ----------
        part : dict : catalog of particles
        '''
        # limits of sanity
        prop_limit_dict = {
            'id': [0, 4e9],
            'id.child': [0, 4e9],
            'id.generation': [0, 4e9],
            'position': [0, 1e6],  # [kpc comoving]
            'velocity': [-1e5, 1e5],  # [km / s]
            'mass': [9, 3e10],  # [M_sun]
            'potential': [-1e9, 1e9],  # [M_sun]
            'temperature': [3, 1e9],  # [K]
            'density': [0, 1e14],  # [M_sun/kpc^3]
            'smooth.length': [0, 1e9],  # [kpc physical]
            'hydrogen.neutral.fraction': [0, 1],
            'sfr': [0, 1000],  # [M_sun/yr]
            'massfraction': [0, 1],
            'form.scalefactor': [0, 1],
        }

        mass_factor_wrt_median = 4  # mass should not vary by more than this!

        self.say('* checking sanity of particle properties')

        for spec in part:
            for prop in [k for k in prop_limit_dict if k in part[spec]]:
                if (part[spec][prop].min() < prop_limit_dict[prop][0] or
                        part[spec][prop].max() > prop_limit_dict[prop][1]):
                    self.say(
                        '! warning: {} {} [min, max] = [{}, {}]'.format(
                            spec, prop,
                            ut.io.get_string_from_numbers(part[spec][prop].min(), 3),
                            ut.io.get_string_from_numbers(part[spec][prop].max(), 3))
                    )
                elif prop is 'mass' and spec in ['star', 'gas', 'dark']:
                    m_min = np.median(part[spec][prop]) / mass_factor_wrt_median
                    m_max = np.median(part[spec][prop]) * mass_factor_wrt_median
                    if part[spec][prop].min() < m_min or part[spec][prop].max() > m_max:
                        self.say(
                            '! warning: {} {} [min, med, max] = [{}, {}, {}]'.format(
                                spec, prop,
                                ut.io.get_string_from_numbers(part[spec][prop].min(), 3),
                                ut.io.get_string_from_numbers(np.median(part[spec][prop]), 3),
                                ut.io.get_string_from_numbers(part[spec][prop].max(), 3))
                        )

        print()

    def assign_center(self, part, method='center-of-mass', compare_centers=False):
        '''
        Assign center position [kpc comoving] and velocity [km / s physical] to galaxy/halo,
        using stars for baryonic simulation or dark matter for dark matter simulation.

        Parameters
        ----------
        part : dict : catalog of particles
        method : string : method of centering: 'center-of-mass', 'potential'
        compare_centers : boolean : whether to compare centers via center-of-mass v potential
        '''
        if 'star' in part and 'position' in part['star'] and len(part['star']['position']):
            spec_for_center = 'star'
            velocity_radius_max = 15
        elif 'dark' in part and 'position' in part['dark'] and len(part['dark']['position']):
            spec_for_center = 'dark'
            velocity_radius_max = 30
        else:
            self.say('! catalog not contain star or dark particles, so cannot assign center')
            return

        self.say('* assigning center of galaxy/halo:')

        if 'position' in part[spec_for_center]:
            # assign to overall dictionary
            part.center_position = ut.particle.get_center_position(
                part, spec_for_center, method, compare_centers=compare_centers)
            # assign to each species dictionary
            for spec in part:
                part[spec].center_position = part.center_position

            self.say('position = (', end='')
            ut.io.print_array(part.center_position, '{:.3f}', end='')
            print(') [kpc comoving]')

        if 'velocity' in part[spec_for_center]:
            # assign to overall dictionary
            part.center_velocity = ut.particle.get_center_velocity(
                part, spec_for_center, velocity_radius_max, part.center_position)
            # assign to each species dictionary
            for spec in part:
                part[spec].center_velocity = part.center_velocity

            self.say('velocity = (', end='')
            ut.io.print_array(part.center_velocity, '{:.1f}', end='')
            print(') [km / s]')

        print()

    def assign_principal_axes(self, part, distance_max=30, mass_percent=90):
        '''
        Assign principal axes (rotation vectors defined by moment of inertia tensor) to galaxy/halo,
        using stars for a baryonic simulation.

        Parameters
        ----------
        part : dict : catalog of particles
        distance_max : float : maximum distance to select particles [kpc physical]
        mass_percent : float : keep particles within the distance that encloses mass percent
            [0, 100] of all particles within distance_max
        '''
        species_name = 'star'
        if species_name not in part or not len(part[species_name]['position']):
            self.say('! catalog not contain star particles, so cannot assign principal axes')
            return

        self.say('* assigning principal axes of galaxy/halo:')

        rotation_vectors, _eigen_values, axes_ratios = ut.particle.get_principal_axes(
            part, species_name, distance_max, mass_percent, scalarize=True, print_results=False)

        part.principal_axes_vectors = rotation_vectors
        part.principal_axes_ratios = axes_ratios
        for spec in part:
            part[spec].principal_axes_vectors = rotation_vectors
            part[spec].principal_axes_ratios = axes_ratios

        self.say('axis ratios: min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
                 axes_ratios[0], axes_ratios[1], axes_ratios[2]))

        print()

    def assign_orbit(
        self, part, species=['star'], center_position=None, center_velocity=None,
        include_hubble_flow=True):
        '''
        Assign derived orbital properties wrt single center to species.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        species : string or list : particle species to compute
        center_position : array : center position to use
        center_velocity : array : center velocity to use
        include_hubble_flow : boolean : whether to include hubble flow
        '''
        species = ut.particle.parse_species(part, species)

        self.say('* assigning orbital properties wrt galaxy/halo to {}'.format(species))

        orb = ut.particle.get_orbit_dictionary(
            part, species, center_position, center_velocity,
            include_hubble_flow=include_hubble_flow, scalarize=False)

        for spec in species:
            for prop in orb[spec]:
                part[spec]['host.' + prop] = orb[spec][prop]

    # write to file ----------
    def rewrite_snapshot(
        self, species='gas', action='delete', value_adjust=None,
        snapshot_value_kind='redshift', snapshot_value=0,
        simulation_directory='.', snapshot_directory='output/'):
        '''
        Read snapshot file[s].
        Rewrite, deleting given species.

        Parameters
        ----------
        species : string or list : name[s] of particle species to delete:
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution
            'star' = stars
            'blackhole' = black holes
            'bulge' or 'disk' = stars for non-cosmological run
        action : string : what to do to snapshot file: 'delete', 'velocity'
        value_adjust : float : value by which to adjust property (if not deleting)
        snapshot_value_kind : string : input snapshot number kind: index, redshift
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        '''
        if np.isscalar(species):
            species = [species]  # ensure is list

        ## read information about snapshot times ----------
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)
        snapshot_index = Snapshot.parse_snapshot_value(snapshot_value_kind, snapshot_value)

        file_name = self.get_snapshot_file_name(snapshot_directory, snapshot_index)
        self.say('* read header from: ' + file_name.replace('./', ''), end='\n\n')

        ## read header ----------
        # open snapshot file and parse header
        with h5py.File(file_name, 'r+') as file_in:
            header = file_in['Header'].attrs  # load header dictionary

            ## read and delete input species ----------
            for file_i in range(header['NumFilesPerSnapshot']):
                # open i'th of multiple files for snapshot
                file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
                file_in = h5py.File(file_name_i, 'r+')

                self.say('read particles from: ' + file_name_i.split('/')[-1])

                if 'delete' in action:
                    part_number_in_file = header['NumPart_ThisFile']
                    part_number = header['NumPart_Total']

                # read and delete particle properties
                for _spec_i, spec in enumerate(species):
                    spec_id = self.species_dict[spec]
                    spec_in = 'PartType' + str(spec_id)
                    self.say('adjusting species = {}'.format(spec))

                    if 'delete' in action:
                        self.say('deleting species = {}'.format(spec))

                        # zero numbers in header
                        part_number_in_file[spec_id] = 0
                        part_number[spec_id] = 0

                        # delete properties
                        #for prop in file_in[spec_in]:
                        #    del(file_in[spec_in + '/' + prop])
                        #    self.say('  deleting {}'.format(prop))

                        del(file_in[spec_in])

                    elif 'velocity' in action and value_adjust:
                        dimension_index = 2  # boost velocity along z-axis
                        self.say('  boosting velocity along axis.{} by {:.1f} km/s'.format(
                                 dimension_index, value_adjust))
                        velocities = file_in[spec_in + '/' + 'Velocities']
                        scalefactor = 1 / (1 + header['Redshift'])
                        velocities[:, 2] += value_adjust / np.sqrt(scalefactor)
                        #file_in[spec_in + '/' + 'Velocities'] = velocities

                    print()

                if 'delete' in action:
                    header['NumPart_ThisFile'] = part_number_in_file
                    header['NumPart_Total'] = part_number


Read = ReadClass()


#===================================================================================================
# write Efficient Binary Format (EBF) file for Galaxia
#===================================================================================================
def write_ebf_file(part=None, distance_limits=[0, 300]):
    '''
    Take Gizmo snapshot, write stars to Efficient Binary Format (EBF) file to use in Galaxia.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    distance_limits : list : min and max distance from center to keep particles [kpc physical]
    '''
    import ebf  # @UnresolvedImport

    if part is None:
        part = Read.read_snapshots('star', 'redshift', 0, element_indices=None, assign_center=True)

    Read.assign_orbit(part, 'star')

    # get particles within the selection region
    part_indices = ut.particle.get_indices_within_distances(part, 'star', distance_limits)

    # determine the plane of the stellar disk
    distance_max = 12  # [kpc physical]
    eigen_vectors, _eigen_values, _axis_ratios = ut.coordinate.get_principal_axes(
        part['star']['host.distance.vector'][part['star']['host.distance'] < distance_max])

    # rotate/shift phase-space coordinates to be wrt galaxy center
    positions = ut.coordinate.get_coordinates_rotated(
        part['star']['host.distance.vector'][part_indices], eigen_vectors)
    velocities = ut.coordinate.get_coordinates_rotated(
        part['star']['host.velocity.vector'][part_indices], eigen_vectors)

    file_name = 'galaxia_stars_{}.ebf'.format(part.snapshot['index'])

    # phase-space coordinates
    ebf.write(file_name, '/pos3', positions, 'w')  # [kpc comoving}
    ebf.write(file_name, '/vel3', velocities, 'a')  # [km/s]

    # stellar age (time since formation) [Gyr]
    ebf.write(file_name, '/age', part['star'].prop('age', part_indices), 'a')
    ebf.write(file_name, '/mass', part['star'].prop('mass', part_indices), 'a')  # [Msun]

    # TODO: store vector with all metallicities (10xN array)
    ebf.write(file_name, '/feh', part['star'].prop('metallicity.iron', part_indices), 'a')
    ebf.write(
        file_name, '/alpha',
        part['star'].prop('metallicity.magnesium - metallicity.iron', part_indices), 'a')

    # id of star particles
    ebf.write(file_name, '/parentid', part['star']['id'][part_indices], 'a')
