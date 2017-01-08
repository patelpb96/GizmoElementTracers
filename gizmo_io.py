'''
Read Gizmo snapshots.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatibility
import collections
import numpy as np
from numpy import log10, Inf  # @UnusedImport
import h5py
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
    # use to translate between element name and index in element table
    element_dict = collections.OrderedDict()
    element_dict['metals'] = element_dict['total'] = 0
    element_dict['helium'] = element_dict['he'] = 1
    element_dict['carbon'] = element_dict['c'] = 2
    element_dict['nitrogen'] = element_dict['n'] = 3
    element_dict['oxygen'] = element_dict['o'] = 4
    element_dict['neon'] = element_dict['ne'] = 5
    element_dict['magnesium'] = element_dict['mg'] = 6
    element_dict['silicon'] = element_dict['si'] = 7
    element_dict['sulphur'] = element_dict['s'] = 8
    element_dict['calcium'] = element_dict['ca'] = 9
    element_dict['iron'] = element_dict['fe'] = 10

    element_pointer = np.arange(len(element_dict))  # use if read only subset of elements

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
                prop_values = np.float(prop_values, dtype=prop_values.dtype)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        ## parsing specific to this catalog ----------
        if 'form.' in property_name or property_name == 'age':
            if property_name == 'age' or ('time' in property_name and 'lookback' in property_name):
                values = self.snapshot['time'] - self.prop('form.time', indices)
            elif 'time' in property_name:
                values = self.Cosmology.get_time(
                    self.prop('form.scalefactor', indices), 'scalefactor')
            elif 'redshift' in property_name:
                values = 1 / self.prop('form.scalefactor', indices) - 1

            return values

        if 'number.density' in property_name or 'density.number' in property_name:
            values = (self.prop('density', indices) * ut.const.proton_per_sun *
                      ut.const.kpc_per_cm ** 3)

            if '.hydrogen' in property_name:
                # number density of hydrogen, using actual hydrogen mass of each particle [cm ^ -3]
                values = values * self.prop('massfraction.hydrogen', indices)
            else:
                # number density of 'hydrogen', assuming solar metallicity for particles [cm ^ -3]
                values = values * ut.const.sun_hydrogen_mass_fraction

            return values

        if 'mass.' in property_name:
            # mass of individual element
            values = (self.prop('mass', indices) *
                      self.prop(property_name.replace('mass.', 'massfraction.'), indices))

            if property_name == 'mass.hydrogen.neutral':
                # mass of neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices)

            return values

        # element string -> index conversion
        if 'massfraction.' in property_name or 'metallicity.' in property_name:
            if 'massfraction.hydrogen' in property_name:
                # special case: mass fraction of hydrogen (excluding helium and metals)
                values = (1 - self.prop('massfraction', indices)[:, 0] -
                          self.prop('massfraction', indices)[:, 1])

                if property_name == 'massfraction.hydrogen.neutral':
                    # mass fraction of neutral hydrogen (excluding helium, metals, and ionized)
                    values = values * self.prop('hydrogen.neutral.fraction', indices)

                return values

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

        self.eos = 5 / 3  # gas equation of state

        # snapshot file does not contain these cosmological parameters, so have to set manually
        # these are from AGORA cosmology
        self.omega_baryon = 0.0455
        self.sigma_8 = 0.807
        self.n_s = 0.961
        self.w = -1.0

        # create ordered dictionary to convert particle species name to its id,
        # set all possible species, and set the order in which to read species
        self.species_dict = collections.OrderedDict()
        # dark-matter species
        self.species_dict['dark'] = 1  # dark matter at highest resolution
        self.species_dict['dark.2'] = 2  # lower-resolution dark matter across all resolutions
        self.species_dict['dark.3'] = 3
        #self.species_dict['dark.4'] = 5
        # baryon species
        self.species_dict['gas'] = 0
        self.species_dict['star'] = 4
        # other - these ids overlap with above, so have to comment in if using them
        self.species_dict['blackhole'] = 5
        #self.species_dict['bulge'] = 2
        #self.species_dict['disk'] = 3

        self.species_names_all = list(self.species_dict.keys())
        self.species_names_read = list(self.species_dict.keys())

    def read_snapshots(
        self, species_names='all',
        snapshot_number_kind='index', snapshot_numbers=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        property_names='all', element_indices=None, particle_subsample_factor=0,
        separate_dark_lowres=True, sort_dark_by_id=False, force_float32=False, assign_center=True,
        check_sanity=True):
        '''
        Read given properties for given particle species from simulation snapshot file[s].
        Return as dictionary class.

        Parameters
        ----------
        species_names : string or list : name[s] of particle species:
            'all' = all species in file
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution
            'star' = stars
            'blackhole' = black holes, if run contains them
            'bulge' or 'disk' = stars for non-cosmological run
        snapshot_number_kind : string :
            input snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_numbers : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : string : directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        property_names : string or list : name[s] of particle properties to read - options:
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
        check_sanity : boolean : whether to check sanity of particle properties after read in

        Returns
        -------
        parts : dictionary or list thereof :
            if single snapshot, return as dictionary, else if multiple snapshots, return as list
        '''
        # parse input species list
        if species_names == 'all' or species_names == ['all'] or not species_names:
            # read all species in snapshot
            species_names = self.species_names_all
        else:
            # read subsample of species in snapshot
            if np.isscalar(species_names):
                species_names = [species_names]  # ensure is list
            # check if input species names are valid
            for spec_name in list(species_names):
                if spec_name not in self.species_dict:
                    species_names.remove(spec_name)
                    self.say('! not recognize input species = {}'.format(spec_name))
        self.species_names_read = species_names

        # read information about snapshot times
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)

        snapshot_numbers = ut.array.arrayize(snapshot_numbers)

        parts = []  # list to store particle dictionaries

        # read all input snapshots
        for snapshot_number in snapshot_numbers:
            snapshot_index = Snapshot.parse_snapshot_number(snapshot_number_kind, snapshot_number)

            # read header from snapshot file
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory, simulation_name)

            # read particles from snapshot file[s]
            part = self.read_particles(
                'index', snapshot_index, simulation_directory, snapshot_directory, property_names,
                element_indices, force_float32, header)

            # assign cosmological parameters
            if header['is.cosmological']:
                # for cosmological parameters not in header, use values set above
                Cosmology = ut.cosmology.CosmologyClass(
                    header['hubble'], header['omega_matter'], header['omega_lambda'],
                    self.omega_baryon, self.sigma_8, self.n_s, self.w)

            # adjust properties for each species
            self.adjust_particles(
                part, header, particle_subsample_factor, separate_dark_lowres, sort_dark_by_id)

            if check_sanity:
                self.check_sanity(part)

            # assign auxilliary information to particle dictionary class
            # store header dictionary
            part.info = header
            for spec_name in part:
                part[spec_name].info = part.info

            # store cosmology class
            part.Cosmology = Cosmology
            for spec_name in part:
                part[spec_name].Cosmology = part.Cosmology

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
            for spec_name in part:
                part[spec_name].snapshot = part.snapshot

            # store information on all snapshot times - may or may not be initialized
            part.Snapshot = Snapshot

            # arrays to store center position and velocity
            part.center_position = []
            part.center_velocity = []
            if assign_center:
                self.assign_center(part)

            # if read only 1 snapshot, return as particle dictionary instead of list
            if len(snapshot_numbers) == 1:
                parts = part
            else:
                parts.append(part)
                print()

        return parts

    def read_simulations(
        self, simulation_directories=[], species_names='all', redshift=0,
        property_names='all', element_indices=[0, 1, 6, 10], force_float32=True):
        '''
        Read snapshots at the same redshift from different simulations.
        Return as list of dictionaries.

        Parameters
        ----------
        directories : list or list of lists :
            list of simulation directories, or list of pairs of directory + simulation name
        species_names : string or list : particle species to read
        redshift : float
        property_names : string or list : names of properties to read
        element_indices : int or list : indices of elements to read
        force_float32 : boolean : whether to force positions to be 32-bit

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
        bad_snapshot_number = 0
        for directory, simulation_name in simulation_directories:
            _header = self.read_header(
                'redshift', redshift, directory, simulation_name=simulation_name)
            try:
                _header = self.read_header(
                    'redshift', redshift, directory, simulation_name=simulation_name)
            except:
                self.say('! could not read snapshot header at z = {:.3f} in {}'.format(
                         redshift, directory))
                bad_snapshot_number += 1

        if bad_snapshot_number:
            self.say('\n! could not read {} snapshots'.format(bad_snapshot_number))
            return

        parts = []
        directories_read = []
        for directory, simulation_name in simulation_directories:
            try:
                part = self.read_snapshots(
                    species_names, 'redshift', redshift, directory, simulation_name=simulation_name,
                    property_names=property_names, element_indices=element_indices,
                    force_float32=force_float32)

                if 'velocity' in property_names:
                    self.assign_orbit(part, 'gas')

                parts.append(part)
                directories_read.append(directory)
            except:
                self.say('! could not read snapshot at z = {:.3f} in {}'.format(
                         redshift, directory))

        if not len(parts):
            self.say('! could not read any snapshots at z = {:.3f}'.format(redshift))
            return

        if 'mass' in property_names and 'star' in part:
            for part, directory in zip(parts, directories_read):
                print('{}: star.mass = {:.3e}'.format(directory, part['star']['mass'].sum()))

        return parts

    def get_file_name(self, directory, snapshot_index):
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

        try:
            path_names, file_indices = ut.io.get_file_names(
                directory + self.snapshot_name_base, int)
        except:
            path_names, file_indices = ut.io.get_file_names(
                directory + self.snapshot_name_base, float)

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
            except:
                Snapshot.read_snapshots('snapshot_scale-factors.txt', directory)
        except:
            raise ValueError('cannot find file of snapshot times in {}'.format(directory))

        self.is_first_print = True

        return Snapshot

    def read_header(
        self, snapshot_number_kind='index', snapshot_number=600, simulation_directory='.',
        snapshot_directory='output/', simulation_name=''):
        '''
        Read header from snapshot file.

        Parameters
        ----------
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_number : int or float : index (number) of snapshot file
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

        if snapshot_number_kind != 'index':
            Snapshot = self.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_number(snapshot_number_kind, snapshot_number)
        else:
            snapshot_index = snapshot_number

        # get snapshot file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)

        self.say('* reading header from: {}'.format(file_name.replace('./', '')), end='\n')

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            header_in = file_in['Header'].attrs  # load header dictionary

            for prop_name_in in header_in:
                prop_name = header_dict[prop_name_in]
                header[prop_name] = header_in[prop_name_in]  # transfer to custom header dict

        # determine whether simulation is cosmological
        if (0 < header['hubble'] < 1 and 0 < header['omega_matter'] < 1 and
                0 < header['omega_lambda'] < 1):
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

        # keep only species that have any particles
        read_particle_number = 0
        for spec_name in self.species_names_all:
            spec_id = self.species_dict[spec_name]
            if spec_name in self.species_names_read:
                self.say('species = {:9s} (id = {}): {} particles'.format(
                         spec_name, spec_id, header['particle.numbers.total'][spec_id]))

                if header['particle.numbers.total'][spec_id] > 0:
                    read_particle_number += header['particle.numbers.total'][spec_id]
                else:
                    self.species_names_read.remove(spec_name)

        if read_particle_number <= 0:
            raise ValueError('! snapshot file[s] contain no particles of species = {}'.format(
                             self.species_names_read))

        # check if simulation contains baryons
        header['has.baryons'] = False
        for spec_name in self.species_names_all:
            if spec_name in ['gas', 'star', 'disk', 'bulge']:
                spec_id = self.species_dict[spec_name]
                if header['particle.numbers.total'][spec_id] > 0:
                    header['has.baryons'] = True
                    break

        # assign simulation name
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.strip('/')
        header['simulation.name'] = simulation_name

        header['catalog.kind'] = 'particle'

        print()

        return header

    def read_particles(
        self, snapshot_number_kind='index', snapshot_number=600, simulation_directory='.',
        snapshot_directory='output/', property_names='all', element_indices=None,
        force_float32=False, header=None):
        '''
        Read particles from snapshot file[s].

        Parameters
        ----------
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_number : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        property_names : string or list : name[s] of particle properties to read - options:
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
        # if comment out any property, will not read it
        property_dict = {
            ## all particles ----------
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',
            ## particles with adaptive smoothing
            'AGS-Softening': 'smooth.length',  # for gas, this is same as SmoothingLength

            ## gas particles ----------
            'InternalEnergy': 'temperature',
            'Density': 'density',
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
        if property_names == 'all' or property_names == ['all'] or not property_names:
            property_names = list(property_dict.keys())
        else:
            if np.isscalar(property_names):
                property_names = [property_names]  # ensure is list
            # make safe list of property names to read in
            property_names_temp = []
            for prop_name in list(property_names):
                prop_name = str.lower(prop_name)
                if 'massfraction' in prop_name or 'metallicity' in prop_name:
                    prop_name = 'massfraction'  # this has several aliases, so ensure default name
                for prop_name_in in property_dict:
                    if (prop_name == str.lower(prop_name_in) or
                            prop_name == str.lower(property_dict[prop_name_in])):
                        property_names_temp.append(prop_name_in)
            property_names = property_names_temp
            del(property_names_temp)

        if 'InternalEnergy' in property_names:
            # need helium mass fraction and electron fraction to compute temperature
            for prop_name in ['ElectronAbundance', 'Metallicity']:
                if prop_name not in property_names:
                    property_names.append(prop_name)

        # parse other input values
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        if snapshot_number_kind != 'index':
            Snapshot = self.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_number(snapshot_number_kind, snapshot_number)
        else:
            snapshot_index = snapshot_number

        if not header:
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory)

        # get snapshot file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            if header['file.number.per.snapshot'] == 1:
                self.say('* reading particles from: {}'.format(file_name.replace('./', '')))
            else:
                self.say('* reading particles')

            # initialize arrays to store each property for each species
            for spec_name in self.species_names_read:
                spec_id = self.species_dict[spec_name]
                part_number_tot = header['particle.numbers.total'][spec_id]

                # add species to particle dictionary
                part[spec_name] = ParticleDictionaryClass()

                # set element pointers if reading only subset of elements
                if (element_indices is not None and element_indices != [] and
                        element_indices != 'all'):
                    if np.isscalar(element_indices):
                        element_indices = [element_indices]
                    for element_i, element_index in enumerate(element_indices):
                        part[spec_name].element_pointer[element_index] = element_i

                # check if snapshot file happens not to have particles of this species
                if part_numbers_in_file[spec_id] <= 0:
                    # this scenario should occur only for multi-file snapshot
                    if header['file.number.per.snapshot'] == 1:
                        raise ValueError(
                            '! no {} particles in single-file snapshot'.format(spec_name))

                    # need to read in other snapshot files until find one with particles of species
                    for file_i in range(1, header['file.number.per.snapshot']):
                        file_name = file_name.replace('.0.', '.{}.'.format(file_i))
                        # open snapshot file
                        with h5py.File(file_name, 'r') as file_in_i:
                            part_numbers_in_file_i = file_in_i['Header'].attrs['NumPart_ThisFile']
                            if part_numbers_in_file_i[spec_id] > 0:
                                # found one!
                                part_in = file_in_i['PartType' + str(spec_id)]
                                break
                    else:
                        # tried all files and still did not find particles of species
                        raise ValueError(
                            '! no {} particles in any snapshot files'.format(spec_name))
                else:
                    part_in = file_in['PartType' + str(spec_id)]

                prop_names_print = []
                ignore_flag = False
                for prop_name_in in part_in:
                    if prop_name_in in property_names:
                        prop_name = property_dict[prop_name_in]

                        # determine shape of property array
                        if len(part_in[prop_name_in].shape) == 1:
                            prop_shape = part_number_tot
                        elif len(part_in[prop_name_in].shape) == 2:
                            prop_shape = [part_number_tot, part_in[prop_name_in].shape[1]]
                            if (prop_name_in == 'Metallicity' and element_indices is not None and
                                    element_indices != 'all'):
                                prop_shape = [part_number_tot, len(element_indices)]

                        # determine data type to store
                        prop_in_dtype = part_in[prop_name_in].dtype
                        if force_float32 and prop_in_dtype == 'float64':
                            prop_in_dtype = np.float32

                        # initialize to -1's
                        part[spec_name][prop_name] = np.zeros(prop_shape, prop_in_dtype) - 1

                        if prop_name == 'id':
                            # initialize so calling an un-itialized value leads to error
                            part[spec_name][prop_name] -= part_number_tot

                        if prop_name_in in property_dict:
                            prop_names_print.append(property_dict[prop_name_in])
                        else:
                            prop_names_print.append(prop_name_in)
                    else:
                        ignore_flag = True

                if ignore_flag:
                    prop_names_print.sort()
                    self.say('reading {:6}: {}'.format(spec_name, prop_names_print))

                # special case: particle mass is fixed and given in mass array in header
                if 'Masses' in property_names and 'Masses' not in part_in:
                    prop_name = property_dict['Masses']
                    part[spec_name][prop_name] = np.zeros(part_number_tot, dtype=np.float32)

        ## read properties for each species ----------
        # initial particle indices to assign to each species from each file
        part_indices_lo = np.zeros(len(self.species_names_read), dtype=np.int64)

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
                for spec_i, spec_name in enumerate(self.species_names_read):
                    spec_id = self.species_dict[spec_name]
                    if part_numbers_in_file[spec_id] > 0:
                        part_in = file_in['PartType' + str(spec_id)]

                        part_index_lo = part_indices_lo[spec_i]
                        part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                        # check if mass of species is fixed, according to header mass array
                        if 'Masses' in property_names and header['particle.masses'][spec_id] > 0:
                            prop_name = property_dict['Masses']
                            part[spec_name][prop_name][
                                part_index_lo:part_index_hi] = header['particle.masses'][spec_id]

                        for prop_name_in in part_in:
                            if prop_name_in in property_names:
                                prop_name = property_dict[prop_name_in]
                                if len(part_in[prop_name_in].shape) == 1:
                                    part[spec_name][prop_name][
                                        part_index_lo:part_index_hi] = part_in[prop_name_in]
                                elif len(part_in[prop_name_in].shape) == 2:
                                    if (prop_name_in == 'Metallicity' and
                                            element_indices is not None and
                                            element_indices != 'all'):
                                        prop_in = part_in[prop_name_in][:, element_indices]
                                    else:
                                        prop_in = part_in[prop_name_in]

                                    part[spec_name][prop_name][
                                        part_index_lo:part_index_hi, :] = prop_in

                        part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        print()

        return part

    def adjust_particles(
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
        spec_name = 'dark.2'
        if spec_name in part and 'mass' in part[spec_name]:
            dark_lowres_masses = np.unique(part[spec_name]['mass'])
            if dark_lowres_masses.size > 9:
                self.say('! warning: {} different masses of low-resolution dark matter'.format(
                         dark_lowres_masses.size))

            if separate_dark_lowres and dark_lowres_masses.size > 1:
                self.say('* separating low-res dark-matter by mass into separate dictionaries')
                dark_lowres = {}
                for prop_name in part[spec_name]:
                    dark_lowres[prop_name] = np.array(part[spec_name][prop_name])

                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec_name = 'dark.{}'.format(dark_i + 2)

                    part[spec_name] = ParticleDictionaryClass()

                    for prop_name in dark_lowres:
                        part[spec_name][prop_name] = dark_lowres[prop_name][spec_indices]
                    self.say('{}: {} particles'.format(spec_name, spec_indices.size))

                del(spec_indices)
                print()

        if sort_dark_by_id:
            # order dark-matter particles by id - should be conserved across snapshots
            for spec_name in part:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    self.say('* sorting {:6} particles by id'.format(spec_name))
                    indices_sorted = np.argsort(part[spec_name]['id'])
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = part[spec_name][prop_name][indices_sorted]
            del(indices_sorted)
            print()

        # apply unit conversions
        for spec_name in part:
            if 'position' in part[spec_name]:
                # convert to [kpc comoving]
                part[spec_name]['position'] /= header['hubble']

            if 'mass' in part[spec_name]:
                # convert to [M_sun]
                part[spec_name]['mass'] *= 1e10 / header['hubble']

            if 'bh.mass' in part[spec_name]:
                # convert to [M_sun]
                part[spec_name]['bh.mass'] *= 1e10 / header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to [km / s physical]
                part[spec_name]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'density' in part[spec_name]:
                # convert to [M_sun / kpc^3 physical]
                part[spec_name]['density'] *= (
                    1e10 / header['hubble'] / (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to [pc physical]
                part[spec_name]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                # convert to Plummer softening. this value should be valid for most simulations
                part[spec_name]['smooth.length'] /= 2.8

            if 'form.scalefactor' in part[spec_name]:
                if header['is.cosmological']:
                    pass
                    # convert from units of scale-factor to [Gyr]
                    #part[spec_name]['form.scalefactor'] = Cosmology.get_time(
                    #    part[spec_name]['form.time'], 'scalefactor').astype(
                    #        part[spec_name]['form.time'].dtype)
                else:
                    part[spec_name]['form.scalefactor'] /= header['hubble']  # convert to [Gyr]

            if 'temperature' in part[spec_name]:
                # convert from [(km / s) ^ 2] to [Kelvin]
                # ignore small corrections from elements beyond He
                helium_mass_fracs = part[spec_name]['massfraction'][:, 1]
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + part[spec_name]['electron.fraction'])
                molecular_weights = mus * ut.const.proton_mass
                part[spec_name]['temperature'] *= (
                    ut.const.centi_per_kilo ** 2 * (self.eos - 1) * molecular_weights /
                    ut.const.boltzmann)
                del(helium_mass_fracs, ys_helium, mus, molecular_weights)

            if 'potential' in part[spec_name]:
                # convert from [km / s^2 comoving] to [km / s^2 physical]
                part[spec_name]['potential'] = part[spec_name]['potential'] / header['scalefactor']

        # renormalize so potential max = 0
        #potential_max = 0
        #for spec_name in part:
        #    if part[spec_name]['potential'].max() > potential_max:
        #        potential_max = part[spec_name]['potential'].max()
        #for spec_name in part:
        #    part[spec_name]['potential'] -= potential_max

        # sub-sample particles, for smaller memory
        if particle_subsample_factor > 1:
            self.say('* periodically subsampling all particles by factor = {}'.format(
                     particle_subsample_factor), end='\n\n')
            for spec_name in part:
                for prop_name in part[spec_name]:
                    part[spec_name][prop_name] = part[spec_name][prop_name][
                        ::particle_subsample_factor]

    def check_sanity(self, part):
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
            'mass': [10, 3e10],  # [M_sun]
            'potential': [-1e9, 1e9],  # [M_sun]
            'temperature': [3, 1e9],  # [K]
            'density': [0, 1e14],  # [M_sun/kpc^3]
            'smooth.length': [0, 1e9],  # [kpc physical]
            'hydrogen.neutral.fraction': [0, 1],
            'sfr': [0, 1000],  # [M_sun/yr]
            'massfraction': [0, 1],
            'form.scalefactor': [0, 1],
        }

        self.say('* checking sanity of particle properties')

        for spec_name in part:
            for prop_name in part[spec_name]:
                if prop_name in prop_limit_dict:
                    if (part[spec_name][prop_name].min() < prop_limit_dict[prop_name][0] or
                            part[spec_name][prop_name].max() > prop_limit_dict[prop_name][1]):
                        self.say('! warning: {} {} [min, max] = [{:.3f}, {:.3f}]'.format(
                                 spec_name, prop_name, part[spec_name][prop_name].min(),
                                 part[spec_name][prop_name].max()))
        print()

    def assign_center(self, part, method='center-of-mass', compare_centers=False):
        '''
        Assign center position [kpc comoving] and velocity [km / s physical] to galaxy/halo,
        using stars for hydro simulation or dark matter for dark matter simulation.

        Parameters
        ----------
        part : dict : catalog of particles
        method : string : method of centering: 'center-of-mass', 'potential'
        compare_centers : boolean : whether to compare centers via center-of-mass v potential
        '''
        if 'star' in part and len(part['star']['position']):
            spec_name = 'star'
            velocity_radius_max = 15
        elif 'dark' in part and len(part['dark']['position']):
            spec_name = 'dark'
            velocity_radius_max = 30
        else:
            self.say('! catalog not contain star or dark particles, skipping center finding')
            return

        self.say('* assigning center of galaxy/halo:')

        if 'position' in part[spec_name]:
            part.center_position = ut.particle.get_center_position(
                part, spec_name, method, compare_centers=compare_centers)
            self.say('position = (', end='')
            ut.io.print_array(part.center_position, '{:.3f}', end='')
            print(') [kpc comoving]')

        if 'velocity' in part[spec_name]:
            part.center_velocity = ut.particle.get_center_velocity(
                part, spec_name, velocity_radius_max, part.center_position)
            self.say('velocity = (', end='')
            ut.io.print_array(part.center_velocity, '{:.1f}', end='')
            print(') [km / s]')

        print()

    # write to file ----------
    def rewrite_snapshot(
        self, species_names='gas', action='delete', value_adjust=None,
        snapshot_number_kind='redshift', snapshot_number=0,
        simulation_directory='.', snapshot_directory='output/'):
        '''
        Read snapshot file[s].
        Rewrite, deleting given species.

        Parameters
        ----------
        species_names : string or list : name[s] of particle species to delete:
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution
            'star' = stars
            'blackhole' = black holes
            'bulge' or 'disk' = stars for non-cosmological run
        action : string : what to do to snapshot file: 'delete', 'velocity'
        value_adjust : float : value by which to adjust property (if not deleting)
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_number : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        '''
        if np.isscalar(species_names):
            species_names = [species_names]  # ensure is list

        ## read information about snapshot times ----------
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)
        snapshot_index = Snapshot.parse_snapshot_number(snapshot_number_kind, snapshot_number)

        # get file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)
        self.say('* reading header from: ' + file_name.replace('./', ''), end='\n\n')

        ## read header ----------
        # open snapshot file and parse header
        with h5py.File(file_name, 'r+') as file_in:
            header = file_in['Header'].attrs  # load header dictionary

            ## read and delete input species ----------
            for file_i in range(header['NumFilesPerSnapshot']):
                # open i'th of multiple files for snapshot
                file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
                file_in = h5py.File(file_name_i, 'r+')

                self.say('reading particles from: ' + file_name_i.split('/')[-1])

                if 'delete' in action:
                    part_number_in_file = header['NumPart_ThisFile']
                    part_number = header['NumPart_Total']

                # read and delete particle properties
                for _spec_i, spec_name in enumerate(species_names):
                    spec_id = self.species_dict[spec_name]
                    spec_name_in = 'PartType' + str(spec_id)
                    self.say('adjusting species = {}'.format(spec_name))

                    if 'delete' in action:
                        self.say('deleting species = {}'.format(spec_name))

                        # zero numbers in header
                        part_number_in_file[spec_id] = 0
                        part_number[spec_id] = 0

                        # delete properties
                        #for prop_name in file_in[spec_name_in]:
                        #    del(file_in[spec_name_in + '/' + prop_name])
                        #    self.say('  deleting {}'.format(prop_name))

                        del(file_in[spec_name_in])

                    elif 'velocity' in action and value_adjust:
                        dimen_index = 2  # boost velocity along z-axis
                        self.say('  boosting velocity along axis.{} by {:.1f} km/s'.format(
                                 dimen_index, value_adjust))
                        velocities = file_in[spec_name_in + '/' + 'Velocities']
                        scalefactor = 1 / (1 + header['Redshift'])
                        velocities[:, 2] += value_adjust / np.sqrt(scalefactor)
                        #file_in[spec_name_in + '/' + 'Velocities'] = velocities

                    print()

                if 'delete' in action:
                    header['NumPart_ThisFile'] = part_number_in_file
                    header['NumPart_Total'] = part_number

Read = ReadClass()


#===================================================================================================
# assign additional properties
#===================================================================================================
def assign_orbit(
    part, species=['star'], center_position=None, center_velocity=None, include_hubble_flow=True):
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

    orb = ut.particle.get_orbit_dictionary(
        part, species, center_position, center_velocity,
        include_hubble_flow=include_hubble_flow, scalarize=False)

    for spec_name in species:
        for prop in orb[spec_name]:
            part[spec_name]['host.' + prop] = orb[spec_name][prop]


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

    assign_orbit(part, 'star')

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
