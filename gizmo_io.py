'''
Read Gizmo snapshots.

Masses in [M_sun], positions in [kpc comoving], distances in [kpc physical].

@author: Andrew Wetzel
'''


# system ----
from __future__ import absolute_import, division, print_function
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
        ## parsing general to all catalogs ##
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

        ## parsing specific to this catalog ##
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

        self.species_names = list(self.species_dict.keys())

        self.read_snapshot = self.read_snapshots  # alias for backwards compatability

    def read_snapshots(
        self, species_names='all',
        snapshot_number_kind='index', snapshot_numbers=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        property_names='all', element_indices=[0, 1], particle_subsample_factor=0,
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
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_numbers : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : root directory of simulation
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
            species_names = list(self.species_dict.keys())
        else:
            # read subsample of species in snapshot
            if np.isscalar(species_names):
                species_names = [species_names]  # ensure is list
            # check if input species names are valid
            for spec_name in list(species_names):
                if spec_name not in self.species_dict:
                    species_names.remove(spec_name)
                    self.say('! not recognize input species = {}'.format(spec_name))
        self.species_names = species_names

        # read information about snapshot times
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)

        snapshot_numbers = ut.array.arrayize(snapshot_numbers)

        parts = []  # list to store particle dictionaries

        # read all input snapshot numbers of simulation
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
            for spec_name in self.species_names:
                part[spec_name].info = part.info

            # store cosmology class
            part.Cosmology = Cosmology
            for spec_name in self.species_names:
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
            for spec_name in self.species_names:
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

        self.say('reading header from: ' + file_name, end='\n')

        file_in = h5py.File(file_name, 'r')  # open hdf5 snapshot file
        header_in = file_in['Header'].attrs  # load header dictionary

        for prop_name_in in header_in:
            prop_name = header_dict[prop_name_in]
            header[prop_name] = header_in[prop_name_in]  # transfer to custom header dict

        file_in.close()

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
        particle_number_min = 0
        for spec_name in list(self.species_names):
            spec_id = self.species_dict[spec_name]
            self.say('  species = {:9s} (id = {}): {} particles'.format(
                     spec_name, spec_id, header['particle.numbers.total'][spec_id]))
            if header['particle.numbers.total'][spec_id] > 0:
                particle_number_min = header['particle.numbers.total'][spec_id]
            else:
                self.species_names.remove(spec_name)

        # check if simulation contains baryons
        if ('gas' not in self.species_names and 'star' not in self.species_names and
                'disk' not in self.species_names and 'bulge' not in self.species_names):
            header['has.baryons'] = False
        else:
            header['has.baryons'] = True

        # assign simulation name
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.strip('/')
        header['simulation.name'] = simulation_name

        header['catalog.kind'] = 'particle'

        if particle_number_min == 0:
            raise ValueError('! found no particles in file')

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
            ## all particles ##
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',
            ## particles with adaptive smoothing ##
            'AGS-Softening': 'smooth.length',  # for gas, this is same as SmoothingLength

            ## gas particles ##
            'InternalEnergy': 'temperature',
            'Density': 'density',
            'SmoothingLength': 'smooth.length',
            #'ArtificialViscosity': 'artificial.viscosity',
            # average free-electron number per proton, averaged over mass of gas particle
            'ElectronAbundance': 'electron.fraction',
            # fraction of hydrogen that is neutral (not ionized)
            'NeutralHydrogenAbundance': 'hydrogen.neutral.fraction',
            'StarFormationRate': 'sfr',  # [M_sun / yr]

            ## star/gas particles ##
            ## id.generation and id.child initialized to 0 for all gas particles
            ## each time a gas particle splits into two:
            ##   'self' particle retains id.child, other particle gets id.child += 2 ^ id.generation
            ##   both particles get id.generation += 1
            ## allows maximum of 30 generations, then restarts at 0
            ##   thus, particles with id.child > 2^30 are not unique anymore
            'ParticleChildIDsNumber': 'id.child',
            'ParticleIDGenerationNumber': 'id.generation',

            ## mass fraction of individual elements ##
            ## 0 = all metals (everything not H, He)
            ## 1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
            'Metallicity': 'massfraction',

            ## star particles ##
            ## 'time' when star particle formed
            ## for cosmological runs, = scale-factor; for non-cosmological runs, = time {Gyr/h}
            'StellarFormationTime': 'form.scalefactor',

            ## black hole particles ##
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

        file_in = h5py.File(file_name, 'r')  # open hdf5 snapshot file
        part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

        # initialize arrays to store each property for each species
        for spec_name in self.species_names:
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
                    file_in_i = h5py.File(file_name, 'r')
                    part_numbers_in_file_i = file_in_i['Header'].attrs['NumPart_ThisFile']
                    if part_numbers_in_file_i[spec_id] > 0:
                        # found one!
                        part_in = file_in_i['PartType' + str(spec_id)]
                        break
                    file_in_i.close()
                else:
                    # tried all files and still did not find particles of species
                    raise ValueError(
                        '! no {} particles in any snapshot files'.format(spec_name))
            else:
                part_in = file_in['PartType' + str(spec_id)]

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
                else:
                    if prop_name_in in property_dict:
                        prop_name_print = property_dict[prop_name_in]
                    else:
                        prop_name_print = prop_name_in
                    self.say('not reading:  {:6} {}'.format(spec_name, prop_name_print))

            # might have opened extra file if using multi-file snapshot
            try:
                file_in_i.close()
            except:
                pass

            # special case: particle mass is fixed and given in mass array in header
            if 'Masses' in property_names and 'Masses' not in part_in:
                prop_name = property_dict['Masses']
                part[spec_name][prop_name] = np.zeros(part_number_tot, dtype=np.float32)

        file_in.close()

        ## read properties for each species ##
        # initial particle indices to assign to each species from each file
        part_indices_lo = np.zeros(len(self.species_names), dtype=np.int64)

        # loop over all files at given snapshot
        for file_i in range(header['file.number.per.snapshot']):
            # open i'th of multiple files for snapshot
            file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
            file_in = h5py.File(file_name_i, 'r')

            self.say('reading particles from: ' + file_name_i.split('/')[-1])

            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # read particle properties
            for spec_i, spec_name in enumerate(self.species_names):
                spec_id = self.species_dict[spec_name]
                if part_numbers_in_file[spec_id] > 0:
                    part_in = file_in['PartType' + str(spec_id)]

                    part_index_lo = part_indices_lo[spec_i]
                    part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                    # check if mass of species is fixed, according to header mass array
                    if 'Masses' in property_names and header['particle.masses'][spec_id] > 0:
                        prop_name = property_dict['Masses']
                        part[spec_name][prop_name][part_index_lo:part_index_hi] = (
                            header['particle.masses'][spec_id])

                    for prop_name_in in part_in:
                        if prop_name_in in property_names:
                            prop_name = property_dict[prop_name_in]
                            if len(part_in[prop_name_in].shape) == 1:
                                part[spec_name][prop_name][part_index_lo:part_index_hi] = (
                                    part_in[prop_name_in])
                            elif len(part_in[prop_name_in].shape) == 2:
                                if (prop_name_in == 'Metallicity' and
                                        element_indices is not None and
                                        element_indices != 'all'):
                                    prop_in = part_in[prop_name_in][:, element_indices]
                                else:
                                    prop_in = part_in[prop_name_in]
                                part[spec_name][prop_name][part_index_lo:part_index_hi, :] = (
                                    prop_in)

                    part_indices_lo[spec_i] = part_index_hi  # set indices for next file

            file_in.close()

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
                self.say(
                    '! warning: {} different masses of low-resolution dark matter'.format(
                        dark_lowres_masses.size))

            if separate_dark_lowres and dark_lowres_masses.size > 1:
                self.say('separating low-resolution dark-matter by mass into separate dictionaries')
                dark_lowres = {}
                for prop_name in part[spec_name]:
                    dark_lowres[prop_name] = np.array(part[spec_name][prop_name])

                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec_name = 'dark.{}'.format(dark_i + 2)

                    part[spec_name] = ParticleDictionaryClass()

                    for prop_name in dark_lowres:
                        part[spec_name][prop_name] = dark_lowres[prop_name][spec_indices]
                    self.say('  {}: {} particles'.format(spec_name, spec_indices.size))

                    if spec_name not in self.species_names:
                        self.species_names.append(spec_name)

                del(spec_indices)
                print()

        if sort_dark_by_id:
            # order dark-matter particles by id - should be conserved across snapshots
            for spec_name in self.species_names:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    self.say('sorting {:6} particles by id'.format(spec_name))
                    indices_sorted = np.argsort(part[spec_name]['id'])
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = part[spec_name][prop_name][indices_sorted]
                    del(indices_sorted)
            print()

        # apply unit conversions
        for spec_name in self.species_names:
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

        if 'potential' in part[self.species_names[0]]:
            # renormalize so potential max = 0
            potential_max = 0
            for spec_name in self.species_names:
                if part[spec_name]['potential'].max() > potential_max:
                    potential_max = part[spec_name]['potential'].max()
            for spec_name in self.species_names:
                part[spec_name]['potential'] -= potential_max

        if particle_subsample_factor > 1:
            # sub-sample highest-resolution particles, for smaller memory
            spec_names = ['dark', 'gas', 'star']
            self.say('subsampling (periodically) {} particles by factor = {}'.format(
                     spec_names, particle_subsample_factor), end='\n\n')
            for spec_name in part:
                if spec_name in spec_names:
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = (
                            part[spec_name][prop_name][::particle_subsample_factor])

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
            'velocity': [-1e5, 1e5],  # [km/sec]
            'mass': [10, 3e10],  # [M_sun]
            'potential': [-1e9, 0],  # [M_sun]
            'temperature': [3, 1e9],  # [K]
            'density': [0, 1e14],  # [M_sun/kpc^3]
            'smooth.length': [0, 1e9],  # [kpc physical]
            'hydrogen.neutral.fraction': [0, 1],
            'sfr': [0, 1000],  # [M_sun/yr]
            'massfraction': [0, 1],
            'form.scalefactor': [0, 1],
        }

        self.say('checking sanity of particle properties')

        for spec_name in self.species_names:
            for prop_name in part[spec_name]:
                if prop_name in prop_limit_dict:
                    if (part[spec_name][prop_name].min() < prop_limit_dict[prop_name][0] or
                            part[spec_name][prop_name].max() > prop_limit_dict[prop_name][1]):
                        self.say(
                            '! warning: {} {} [min, max] = [{:.3f}, {:.3f}]'.format(
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

        self.say('assigning center of galaxy/halo:')

        if 'position' in part[spec_name]:
            part.center_position = ut.particle.get_center_position(
                part, spec_name, method, compare_centers=compare_centers)
            print('    position = [', end='')
            ut.io.print_array(part.center_position, '{:.3f}', end='')
            print('] kpc comoving')

        if 'velocity' in part[spec_name]:
            part.center_velocity = ut.particle.get_center_velocity(
                part, spec_name, velocity_radius_max, part.center_position)
            print('    velocity = [', end='')
            ut.io.print_array(part.center_velocity, '{:.1f}', end='')
            print('] km / sec')

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

        ## read information about snapshot times ##
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot = self.read_snapshot_times(simulation_directory)
        snapshot_index = Snapshot.parse_snapshot_number(snapshot_number_kind, snapshot_number)

        # get file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)
        self.say('reading header from: ' + file_name, end='\n\n')

        ## read header ##
        # open file and parse header
        file_in = h5py.File(file_name, 'r+')  # open hdf5 snapshot file
        header = file_in['Header'].attrs  # load header dictionary

        ## read and delete input species ##
        for file_i in range(header['NumFilesPerSnapshot']):
            # open i'th of multiple files for snapshot
            file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
            file_in = h5py.File(file_name_i, 'r+')

            self.say('reading properties from: ' + file_name_i.split('/')[-1])

            if 'delete' in action:
                part_number_in_file = header['NumPart_ThisFile']
                part_number = header['NumPart_Total']

            # read and delete particle properties
            for _spec_i, spec_name in enumerate(species_names):
                spec_id = self.species_dict[spec_name]
                spec_name_in = 'PartType' + str(spec_id)
                self.say('  adjusting species = {}'.format(spec_name))

                if 'delete' in action:
                    self.say('  deleting species = {}'.format(spec_name))

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

            file_in.close()


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


def assign_star_form_snapshot_index(part):
    '''
    Assign to each star particle the first snapshot index after it formed,
    to be able to track it back as far as possible.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    '''
    # increase formation time slightly for safety, because output snapshots do not exactly
    # coincide with input snapshot scale-factors
    padding_factor = (1 + 1e-7)

    form_scalefactors = np.array(part['star']['form.scalefactor'])
    form_scalefactors[form_scalefactors < 1] *= padding_factor

    part['star']['form.index'] = part.Snapshot.get_snapshot_indices(
        'scalefactor', form_scalefactors, round_kind='up')


def assign_star_form_distance(
    part, use_child_id=False, part_indices=None, snapshot_index_limits=[]):
    '''
    Assign to each star particle the distance wrt the host galaxy center at the first snapshot
    after it formed.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    use_child_id : boolean : whether to use id.child to match particles with redundant ids
    part_indices : array-like : list of particle indices to assign to
    snapshot_index_limits : list : min and max snapshot indices to impose matching to
    '''
    Say = ut.io.SayClass(assign_star_form_distance)

    spec_name = 'star'

    part[spec_name]['form.distance'] = np.zeros(
        part[spec_name]['position'].shape[0], part[spec_name]['position'].dtype)
    part[spec_name]['form.distance'] -= 1  # initialize to -1

    if 'form.index' not in part['star']:
        assign_star_form_snapshot_index(part)

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.get_arange(part[spec_name]['position'].shape[0])

    if use_child_id and 'id.child' in part[spec_name]:
        pis_unsplit = part_indices[(part[spec_name]['id.child'][part_indices] == 0) *
                                   (part[spec_name]['id.generation'][part_indices] == 0)]

        pis_oversplit = part_indices[part[spec_name]['id.child'][part_indices] >
                                     2 ** part[spec_name]['id.generation'][part_indices]]
    else:
        pis_unique = ut.particle.get_indices_id_kind(part, spec_name, 'unique', part_indices)
        pis_multiple = ut.particle.get_indices_id_kind(part, spec_name, 'multiple', part_indices)
        Say.say('particles with id that is: unique {}, redundant {}'.format(
                pis_unique.size, pis_multiple.size))

    part_indices = pis_unique  # particles to assign to

    # get snapshot indices to sort going back in time
    form_indices = np.unique(part[spec_name]['form.index'][part_indices])[::-1]
    if snapshot_index_limits is not None and len(snapshot_index_limits):
        form_indices = form_indices[form_indices >= min(snapshot_index_limits)]
        form_indices = form_indices[form_indices <= max(snapshot_index_limits)]

    assigned_number_tot = 0
    form_offset_number_tot = 0
    no_id_number_tot = 0

    for snapshot_index in form_indices:
        pis_form_all = part_indices[part[spec_name]['form.index'][part_indices] == snapshot_index]
        pids_form = part[spec_name]['id'][pis_form_all]

        part_snap = Read.read_snapshots(
            spec_name, 'index', snapshot_index,
            property_names=['position', 'mass', 'id', 'form.scalefactor'], element_indices=[0],
            force_float32=True, assign_center=True, check_sanity=True)

        ut.particle.assign_id_to_index(
            part_snap, spec_name, 'id', id_min=0, store_as_dict=True, print_diagnostic=False)

        Say.say('\n# {} particles to assign formation distance'.format(pids_form.size))
        assigned_number_tot += pids_form.size

        pis_snap = []
        pis_form = []
        no_id_number = 0
        for pii, pid in enumerate(pids_form):
            try:
                pi = part_snap[spec_name].id_to_index[pid]
                if np.isscalar(pi):
                    pis_snap.append(pi)
                    pis_form.append(pis_form_all[pii])
                else:
                    no_id_number += 1
            except:
                no_id_number += 1
        pis_snap = np.array(pis_snap, dtype=pis_form_all.dtype)
        pis_form = np.array(pis_form, dtype=pis_form_all.dtype)

        if no_id_number:
            Say.say('! {} particles not have id match'.format(no_id_number))
            no_id_number_tot += no_id_number

        # sanity check
        form_dif_frac_tolerance = 1e-5
        form_scalefactor_difs = np.abs(
            part_snap[spec_name]['form.scalefactor'][pis_snap] -
            part[spec_name]['form.scalefactor'][pis_form]) / part_snap.snapshot['scalefactor']
        form_offset_number = np.sum(form_scalefactor_difs > form_dif_frac_tolerance)
        if form_offset_number:
            Say.say('! {} particles have offset formation time, max = {:.3f} Gyr'.format(
                    form_offset_number, np.max(form_scalefactor_difs)))
            form_offset_number_tot += form_offset_number

        # compute 3-D distance [kpc physical]
        distances = ut.coordinate.get_distances(
            'scalar', part_snap[spec_name]['position'][pis_snap], part_snap.center_position,
            part_snap.info['box.length']) * part_snap.snapshot['scalefactor']  # [kpc physical]

        # assign to catalog
        part[spec_name]['form.distance'][pis_form] = distances

        # continuously write as go, in case happens to crash along the way
        pickle_star_form_distance(part, 'write')

        # print cumulative diagnostics
        Say.say('# totals so far')
        Say.say('{} (of {}) particles assigned'.format(assigned_number_tot, part_indices.size))
        Say.say('{} particles not have id match'.format(no_id_number_tot))
        Say.say('{} particles have offset formation time'.format(form_offset_number_tot))

    Say.say('\n')
    Say.say('# totals across all snapshots:')
    Say.say('{} particles not have id match'.format(no_id_number_tot))
    Say.say('{} particles have offset formation time'.format(form_offset_number_tot))


def pickle_star_form_distance(part, pickle_direction='read'):
    '''
    Read or write, for each star particle, its distance wrt the host galaxy center at the first
    snapshot after it formed.
    If read, assign to particle catalog.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    pickle_direction : string : pickle direction: 'read', 'write'
    '''
    Say = ut.io.SayClass(pickle_star_form_distance)

    spec_name = 'star'

    file_name = 'star_form_distance_{:03d}'.format(part.snapshot['index'])

    if pickle_direction == 'write':
        pickle_object = [part[spec_name]['form.distance'], part[spec_name]['id']]
        ut.io.pickle_object(file_name, pickle_direction, pickle_object)

    elif pickle_direction == 'read':
        part[spec_name]['form.distance'], pids = ut.io.pickle_object(file_name, pickle_direction)

        bad_id_number = np.sum(part[spec_name]['id'] != pids)
        if bad_id_number:
            Say.say('! {} particles with mismatched ids. this is not right.'.format(bad_id_number))

    else:
        raise ValueError('! not recognize pickle_direction = {}'.format(pickle_direction))


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
