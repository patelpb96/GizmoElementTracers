'''
Read Read snapshot files.

Masses in {M_sun}, positions in {kpc comoving}, distances in {kpc physical}.

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
# utility
#===================================================================================================
class ParticleDictionaryClass(dict):
    '''
    Dictionary class to store particle data.
    Allows greater flexibility for producing derived quantities.
    '''
    # use to translate between element name and index in element table
    element_dict = collections.OrderedDict()
    element_dict['metals'] = 0
    element_dict['helium'] = 1
    element_dict['carbon'] = 2
    element_dict['nitrogen'] = 3
    element_dict['oxygen'] = 4
    element_dict['neon'] = 5
    element_dict['magnesium'] = 6
    element_dict['silicon'] = 7
    element_dict['sulphur'] = 8
    element_dict['calcium'] = 9
    element_dict['iron'] = 10
    element_pointer = np.arange(len(element_dict))  # use if read only subset of elements

    def prop(self, property_name='', indices=None):
        '''
        Get property, either from self dictionary or derive.
        If several properties, need to provide mathematical relationship.

        Parameters
        ----------
        property_name : string : name of property
        indices : array : indices to select on
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

            prop_values = self.prop(prop_names[0], indices)
            if not np.isscalar(indices):
                # make copy so not change values in input catalog
                prop_values = np.array(prop_values)

            for prop_name in prop_names[1:]:
                if '/' in property_name:
                    if np.isscalar(prop_values):
                        if self.prop(prop_name, indices) == 0:
                            prop_values = np.nan
                        else:
                            prop_values = prop_values / self.prop(prop_name, indices)
                    else:
                        masks = self.prop(prop_name, indices) != 0
                        prop_values[masks] /= self.prop(prop_name, indices)[masks]
                        masks = self.prop(prop_name, indices) == 0
                        prop_values[masks] = np.nan
                if '*' in property_name:
                    prop_values *= self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values += self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values -= self.prop(prop_name, indices)

            #if prop_values.size == 1:
            #    prop_values = np.float(prop_values, dtype=prop_values.dtype)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        ## parsing specific to this catalog ##
        if 'massfraction.hydrogen' in property_name:
            # mass fraction of hydrogen (excluding helium and metals)
            values = (1 - self.prop('massfraction', indices)[:, 0] -
                      self.prop('massfraction', indices)[:, 1])

            if property_name == 'massfraction.hydrogen.neutral':
                # mass fraction of neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices)

            return values

        if 'mass.hydrogen' in property_name:
            # mass of hydrogen (excluding helium and metals)
            values = self.prop('mass', indices) * self.prop('massfraction.hydrogen', indices)

            if property_name == 'mass.hydrogen.neutral':
                # mass of neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices)

            return values

        if property_name in ['number.density', 'density.number']:
            # number density of hydrogen {cm ^ -3}
            return (self.prop('density', indices) * self.prop('massfraction.hydrogen', indices) *
                    ut.const.proton_per_sun * ut.const.kpc_per_cm ** 3)

        if 'form.time' in property_name and 'lookback' in property_name:
            prop_name = property_name.replace('.lookback', '')
            return self.snapshot['time'] - self.prop(prop_name, indices)

        # element string -> index conversion
        if 'massfraction.' in property_name or 'metallicity.' in property_name:
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
                values = values / ut.const.sun_composition[element_name + '.massfraction']

            return values

        raise ValueError(
            'property = {} is not valid input to {}'.format(property_name, self.__class__))


#===================================================================================================
# read class
#===================================================================================================
class ReadClass(ut.io.SayClass):
    '''
    Read Read snapshot.
    '''
    def __init__(
        self, snapshot_name_base='snap*', file_extension='.hdf5'):
        '''
        Set properties for snapshot file names.

        snapshot_name_base : string : name base of snapshot file/directory
        file_extension : string : snapshot file extension
        '''
        self.snapshot_name_base = snapshot_name_base
        self.file_extension = file_extension

        self.eos = 5 / 3  # gas equation of state

        # snapshot file does not contain these cosmological parameters, so have to set manually
        # these are from AGORA cosmology
        self.omega_baryon = 0.0455
        self.sigma_8 = 0.807
        self.n_s = 0.961
        self.w = -1.0

    def read_snapshot(
        self, species_names='all',
        snapshot_number_kind='index', snapshot_number=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        property_names='all', element_indices=[0, 1], particle_subsample_factor=0,
        assign_center=True, sort_dark_by_id=False, separate_dark_lowres=True, force_float32=False,
        get_header_only=False):
        '''
        Read given properties for given particle species from simulation snapshot file[s].
        Return as dictionary class.

        Parameters
        ----------
        species_names : string or list : name[s] of particle species - options:
            'all' = all species in file
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution for cosmological
            'star' = stars
            'blackhole' = black holes, if run contains them
            'bulge' or 'disk' = stars for non-cosmological run
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_number : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        property_names : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_dict
        element_indices : int or list : indices of elements to keep
            note: 0 = total metals, 1 = helium, 10 = iron, None or 'all' = read all elements
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        assign_center : boolean : whether to assign center position and velocity of galaxy/halo
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        separate_dark_lowres : boolean :
            whether to separate low-resolution dark matter into separate dicts according to mass
        force_float32 : boolean : whether to force all floats to 32-bit, to save memory
        get_header_only : boolean : whether to read only header

        Returns
        -------
        dictionary class, with keys for each particle species
        '''
        # convert particle species name to its id, set all possible species,
        # set the order in which to read species
        species_dict = collections.OrderedDict()
        # dark-matter species in snapshot file
        species_dict['dark'] = 1  # dark matter at highest resolution
        species_dict['dark.2'] = 2  # can include lower-res dark matter at many masses/refinements
        species_dict['dark.3'] = 3
        species_dict['dark.4'] = 5
        # baryon species in snapshot file
        species_dict['gas'] = 0
        species_dict['star'] = 4
        # other - these ids overlap with above, so have to comment in if using them
        #species_dict['blackhole'] = 5
        #species_dict['bulge'] = 2
        #species_dict['disk'] = 3

        # convert name in snapshot's particle dictionary to custon name preference
        # if comment out any property, will not read it
        property_dict = {
            ## all particles ##
            'ParticleIDs': 'id',
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
            'StarFormationRate': 'sfr',  # {M_sun / yr}

            ## star/gas - mass fraction of individual elements ##
            ## 0 = all metals (everything not H, He)
            ## 1 = He
            ## 2 = C
            ## 3 = N
            ## 4 = O
            ## 5 = Ne
            ## 6 = Mg
            ## 7 = Si
            ## 8 = S
            ## 9 = Ca
            ## 10 = Fe
            'Metallicity': 'massfraction',

            ## star particles ##
            ## 'time' when star particle formed
            ## for cosmological runs, = scale-factor; for non-cosmological runs, = time {Gyr / h}
            'StellarFormationTime': 'form.time',

            ## black hole particles ##
            'BH_Mass': 'bh.mass',
            'BH_Mdot': 'form.rate'
        }

        # convert name in snapshot's header dictionary to custom name preference
        header_dict = {
            # 6-element array of number of particles of each type in file
            'NumPart_ThisFile': 'particle.numbers.in.file',
            # 6-element array of total number of particles of each type (across all files)
            'NumPart_Total': 'particle.numbers.total',
            'NumPart_Total_HighWord': 'particle.numbers.total.high.word',
            # mass for each particle type, if all are same (0 if they are different, usually true)
            'MassTable': 'particle.masses',
            'Time': 'time',  # {Gyr / h}
            'BoxSize': 'box.length',  # {kpc / h comoving}
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

        part = ut.array.DictClass()  # dictionary class to store properties for all particle species
        header = {}  # dictionary to store header information

        ## parse input species list ##
        # if input 'all' for species, read all species in snapshot
        if species_names == 'all' or species_names == ['all'] or not species_names:
            species_names = list(species_dict.keys())
        else:
            if np.isscalar(species_names):
                species_names = [species_names]  # ensure is list
            # check if input species names are valid
            for spec_name in list(species_names):
                if spec_name not in species_dict:
                    species_names.remove(spec_name)
                    self.say('! not recognize input species = {}'.format(spec_name))

        ## parse input property list ##
        # if input 'all' for particle properties, read all properties in snapshot
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

        ## read information about snapshot times ##
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot, snapshot_index = self.read_snapshot_times(
            simulation_directory, snapshot_number_kind, snapshot_number)

        ## read header ##
        # get file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)
        self.say('reading header from: ' + file_name, end='\n\n')

        # open file and parse header
        file_in = h5py.File(file_name, 'r')  # open hdf5 snapshot file
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
            header['box.length'] /= header['hubble']  # convert to {kpc comoving}
        else:
            header['time'] /= header['hubble']  # convert to {Gyr}

        # keep only species that have any particles
        particle_number_min = 0
        for spec_name in list(species_names):
            spec_id = species_dict[spec_name]
            self.say('species = {:7s} (id = {}): {} particles'.format(
                     spec_name, spec_id, header['particle.numbers.total'][spec_id]))
            if header['particle.numbers.total'][spec_id] > 0:
                particle_number_min = header['particle.numbers.total'][spec_id]
            else:
                species_names.remove(spec_name)
        print()

        # check if simulation contains baryons
        if ('gas' not in species_names and 'star' not in species_names and
                'disk' not in species_names and 'bulge' not in species_names):
            header['has.baryons'] = False
        else:
            header['has.baryons'] = True

        header['catalog.kind'] = 'particle'
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.strip('/')
        header['simulation.name'] = simulation_name

        if get_header_only or particle_number_min == 0:
            # only return header
            if particle_number_min == 0:
                self.say('! found no particles in file', end='\n\n')
            file_in.close()
            return header

        ## finish reading header ##

        # assign cosmological parameters
        if header['is.cosmological']:
            # for cosmological parameters not in header, use values set above
            Cosmology = ut.cosmology.CosmologyClass(
                header['hubble'], header['omega_matter'], header['omega_lambda'],
                self.omega_baryon, self.sigma_8, self.n_s, self.w)

        ## initialize arrays to store each property for each species ##
        for spec_name in species_names:
            spec_id = species_dict[spec_name]
            part_in = file_in['PartType' + str(spec_id)]
            part_number_tot = header['particle.numbers.total'][spec_id]

            # add species to particle dictionary
            part[spec_name] = ParticleDictionaryClass()

            # set element pointers if reading only subset of elements
            if element_indices is not None and element_indices != 'all':
                if np.isscalar(element_indices):
                    element_indices = [element_indices]
                for element_i, element_index in enumerate(element_indices):
                    part[spec_name].element_pointer[element_index] = element_i

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
                    self.say('not reading {:6} {}'.format(spec_name, prop_name_in))

            # special case: particle mass is fixed and given in mass array in header
            if 'Masses' in property_names and 'Masses' not in part_in:
                prop_name = property_dict['Masses']
                part[spec_name][prop_name] = np.zeros(part_number_tot, dtype=np.float32)

        # initial particle indices[s] to assign to each species from each file
        part_indices_lo = np.zeros(len(species_names))

        ## start reading properties for each species ##
        # loop over all files at given snapshot
        for file_i in range(header['file.number.per.snapshot']):
            if file_i == 0:
                file_name_i = file_name
            else:
                # open i'th of multiple files for snapshot
                file_in.close()
                file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
                file_in = h5py.File(file_name_i, 'r')

            self.say('reading properties from: ' + file_name_i.split('/')[-1])

            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # read particle properties
            for spec_i, spec_name in enumerate(species_names):
                if part_numbers_in_file[spec_id]:
                    spec_id = species_dict[spec_name]
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
                                        element_indices is not None and element_indices != 'all'):
                                    prop_in = part_in[prop_name_in][:, element_indices]
                                else:
                                    prop_in = part_in[prop_name_in]
                                part[spec_name][prop_name][part_index_lo:part_index_hi, :] = prop_in

                    part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        file_in.close()
        print()
        ## end reading properties for each species ##

        ## start adjusting properties for each species ##
        # if dark.2 contains different masses (refinements), split into separate dicts
        spec_name = 'dark.2'
        if spec_name in part and 'mass' in part[spec_name]:
            dark_lowres_masses = np.unique(part[spec_name]['mass'])
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

                    if spec_name not in species_names:
                        species_names.append(spec_name)

                del(spec_indices)
                print()

        # order dark-matter particles by id - should be conserved across snapshots
        if sort_dark_by_id:
            for spec_name in species_names:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    self.say('sorting {:6} particles by id'.format(spec_name))
                    indices_sorted = np.argsort(part[spec_name]['id'])
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = part[spec_name][prop_name][indices_sorted]
            print()

        # apply unit conversions
        for spec_name in species_names:

            if 'position' in part[spec_name]:
                # convert to {kpc comoving}
                part[spec_name]['position'] /= header['hubble']

            if 'mass' in part[spec_name]:
                # convert to {M_sun}
                part[spec_name]['mass'] *= 1e10 / header['hubble']
                if np.min(part[spec_name]['mass']) < 1 or np.max(part[spec_name]['mass']) > 2e10:
                    self.say(
                        '! unsure about masses: read [min, max] = [{:.3e}, {:.3e}] M_sun'.format(
                            np.min(part[spec_name]['mass']), np.max(part[spec_name]['mass'])))

            if 'bh.mass' in part[spec_name]:
                # convert to {M_sun}
                part[spec_name]['bh.mass'] *= 1e10 / header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to {km / sec physical}
                part[spec_name]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'density' in part[spec_name]:
                # convert to {M_sun / kpc ^ 3 physical}
                part[spec_name]['density'] *= (1e10 / header['hubble'] /
                                               (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to {pc physical}
                part[spec_name]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                part[spec_name]['smooth.length'] /= 2.8  # Plummer softening, valid for most runs

            if 'form.time' in part[spec_name]:
                if header['is.cosmological']:
                    # convert from units of scale-factor to {Gyr}
                    part[spec_name]['form.time'] = Cosmology.get_time_from_redshift(
                        1 / part[spec_name]['form.time'] - 1).astype(
                            part[spec_name]['form.time'].dtype)
                else:
                    # convert to {Gyr}
                    part[spec_name]['form.time'] /= header['hubble']

            if 'temperature' in part[spec_name]:
                # convert from {(km / s) ^ 2} to {Kelvin}
                # ignore small corrections from elements beyond He
                helium_mass_fracs = part[spec_name]['massfraction'][:, 1]
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + part[spec_name]['electron.fraction'])
                molecular_weights = mus * ut.const.proton_mass
                part[spec_name]['temperature'] *= (
                    ut.const.centi_per_kilo ** 2 * (self.eos - 1) * molecular_weights /
                    ut.const.boltzmann)

        if 'potential' in part[species_names[0]]:
            # renormalize so potential max = 0
            potential_max = 0
            for spec_name in species_names:
                if part[spec_name]['potential'].max() > potential_max:
                    potential_max = part[spec_name]['potential'].max()
            for spec_name in species_names:
                part[spec_name]['potential'] -= potential_max

        # sub-sample highest-resolution particles for smaller memory
        if particle_subsample_factor > 1:
            spec_names = ['dark', 'gas', 'star']
            self.say('subsampling (periodically) {} particles by factor = {}'.format(
                     spec_names, particle_subsample_factor), end='\n\n')
            for spec_name in part:
                if spec_name in spec_names:
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = \
                            part[spec_name][prop_name][::particle_subsample_factor]

        ## end adjusting properties for each species ##

        ## assign auxilliary information ##

        # store cosmology class
        part.Cosmology = Cosmology
        for spec_name in species_names:
            part[spec_name].Cosmology = part.Cosmology

        # store header dictionary
        part.info = header
        for spec_name in species_names:
            part[spec_name].info = part.info

        # store information about snapshot time
        time = Cosmology.get_time_from_redshift(header['redshift'])
        part.snapshot = {
            'index': snapshot_index,
            'redshift': header['redshift'],
            'scalefactor': header['scalefactor'],
            'time': time,
            'time.lookback': Cosmology.get_time_from_redshift(0) - time,
            'time.hubble': ut.const.Gyr_per_sec / Cosmology.get_hubble_parameter(0),
        }
        for spec_name in species_names:
            part[spec_name].snapshot = part.snapshot

        # store information on all snapshot times - may or may not be initialized
        part.Snapshot = Snapshot

        # arrays to store center position and velocity
        part.center_position = []
        part.center_velocity = []
        if assign_center:
            self.assign_center(part)

        return part

    def read_snapshot_times(self, directory, snapshot_number_kind, snapshot_number):
        '''
        Read snapshot file that contains scale-factors[, redshifts, times, time spacings].
        Return as dictionary class.

        Parameters
        ----------
        directory : string : directory of snapshot file
        snapshot_number_kind : string : kind of number that am supplying: 'redshift', 'index'
        snapshot_number : int or float : corresponding number

        Returns
        -------
        dictionary class of snapshot information
        '''
        directory = ut.io.get_path(directory)

        Snapshot = ut.simulation.SnapshotClass()

        try:
            try:
                Snapshot.read_snapshots('snapshot_times.txt', directory)
            except:
                Snapshot.read_snapshots('snapshot_scale-factors.txt', directory)
        except:
            if snapshot_number_kind in ['redshift', 'scalefactor', 'time']:
                raise ValueError(
                    'input {} for snapshot, but cannot find snapshot time file in ' +
                    snapshot_number_kind, directory)

        self.is_first_print = True

        if snapshot_number_kind in ['redshift', 'scalefactor', 'time']:
            snapshot_index = Snapshot.get_index(snapshot_number, time_kind=snapshot_number_kind)
            snapshot_time_kind = snapshot_number_kind
            snapshot_time_value = Snapshot[snapshot_number_kind][snapshot_index]
            self.say('input {} = {:.3f}'.format(snapshot_number_kind, snapshot_number))
        else:
            snapshot_index = snapshot_number
            snapshot_time_kind = 'redshift'
            snapshot_time_value = Snapshot['redshift'][snapshot_index]
        self.say(
            'reading snapshot index = {}, {} = {:.3f}\n'.format(
                snapshot_index, snapshot_time_kind, snapshot_time_value))

        return Snapshot, snapshot_index

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
            path_names, file_indices = ut.io.get_file_names(directory + self.snapshot_name_base,
                                                            int)
        except:
            path_names, file_indices = ut.io.get_file_names(directory + self.snapshot_name_base,
                                                            float)

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

    def assign_center(self, part, method='center-of-mass', compare_centers=True):
        '''
        Assign center position {kpc comoving} and velocity {km / sec physical} to galaxy/halo,
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

        self.say('assigning galaxy/halo center:')

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

Read = ReadClass()


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
