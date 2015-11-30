'''
Read Gizmo snapshot files.

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
# use to translate between element name and index in element table
element_dict = collections.OrderedDict()
element_dict['metal'] = 0
element_dict['helium'] = 1
element_dict['carbon'] = 2
element_dict['nitrogen'] = 3
element_dict['oxygen'] = 4
element_dict['neon'] = 5
element_dict['magnesium'] = 6
element_dict['silicon'] = 7,
element_dict['sulphur'] = 8,
element_dict['calcium'] = 9
element_dict['iron'] = 10
element_dict['ironderived'] = 0


class ParticleDictionaryClass(dict):
    '''
    Dictionary class to store particle data.
    Allows greater flexibility for producing derived quantities.
    '''
    def prop(self, property_name='', indices=None):
        '''
        Get property, either from self dictionary or derive.
        If several properties, need to provide mathematical relationship.

        Parameters
        ----------
        property_name : string : name of property
        indices : array : indices to select on
        '''
        property_name = property_name.strip()  # strip white space

        # check if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]

        # check for relational property (involves combining more than one property)
        if ('/' in property_name or '*' in property_name or '+' in property_name or
                '-' in property_name):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise ValueError('property = %s is not a valid input to halo catalog' %
                                 property_name)

            prop_values = np.array(self.prop(prop_names[0], indices))
            for prop_name in prop_names[1:]:
                if '/' in property_name:
                    masks = self.prop(prop_name, indices) != 0
                    prop_values[masks] /= self.prop(prop_name, indices)[masks]
                    masks = self.prop(prop_name, indices) == 0
                    prop_values[masks] = 0
                if '*' in property_name:
                    prop_values *= self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values += self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values -= self.prop(prop_name, indices)

            return prop_values

        # check for math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(
                self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        if 'hydrogen.fraction' in property_name:
            # mass fraction of hydrogen (excluding helium and metals)
            return(1 - self.prop('metallicity', indices)[:, 0] -
                   self.prop('metallicity', indices)[:, 1])

        if 'mass.hydrogen' in property_name:
            # mass of hydrogen (excluding helium and metals)
            values = self.prop('mass', indices) * self.prop('hydrogen.fraction', indices)

            if property_name == 'mass.hydrogen.neutral':
                # mass of neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = np.array(values) * self.prop('hydrogen.neutral.fraction', indices)

            return values

        if property_name in ['number.density', 'density.number']:
            # number density of hydrogen {cm ^ -3}
            return (self.prop('density', indices) * self.prop('hydrogen.fraction', indices) *
                    ut.const.proton_per_sun * ut.const.kpc_per_cm ** 3)

        if 'form.time' in property_name and 'lookback' in property_name:
            prop_name = property_name.replace('.lookback', '')
            return self.snapshot['time'] - self.prop(prop_name, indices)

        # check for metallicity string -> index conversion
        if 'metallicity.' in property_name:
            metal_index = None
            for prop_name in property_name.split('.'):
                if prop_name in element_dict:
                    metal_index = element_dict[prop_name]
                    metal_name = prop_name
                    break

            if metal_index is None:
                ValueError('property = %s is not a valid input to halo catalog' % property_name)

            if indices is None:
                values = self['metallicity'][:, metal_index]
            else:
                values = self['metallicity'][indices, metal_index]

            if '.ironderived' in property_name:
                # conversion to [Fe/H] solar from Ma et al 2015
                values = np.array(values) / 10 ** 0.2 * 0.02

            if '.solar' in property_name:
                values = np.array(values) / ut.const.sun_composition[metal_name + '.mass.fraction']

            return values

        raise ValueError('property = %s is not a valid input to halo catalog' % property_name)


#===================================================================================================
# read class
#===================================================================================================
class GizmoClass(ut.io.SayClass):
    '''
    Read Gizmo snapshot.
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

    def read_snapshot(
        self, species_types='all', snapshot_number_kind='index', snapshot_number=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        property_names='all', property_names_exclude=[],
        metal_index_max=1, particle_subsample_factor=0,
        assign_center=True, sort_dark_by_id=False, force_float32=False, get_header_only=False):
        '''
        Read given properties for given particle species from simulation snapshot file[s].
        Return as dictionary class.

        Parameters
        ----------
        species_types : string or int, or list of these : type[s] of particle species - options:
            'all' = all species in file
            0 or gas = gas
            1 or dark = dark matter at highest resolution
            2 or dark.2 = dark matter at 2nd highest resolutions for cosmological
            3 or dark.3 = dark matter at 3rd highest resolutions for cosmological
            4 or star = stars
            5 or dark.4 = dark matter at all lower resolutions for cosmological, non black hole runs
            5 or black.hole = black holes, if run contains them
            2 or bulge, 3 or disk = stars for non-cosmological run
        snapshot_number_kind : string : input snapshot number kind: index, redshift
        snapshot_number : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        property_names : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_name_dict
        property_names_exclude : string or list : name[s] of particle properties not to read
            note: can use this instead of property_names if just want to exclude a few properties
        metal_index_max : int : maximum metal index to keep
            options: None = keep all, 0 = total, 1 = total + helium, 10 = iron (no r-process)
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        assign_center : boolean : whether to assign center position and velocity of galaxy/halo
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        force_float32 : boolean : whether to force all floats to 32-bit, to save memory
        get_header_only : boolean : whether to read only header

        Returns
        -------
        dictionary class, with keys for each particle species
        '''
        # connects particle species name to id, and determines all possible species types
        species_name_dict = {
            'gas': 0,
            'dark': 1,
            'dark.2': 2,
            'dark.3': 3,
            # 'bulge': 2,
            # 'disk': 3,
            'star': 4,
            # 'black.hole': 5,
            'dark.4': 5,

            # use below types to divvy out coarser dark matter
            'dark.5': 6,
            'dark.6': 7,
            'dark.7': 8,
            'dark.8': 9,
        }

        # sets the order in which to read species
        species_name_list = [
            'dark',
            'dark.1',
            'dark.2',
            'dark.3',
            'dark.4',
            'gas',
            'star',
            'black.hole',
            'bulge',
            'disk',
        ]

        # converts each key in input header dictionary to another naming preference
        header_name_dict = {
            # 6-element array of number of particles of each type in file
            'NumPart_ThisFile': 'particle.numbers.in.file',
            # 6-element array of total number of particles of each type (across all files)
            'NumPart_Total': 'particle.numbers.total',
            'NumPart_Total_HighWord': 'particle.numbers.total.high.word',
            # mass for each particle type, if all are same (0 if they are different, usually true)
            'MassTable': 'mass.array',
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

        # converts each key in input particle dictionary to another naming preference
        # if comment out any property, will not read it
        property_name_dict = {
            ## all particles ##
            'ParticleIDs': 'id',
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',

            ## particles with adaptive smoothing ##
            'AGS-Softening': 'soften.length',

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

            ## star/gas metallicity {mass fraction} ##
            ## 0 = total metal mass fraction (everything not H, He)
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
            'Metallicity': 'metallicity',

            ## star particles ##
            ## 'time' when star particle formed
            ## for cosmological runs, = scale-factor; for non-cosmological runs, = time {Gyr / h}
            'StellarFormationTime': 'form.time',

            ## black hole particles ##
            'BH_Mass': 'bh.mass',
            'BH_Mdot': 'dm/dt'
        }

        header = {}  # dictionary to store header information

        # dictionary class to store properties for all particle species
        part = ut.array.DictClass()

        ## parse input species list ##
        # if input 'all' for species, read all species in snapshot
        if species_types == 'all' or species_types == ['all'] or not species_types:
            species_types = [spec for spec in species_name_list if spec in species_name_dict.keys()]
        else:
            if np.isscalar(species_types):
                species_types = [species_types]  # ensure is list

        # check if input species types are string or int, and assign species id list
        species_ids = []
        for spec_type in species_types:
            if isinstance(spec_type, str):
                species_ids.append(species_name_dict[spec_type])
            elif isinstance(spec_type, int):
                if spec_type < 0 or spec_type > 9:
                    raise ValueError('! not recognize species type = %d' % spec_type)
                species_ids.append(spec_type)
            else:
                raise ValueError('not recognize species type = %s' % spec_type)

        # assign species name list
        species_names = []
        for spec_id in species_ids:
            for spec_name in species_name_dict:
                if species_name_dict[spec_name] == spec_id:
                    species_names.append(spec_name)

        ## parse input property list ##
        # if input 'all' for particle properties, read all properties in snapshot
        if property_names == 'all' or property_names == ['all'] or not property_names:
            property_names = [prop_name for prop_name in property_name_dict]
        else:
            if np.isscalar(property_names):
                property_names = [property_names]  # ensure is list
            # make safe list of property names to read in
            property_names_temp = []
            for prop_name in property_names:
                prop_name = str.lower(prop_name)
                for prop_name_in in property_name_dict:
                    if (prop_name == str.lower(prop_name_in) or
                            prop_name == str.lower(property_name_dict[prop_name_in])):
                        property_names_temp.append(prop_name_in)
            property_names = property_names_temp
            del(property_names_temp)

        if property_names_exclude:
            if np.isscalar(property_names_exclude):
                property_names_exclude = [property_names_exclude]  # ensure is list
            for prop_name in property_names_exclude:
                prop_name = str.lower(prop_name)
                for prop_name_in in property_names:
                    if (prop_name == str.lower(prop_name_in) or
                            prop_name == str.lower(property_name_dict[prop_name_in])):
                        property_names.remove(prop_name_in)

        if 'InternalEnergy' in property_names:
            # need helium and electron fraction to compute temperature
            for prop_name in ['ElectronAbundance', 'Metallicity']:
                if prop_name not in property_names:
                    property_names.append(prop_name)

        ## read information on snapshot times ##
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot, snapshot_index = self.read_snapshot_times(
            simulation_directory, snapshot_number_kind, snapshot_number)

        ## read and assign header ##
        # get file name
        file_name = self.get_file_name(snapshot_directory, snapshot_index)
        self.say('reading header from: ' + file_name, end='\n\n')

        # open file and parse header
        file_in = h5py.File(file_name, 'r')  # open hdf5 snapshot file
        header_in = file_in['Header'].attrs  # load header dictionary

        for prop_name_in in header_in:
            prop_name = header_name_dict[prop_name_in]
            header[prop_name] = header_in[prop_name_in]  # transfer to custom header dict

        # infer whether simulation is cosmological
        if (0 < header['hubble'] < 1 and 0 < header['omega_matter'] < 1 and
                0 < header['omega_lambda'] < 1):
            header['is.cosmological'] = True
        else:
            header['is.cosmological'] = False
            self.say('assuming that simulation is not cosmological')
            self.say('read h = %.3f, omega_matter_0 = %.3f, omega_lambda_0 = %>3f' %
                     (header['hubble'], header['omega_matter'], header['omega_lambda']))

        # convert some header quantities
        if header['is.cosmological']:
            header['scalefactor'] = float(header['time'])
            del(header['time'])
            header['box.length/h'] = float(header['box.length'])
            header['box.length'] /= header['hubble']  # convert to {kpc comoving}
        else:
            header['time'] /= header['hubble']  # convert to {Gyr}

        # check which species have any particles
        part_number_min = 0
        species_names_keep = []
        species_ids_keep = []
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]
            self.say('species name = %7s (id = %s): %d particles' %
                     (spec_name, spec_id, header['particle.numbers.total'][spec_id]))
            if header['particle.numbers.total'][spec_id] > 0:
                part_number_min = header['particle.numbers.total'][spec_id]
                species_ids_keep.append(spec_id)
                species_names_keep.append(spec_name)
        print()

        # keep only names and ids of species that have particles
        species_names = species_names_keep
        species_ids = species_ids_keep

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

        if get_header_only or part_number_min == 0:
            # only return header
            if part_number_min == 0:
                self.say('! found no particles in file', end='\n\n')
            file_in.close()
            return header

        ## end reading header ##

        # assign cosmological parameters
        if header['is.cosmological']:
            # assume that cosmology parameters not in header are from AGORA
            omega_baryon = 0.0455
            sigma_8 = 0.807
            n_s = 0.961
            w = -1.0
            Cosmology = ut.cosmology.CosmologyClass(
                header['hubble'], header['omega_matter'], header['omega_lambda'], omega_baryon,
                sigma_8, n_s, w)

        ## initialize arrays to store each property for each species ##
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]

            part_in = file_in['PartType' + str(spec_id)]

            part_num_tot = header['particle.numbers.total'][spec_id]

            # add species to particle dictionary
            #part[spec_name] = {}
            part[spec_name] = ParticleDictionaryClass()

            for prop_name_in in part_in:
                if prop_name_in in property_names:
                    prop_name = property_name_dict[prop_name_in]

                    # determine shape of property array
                    if len(part_in[prop_name_in].shape) == 1:
                        prop_shape = part_num_tot
                    elif len(part_in[prop_name_in].shape) == 2:
                        prop_shape = [part_num_tot, part_in[prop_name_in].shape[1]]
                        if prop_name_in == 'Metallicity' and metal_index_max:
                            prop_shape = [part_num_tot, metal_index_max + 1]

                    # determine data type to store
                    prop_in_dtype = part_in[prop_name_in].dtype
                    if force_float32 and prop_in_dtype == 'float64':
                        prop_in_dtype = np.float32

                    # initialize to -1's
                    part[spec_name][prop_name] = np.zeros(prop_shape, prop_in_dtype) - 1

                    if prop_name == 'id':
                        # initialize so calling an un-itialized value leads to error
                        part[spec_name][prop_name] -= part_num_tot
                else:
                    self.say('not reading %s %s' % (spec_name, prop_name_in))

            # special case: particle mass is fixed and given in mass array in header
            if 'Masses' in property_names and 'Masses' not in part_in:
                prop_name = property_name_dict['Masses']
                part[spec_name][prop_name] = np.zeros(part_num_tot, dtype=np.float32)

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
                file_name_i = file_name.replace('.0.', '.%d.' % file_i)
                file_in = h5py.File(file_name_i, 'r')

            self.say('reading properties from: ' + file_name_i.split('/')[-1])

            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # read particle properties
            for spec_i, spec_name in enumerate(species_names):
                if part_numbers_in_file[spec_id]:
                    spec_id = species_name_dict[spec_name]
                    part_in = file_in['PartType' + str(spec_id)]

                    part_index_lo = part_indices_lo[spec_i]
                    part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                    # check if mass of species is fixed, according to header mass array
                    if 'Masses' in property_names and header['mass.array'][spec_id] > 0:
                        prop_name = property_name_dict['Masses']
                        part[spec_name][prop_name][part_index_lo:part_index_hi] = \
                            header['mass.array'][spec_id]

                    for prop_name_in in part_in:
                        if prop_name_in in property_names:
                            prop_name = property_name_dict[prop_name_in]
                            if len(part_in[prop_name_in].shape) == 1:
                                part[spec_name][prop_name][part_index_lo:part_index_hi] = \
                                    part_in[prop_name_in]
                            elif len(part_in[prop_name_in].shape) == 2:
                                if prop_name_in == 'Metallicity' and metal_index_max:
                                    prop_in = part_in[prop_name_in][:, :metal_index_max + 1]
                                else:
                                    prop_in = part_in[prop_name_in]
                                part[spec_name][prop_name][part_index_lo:part_index_hi, :] = prop_in

                    part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        file_in.close()
        print()
        ## end reading properties for each species ##

        ## start adjusting properties for each species ##
        # if species dark.2 contains several mass levels of dark matter, split into separate dicts
        spec_name = 'dark.2'
        if spec_name in part and 'mass' in part[spec_name]:
            dark_lowres_masses = np.unique(part[spec_name]['mass'])
            if dark_lowres_masses.size > 1:
                self.say('separating lower-resolution dark-matter species:')
                dark_lowres = {}
                for prop_name in part[spec_name]:
                    dark_lowres[prop_name] = np.array(part[spec_name][prop_name])

                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec_name = 'dark.%d' % (dark_i + 2)

                    #part[spec_name] = {}
                    part[spec_name] = ParticleDictionaryClass()

                    for prop_name in dark_lowres:
                        part[spec_name][prop_name] = dark_lowres[prop_name][spec_indices]
                    self.say('  %s: %d particles' % (spec_name, spec_indices.size))
                    if spec_name not in species_names:
                        species_names.append(spec_name)
                    if spec_id not in species_ids:
                        species_ids.append(species_name_dict[spec_name])

                del(spec_indices)
                print()

        # order dark-matter particles by id - should be conserved across snapshots
        if sort_dark_by_id:
            for spec_name in species_names:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    self.say('sorting %s particles by id' % spec_name)
                    indices_sorted_by_id = np.argsort(part[spec_name]['id'])
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = \
                            part[spec_name][prop_name][indices_sorted_by_id]
            print()

        # apply unit conversions
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]

            if 'position' in part[spec_name]:
                # convert to {kpc comoving}
                part[spec_name]['position'] /= header['hubble']

            if 'mass' in part[spec_name]:
                # convert to {M_sun}
                part[spec_name]['mass'] *= 1e10 / header['hubble']
                if np.min(part[spec_name]['mass']) < 1 or np.max(part[spec_name]['mass']) > 2e10:
                    self.say('! unsure about masses: read [min, max] = [%.3e, %.3e] M_sun' %
                             (np.min(part[spec_name]['mass']), np.max(part[spec_name]['mass'])))

            if 'bh.mass' in part[spec_name]:
                # convert to {M_sun}
                part[spec_name]['bh.mass'] *= 1e10 / header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to {km / sec physical}
                part[spec_name]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'density' in part[spec_name]:
                # convert to {M_sun / kpc ** 3 physical}
                part[spec_name]['density'] *= (1e10 / header['hubble'] /
                                               (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to {pc physical}
                part[spec_name]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                part[spec_name]['smooth.length'] /= 2.8  # Plummer softening, valid for most runs

            if 'soften.length' in part[spec_name]:
                # convert to {pc physical}
                part[spec_name]['soften.length'] *= 1000 * header['scalefactor'] / header['hubble']
                part[spec_name]['soften.length'] /= 2.8  # Plummer softening, valid for most runs

            if 'form.time' in part[spec_name]:
                if header['is.cosmological']:
                    # convert from units of scale-factor to {Gyr}
                    part[spec_name]['form.time'] = Cosmology.time_from_redshift(
                        1 / part[spec_name]['form.time'] - 1).astype(
                            part[spec_name]['form.time'].dtype)
                else:
                    # convert to {Gyr}
                    part[spec_name]['form.time'] /= header['hubble']

            if 'temperature' in part[spec_name]:
                # convert from {(km / s) ^ 2} to {Kelvin}
                # ignore small corrections from metals (beyond He)
                helium_mass_fracs = part[spec_name]['metallicity'][:, 1]
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
            self.say('subsampling (periodically) %s particles by factor = %d' %
                     (spec_names, particle_subsample_factor), end='\n\n')
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

        # store information on snapshot time
        time = Cosmology.time_from_redshift(header['redshift'])
        part.snapshot = {
            'redshift': header['redshift'],
            'scalefactor': header['scalefactor'],
            'time': time,
            'time.lookback': Cosmology.time_from_redshift(0) - time,
            'time.hubble': ut.const.Gyr_per_sec / Cosmology.hubble_parameter(0),
        }
        for spec_name in species_names:
            part[spec_name].snapshot = part.snapshot

        # store information on all snapshot times - may or may not be initialized
        part.Snapshot = Snapshot

        # arrays to store center position and velocity
        part.center_position = []
        part.center_velocity = []
        if assign_center:
            # assign center now
            self.assign_center(part)

        return part

    def read_snapshot_times(self, directory, snapshot_number_kind, snapshot_number):
        '''
        Read snapshot file that contains scale-factors[, redshifts, times, time spacings].
        Return as dictionary class.

        Parameters
        ----------
        directory : string : directory of snaphot file
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
            if snapshot_number_kind == 'redshift':
                raise ValueError(
                    'input redshift for snapshot, but cannot find file of snapshot times in %s' %
                    directory)

        if snapshot_number_kind == 'redshift':
            snapshot_redshift = snapshot_number
            snapshot_index = Snapshot.get_index(snapshot_number)
            self.say('input redshift = %.3f -> reading snapshot index = %d, redshift = %.3f\n' %
                     (snapshot_redshift, snapshot_index, Snapshot['redshift'][snapshot_index]))

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
            raise ValueError('cannot find snapshot index = %d in: %s' %
                             (snapshot_index, path_names))

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
                raise ValueError('cannot find 0th snapshot file in: %s' % path_file_names)

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

        self.say('assigning center:')

        if 'position' in part[spec_name]:
            part.center_position = ut.particle.get_center_position(
                part, spec_name, method, compare_centers=compare_centers)
            print('    position: ', end='')
            ut.io.print_array(part.center_position, '%.3f', end='')
            print(' kpc comoving')

        if 'velocity' in part[spec_name]:
            part.center_velocity = ut.particle.get_center_velocity(
                part, spec_name, velocity_radius_max, part.center_position)
            print('    velocity: ', end='')
            ut.io.print_array(part.center_velocity, '%.1f', end='')
            print(' km / sec')

        print()

    def assign_orbit(
        self, part, species=['star'], center_position=None, center_velocity=None,
        include_hubble_flow=True):
        '''
        Assign derived orbital properties to species.

        Parameters
        ----------
        part : dict : catalog of particles at snapshot
        species : string or list : particle species to compute
        center_position : array : center position to use
        center_velocity : array : center velocity to use
        include_hubble_flow : boolean : whether to include hubble flow
        '''
        from . import gizmo_analysis

        if np.isscalar(species):
            species = [species]  # ensure is list

        orb = gizmo_analysis.get_orbit_dictionary(
            part, species, center_position, center_velocity, include_hubble_flow, scalarize=False)

        for spec_name in species:
            for prop in orb[spec_name]:
                part[spec_name]['host.' + prop] = orb[spec_name][prop]


Gizmo = GizmoClass()
