'''
Read gizmo/gadget snapshots.

Masses in {M_sun}, positions in {kpc comoving}, distances & radii in {kpc physical}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
import h5py as h5py
import os
# local ----
from utilities import utility as ut
from utilities import cosmology


class GizmoClass(ut.io.SayClass):
    '''
    Read in gizmo/gadget snapshots.
    '''
    def __init__(
        self, file_name_base='snapshot', file_extension='.hdf5', use_four_character_index=False):
        '''
        Set properties for snapshot file names.

        file_name_base : string : snapshot file name base
        file_extension : string : snapshot file extension
        use_four_character_index : boolean : whether to use four characters for snapshot index
        '''
        self.file_name_base = file_name_base
        self.file_extension = file_extension
        self.use_four_character_index = use_four_character_index

    def read_snapshot(
        self, species_types='all', snapshot_index=400, directory='.', property_names='all',
        property_names_exclude=[], sort_dark_by_id=True, force_float32=False,
        particle_subsample_factor=1, get_header_only=False):
        '''
        Read simulation snapshot, return as dictionary.

        Parameters
        ----------
        species_types : string or int, or list of these : type[s] of particle species
            options:
            'all' = all species in file
            0 or gas = gas
            1 or dark = dark matter at highest resolution
            2 or dark.2 = dark matter at 2nd highest resolutions for cosmological
            3 or dark.3 = dark matter at 3rd highest resolutions for cosmological
            4 or star = stars
            5 or dark.4 = dark matter at all lower resolutions for cosmological, non black hole runs
            5 or black.hole = black holes, if run contains them
            2 or bulge, 3 or disk = stars for non-cosmological run
        snapshot_index : int : index (number) of snapshot file
        directory: string : directory of snapshot file
        property_names : string or list : name[s] of particle properties to read
            options:
            'all' = all species in file
            otherwise, choose subset from among property_name_dict
        property_names_exclude : string or list : name[s] of particle properties not to read
            note: can use this instead of property_names if just want to exclude a few properties
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        force_float32 : boolean : whether to force floats to 32-bit, to save memory
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        get_header_only : boolean : whether to read only header

        Returns
        -------
        generalized dictionary class, with keys for each species
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

            ## gas particles ##
            'InternalEnergy': 'energy.internal',
            'Density': 'density',
            'SmoothingLength': 'smooth.length',
            #'ArtificialViscosity': 'artificial.viscosity',
            # average free-electron number per proton, averaged over mass of gas particle
            'ElectronAbundance': 'electron.fraction',
            'NeutralHydrogenAbundance': 'neutral.hydrogen.fraction',  # neutral hydrogen fraction
            'StarFormationRate': 'sfr',  # {M_sun / yr}

            ## star/gas metallicity {mass fraction} ('solar' is ~0.02 in total metallicity) ##
            ## stars inherit metallicity from gas particle
            ## 0 = 'total' metal mass (everything not H, He)
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
            ## for cosmological runs, = scale factor; for non-cosmological runs, = time {Gyr / h}
            'StellarFormationTime': 'form.time',

            ## black hole particles ##
            'BH_Mass': 'bh.mass',
            'BH_Mdot': 'dm/dt'
        }

        header = {}  # custom dictionary to store header information
        part = {}  # custom dictionary to store properties for all particle species

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

        # get file name
        file_name, file_name_base = self.get_file_name(directory, snapshot_index)

        self.say('reading header from file: ' + file_name)

        ### start reading/assigning header ###

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
            header['scale.factor'] = float(header['time'])
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
            if header['particle.numbers.total'][spec_id] <= 0:
                self.say('species name = %8s (id = %s): no particles' % (spec_name, spec_id))
            else:
                self.say('species name = %8s (id = %s): %d particles' %
                         (spec_name, spec_id, header['particle.numbers.total'][spec_id]))
                part_number_min = header['particle.numbers.total'][spec_id]
                species_ids_keep.append(spec_id)
                species_names_keep.append(spec_name)
        # keep only names & ids of species that have particles
        species_names = species_names_keep
        species_ids = species_ids_keep

        # check if simulation contains baryons
        if ('gas' not in species_names and 'star' not in species_names and
                'disk' not in species_names and 'bulge' not in species_names):
            header['has.baryons'] = False
        else:
            header['has.baryons'] = True

        # only want to return header?
        if get_header_only:
            file_in.close()
            return header

        ### end reading/assigning header ###

        # sanity check
        if part_number_min == 0:
            self.say('! found no particles in file')
            file_in.close()
            return

        ### initialize arrays to store each property for each species ###

        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]
            part_in = file_in['PartType' + str(spec_id)]
            part[spec_name] = {}  # add species to particle dictionary
            part_num_tot = header['particle.numbers.total'][spec_id]

            for prop_name_in in part_in:
                if prop_name_in in property_names:
                    prop_name = property_name_dict[prop_name_in]

                    # determine shape of property array
                    if len(part_in[prop_name_in].shape) == 1:
                        prop_shape = part_num_tot
                    elif len(part_in[prop_name_in].shape) == 2:
                        prop_shape = [part_num_tot, part_in[prop_name_in].shape[1]]

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
                    self.say('not reading %s for species %s' % (prop_name_in, spec_name))

            # special case: particle mass is fixed and given in mass array in header
            if 'Masses' in property_names and 'Masses' not in part_in:
                prop_name = property_name_dict['Masses']
                part[spec_name][prop_name] = np.zeros(part_num_tot, dtype=np.float32)

        # initial particle indices[s] to assign to each species from each file
        part_indices_lo = np.zeros(len(species_names))

        ### start reading properties for each species ###

        # loop over all files at given snapshot
        for file_i in xrange(header['file.number.per.snapshot']):
            if header['file.number.per.snapshot'] > 1:
                file_in.close()
                file_name = file_name_base + '.' + str(file_i) + self.file_extension
                file_in = h5py.File(file_name, 'r')  # open snapshot file

            self.say('loading file: ' + file_name)

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
                                part[spec_name][prop_name][part_index_lo:part_index_hi, :] = \
                                    part_in[prop_name_in]

                    part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        file_in.close()

        ### end reading properties for each species ###

        ### start adjusting properties for each species ###

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
                    part[spec_name] = {}
                    for prop_name in dark_lowres:
                        part[spec_name][prop_name] = dark_lowres[prop_name][spec_indices]
                    self.say('  %s: %d particles' % (spec_name, spec_indices.size))
                    if spec_name not in species_names:
                        species_names.append(spec_name)
                    if spec_id not in species_ids:
                        species_ids.append(species_name_dict[spec_name])
                del(spec_indices)

        # order dark-matter particles by id - should be conserved across snapshots
        if sort_dark_by_id:
            for spec_name in species_names:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    indices_sorted_by_id = np.argsort(part[spec_name]['id'])
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = \
                            part[spec_name][prop_name][indices_sorted_by_id]

        # apply cosmological conversions
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]
            if 'position' in part[spec_name]:
                part[spec_name]['position'] /= header['hubble']  # convert to {kpc comoving}

            if 'mass' in part[spec_name]:
                part[spec_name]['mass'] *= 1e10 / header['hubble']  # convert to {M_sun}
                if np.min(part[spec_name]['mass']) < 10 or np.max(part[spec_name]['mass']) > 2e10:
                    self.say('! unsure about masses: read [min, max] = [%.3e, %.3e] M_sun' %
                             (np.min(part[spec_name]['mass']), np.max(part[spec_name]['mass'])))

            if 'bh.mass' in part[spec_name]:
                part[spec_name]['bh.mass'] *= 1e10 / header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to {km / s physical}
                part[spec_name]['velocity'] *= np.sqrt(header['scale.factor'])

            if 'density' in part[spec_name]:
                # convert to {M_sun / kpc ** 3 physical}
                part[spec_name]['density'] *= (1e10 / header['hubble'] /
                                               (header['scale.factor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to {kpc physical}
                part[spec_name]['smooth.length'] *= header['scale.factor'] / header['hubble']
                part[spec_name]['smooth.length'] /= 2.8  # Plummer softening, valid for most runs

            if 'form.time' in part[spec_name]:
                # convert to {Gyr}
                part[spec_name]['form.time'] /= header['hubble']

        ### end adjusting properties for each species ###

        # sub-sample highest-resolution particles for smaller memory
        if particle_subsample_factor > 1:
            for spec_name in part:
                if spec_name in ['dark', 'gas', 'star']:
                    for prop_name in part[spec_name]:
                        part[spec_name][prop_name] = \
                            part[spec_name][prop_name][::particle_subsample_factor]

        # convert particle dictionary to generalized dictionary class to increase flexibility
        part_return = ut.array.DictClass()
        for spec_name in part:
            part_return[spec_name] = part[spec_name]

        # assign cosmological parameters
        if header['is.cosmological']:
            # assume that cosmology parameters not in header are from AGORA
            omega_baryon = 0.0455
            sigma_8 = 0.807
            n_s = 0.961
            w = -1.0
            Cosmo = cosmology.CosmologyClass(
                header['hubble'], header['omega_matter'], header['omega_lambda'], omega_baryon,
                sigma_8, n_s, w)
            part_return.Cosmo = Cosmo  # store cosmology information
            for spec_name in part:
                part[spec_name]

        # store header information
        part_return.info = header

        # store snapshot time information
        part_return.snap = {
            'redshift': header['redshift'],
            'scale.factor': header['scale.factor'],
            'time': Cosmo.age(header['redshift'])
        }

        return part_return

    def get_file_name(self, directory, snapshot_index):
        '''
        Get full name (with relative path) of file to read in.

        Parameters
        ----------
        directory: string : directory to check for files
        snapshot_index : int : index of snapshot

        Returns
        -------
        full file name (with relative path): string
        file file name basse (with relative path): string
        '''
        directory = ut.io.get_path(directory)

        s0 = directory.split('/')
        snapshot_directory_specific = s0[len(s0) - 1]
        if len(snapshot_directory_specific) <= 1:
            snapshot_directory_specific = s0[len(s0) - 2]

        snapshot_index_formatted = '00' + str(snapshot_index)
        if snapshot_index >= 10:
            snapshot_index_formatted = '0' + str(snapshot_index)
        if snapshot_index >= 100:
            snapshot_index_formatted = str(snapshot_index)
        if self.use_four_character_index:
            snapshot_index_formatted = '0' + snapshot_index_formatted
        if snapshot_index >= 1000:
            snapshot_index_formatted = str(snapshot_index)

        file_name_base_snapshot = self.file_name_base + '_' + snapshot_index_formatted

        # try several common notations for directory/filename structure
        file_name_base = directory + file_name_base_snapshot
        file_name = file_name_base + self.file_extension
        if not os.path.exists(file_name):
            # is a multi-part file?
            file_name = file_name_base + '.0' + self.file_extension
        if not os.path.exists(file_name):
            # is file name 'snap(snapdir)' instead of 'snapshot'?
            file_name_base = (directory + 'snap_' + snapshot_directory_specific + '_' +
                              snapshot_index_formatted)
            file_name = file_name_base + self.file_extension
        if not os.path.exists(file_name):
            # is file in snapshot sub-directory? assume this means multi-part files
            file_name_base = (directory + 'snapdir_' + snapshot_index_formatted + '/' +
                              file_name_base_snapshot)
            file_name = file_name_base + '.0' + self.file_extension
        if not os.path.exists(file_name):
            # give up
            raise ValueError('! cannot find file to read with name = %s*%s' %
                             (file_name_base, self.file_extension))

        return file_name, file_name_base

Gizmo = GizmoClass()
