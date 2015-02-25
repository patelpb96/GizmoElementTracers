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
    def read_snapshot(
        self, species_types, snapshot_index=440, directory='.', file_name_base='snapshot',
        file_extension='.hdf5', use_four_character_index=False, get_header_only=False,
        subsample_factor=1):
        '''
        Read simulation snapshot, return as dictionary.

        Parameters
        ----------
        type[s] of particle species: string or int, or list thereof
            options:
            0 or gas = gas
            1 or dark = dark matter at highest resolution
            2 or dark.2 = dark matter at 2nd highest resolutions for cosmological
            3 or dark.3 = dark matter at 3rd highest resolutions for cosmological
            4 or star = stars
            5 or dark.4 = dark matter at all lower resolutions for cosmological, non black hole runs
            5 or black.hole = black holes, if run contains them
            2 or bulge, 3 or disk = stars for non-cosmological run
        snapshot index: int
        directory: string
        snapshot file name base: string
        snapshot file extension: string
        whether to use four characters for snapshot index: boolean
        whether to read only header: boolean

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
            #'bulge': 2,
            #'disk': 3,
            'star': 4,
            #'black.hole': 5,
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
            'Time': 'time',    # {Gyr / h}
            'BoxSize': 'box.length',    # {kpc / h comoving}
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
        # if comment out given property, will not read that one
        particle_name_dict = {
            # all particles
            'ParticleIDs': 'id',
            'Coordinates': 'position',
            #'Velocities': 'velocity',
            'Masses': 'mass',
            #'Potential': 'potential',

            'InternalEnergy': 'energy.internal',
            'Density': 'density',
            'SmoothingLength': 'smooth.length',
            #'ArtificialViscosity': 'artificial.viscosity',

            # average free-electron number per proton, averaged over mass of gas particle
            'ElectronAbundance': 'electron.fraction',

            # neutral hydrogen fraction
            'NeutralHydrogenAbundance': 'neutral.hydrogen.fraction',

            'StarFormationRate': 'sfr',    # {M_sun / yr}

            # metallicity {mass fraction} ('solar' would be ~0.02 in total metallicity)
            # for stars, this is inherited metallicity from gas particle
            # 0 = 'total' metal mass (everything not H, He)
            # 1 = He
            # 2 = C
            # 3 = N
            # 4 = O
            # 5 = Ne
            # 6 = Mg
            # 7 = Si
            # 8 = S
            # 9 = Ca
            # 10 = Fe
            #'Metallicity': 'metallicity',

            # 'time' when the star particle formed
            # for cosmological runs, this is the scale factor
            # for non-cosmological runs, this is the time {Gyr / h}
            'StellarFormationTime': 'form.time',

            'BH_Mass': 'bh.mass',
            'BH_Mdot': 'dm/dt'
        }

        header = {}    # custom dictionary to store header information
        part = {}    # custom dictionary to store properties for all particle species

        # if input 'all', read everything possible
        if species_types == 'all' or species_types == ['all']:
            species_types = [spec for spec in species_name_list if spec in species_name_dict.keys()]

        # ensure species is list even if just one
        if np.isscalar(species_types):
            species_types = [species_types]

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

        # get file name
        file_name, file_name_base = self.get_file_name(
            directory, snapshot_index, file_name_base=file_name_base, file_extension=file_extension,
            use_four_character_index=use_four_character_index)

        self.say('loading file: ' + file_name)

        ### start reading/assigning header ###

        # open file and parse header
        file_in = h5py.File(file_name, 'r')    # open hdf5 snapshot file
        header_in = file_in['Header'].attrs    # load header dictionary

        for prop in header_in:
            header[header_name_dict[prop]] = header_in[prop]    # transfer to custom header dict

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
            header['box.length'] /= header['hubble']    # convert to {kpc comoving}
        else:
            header['time'] /= header['hubble']    # convert to {Gyr}

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
            part[spec_name] = {}    # add species to particle dictionary
            part_num_tot = header['particle.numbers.total'][spec_id]

            for prop_in in part_in:
                if prop_in in particle_name_dict:
                    prop = particle_name_dict[prop_in]
                    # determine shape of property array
                    if len(part_in[prop_in].shape) == 1:
                        prop_shape = part_num_tot
                    elif len(part_in[prop_in].shape) == 2:
                        prop_shape = [part_num_tot, part_in[prop_in].shape[1]]
                    # initialize to -1's
                    part[spec_name][prop] = np.zeros(prop_shape, part_in[prop_in].dtype) - 1
                    if prop == 'id':
                        # initialize so calling an un-itialized value leads to error
                        part[spec_name][prop] -= part_num_tot
                else:
                    self.say('not reading %s for species %s' % (prop_in, spec_name))

            # check for special case: particle mass is fixed and is given in mass array in header
            if 'Masses' in particle_name_dict and 'Masses' not in part_in:
                prop = particle_name_dict['Masses']
                part[spec_name][prop] = np.zeros(part_num_tot, dtype=np.float32)

        # initial particle indices[s] to assign to each species from each file
        part_indices_lo = np.zeros(len(species_names))

        ### start reading properties for each species ###

        # loop over all files at given snapshot
        for file_i in xrange(header['file.number.per.snapshot']):
            if header['file.number.per.snapshot'] > 1:
                file_in.close()
                file_name = file_name_base + '.' + str(file_i) + file_extension
                file_in = h5py.File(file_name, 'r')    # open snapshot file

            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # read particle properties
            for spec_i, spec_name in enumerate(species_names):
                if part_numbers_in_file[spec_id]:
                    spec_id = species_name_dict[spec_name]
                    part_in = file_in['PartType' + str(spec_id)]

                    part_index_lo = part_indices_lo[spec_i]
                    part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                    # check if mass of species is fixed, according to header mass array
                    if 'Masses' in particle_name_dict and header['mass.array'][spec_id] > 0:
                        prop = particle_name_dict['Masses']
                        part[spec_name][prop][part_index_lo:part_index_hi] = \
                            header['mass.array'][spec_id]

                    for prop_in in part_in:
                        if prop_in in particle_name_dict:
                            prop = particle_name_dict[prop_in]
                            if len(part_in[prop_in].shape) == 1:
                                part[spec_name][prop][part_index_lo:part_index_hi] = \
                                    part_in[prop_in]
                            elif len(part_in[prop_in].shape) == 2:
                                part[spec_name][prop][part_index_lo:part_index_hi, :] = \
                                    part_in[prop_in]

                    part_indices_lo[spec_i] = part_index_hi    # set indices for next file

        file_in.close()

        ### end reading properties for each species ###

        ### start adjusting properties for each species ###

        # if dark.2 contains several resolution levels of dark matter, separate them by mass
        if 'dark.2' in part:
            dark_lowres_masses = np.unique(part['dark.2']['mass'])
            if dark_lowres_masses.size > 1:
                self.say('separating lower-resolution dark-matter species:')
                dark_lowres = {}
                for prop in part['dark.2']:
                    dark_lowres[prop] = np.array(part['dark.2'][prop])
                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec_name = 'dark.%d' % (dark_i + 2)
                    part[spec_name] = {}
                    for prop in dark_lowres:
                        part[spec_name][prop] = dark_lowres[prop][spec_indices]
                    self.say('  %s: %d particles' % (spec_name, spec_indices.size))
                    if spec_name not in species_names:
                        species_names.append(spec_name)
                    if spec_id not in species_ids:
                        species_ids.append(species_name_dict[spec_name])
                del(spec_indices)

        # order dark-matter particles by id - should be conserved across snapshots
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]
            if 'dark' in spec_name:
                indices_sorted_by_id = np.argsort(part[spec_name]['id'])
                for prop in part[spec_name]:
                    part[spec_name][prop] = part[spec_name][prop][indices_sorted_by_id]

        # check if need bit-flip
        for spec_name in species_names:
            if np.min(part[spec_name]['id']) < 0 or np.max(part[spec_name]['id']) > 5e9:
                masks = part[spec_name]['id'] < 0 or part[spec_name]['id'] > 5e9
                self.say('detected possible bit-flip for %d particles of species: %s' %
                         (np.sum(masks), spec_name))
                part[spec_name]['id'][masks] += 1L << 31

        # apply cosmological conversions
        for spec_name in species_names:
            spec_id = species_name_dict[spec_name]
            if 'position' in part[spec_name]:
                part[spec_name]['position'] /= header['hubble']    # convert to {kpc comoving}

            if 'mass' in part[spec_name]:
                part[spec_name]['mass'] *= 1e10 / header['hubble']    # convert to {M_sun}
                if np.min(part[spec_name]['mass']) < 10 or np.max(part[spec_name]['mass']) > 2e10:
                    self.say('! unsure about masses: read [min, max] = [%.3e, %.3e] M_sun' %
                             (np.min(part[spec_name]['mass']), np.max(part[spec_name]['mass'])))

            if 'bh.mass' in part[spec_name]:
                part[spec_name]['bh.mass'] /= header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to {km / s physical}
                part[spec_name]['velocity'] *= np.sqrt(header['scale.factor'])

            if 'density' in part[spec_name]:
                # convert to {M_sun / kpc ** 3 physical}
                part[spec_name]['density'] *= (1 / header['hubble'] /
                                               (header['scale.factor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to {kpc physical}
                part[spec_name]['smooth.length'] *= header['scale.factor'] / header['hubble']

            if 'form.time' in part[spec_name]:
                # convert to {Gyr}
                part[spec_name]['form.time'] /= header['hubble']

        ### end adjusting properties for each species ###

        # convert particle dictionary to generalized dictionary class to increase flexibility
        part_return = ut.array.DictClass()
        for spec_name in part:
            part_return[spec_name] = part[spec_name]

        # sub-sample highest-resolution particles for smaller memory
        if subsample_factor > 1:
            for spec_name in part:
                if spec_name in ['dark', 'gas', 'star']:
                    for prop in part[spec_name]:
                        part[spec_name][prop] = part[spec_name][prop][::subsample_factor]

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
            part_return.Cosmo = Cosmo    # store cosmology information

        part_return.info = header    # store header information
        # store snapshot time information
        part_return.snap = {'redshift': header['redshift'], 'scale.factor': header['scale.factor'],
                            'time': Cosmo.age(header['redshift'])}

        return part_return

    def get_file_name(
        self, directory, snapshot_index, file_name_base='snapshot', file_extension='.hdf5',
        use_four_character_index=False):
        '''
        Get full name (with relative path) of file to read in.

        Parameters
        ----------
        directory: string
        index of snapshot: int
        name base of file: string
        extention of file: string
        whether to use four characters in snapshot index: boolean

        Returns
        -------
        full file name (with relative path): string
        file file name basse (with relative path): string
        '''
        directory = ut.io.get_safe_path(directory)

        s0 = directory.split('/')
        snapshot_directory_specific = s0[len(s0) - 1]
        if len(snapshot_directory_specific) <= 1:
            snapshot_directory_specific = s0[len(s0) - 2]

        snapshot_index_formatted = '00' + str(snapshot_index)
        if snapshot_index >= 10:
            snapshot_index_formatted = '0' + str(snapshot_index)
        if snapshot_index >= 100:
            snapshot_index_formatted = str(snapshot_index)
        if use_four_character_index:
            snapshot_index_formatted = '0' + snapshot_index_formatted
        if snapshot_index >= 1000:
            snapshot_index_formatted = str(snapshot_index)

        file_name_base_snapshot = file_name_base + '_' + snapshot_index_formatted

        # try several common notations for directory/filename structure
        file_name_base = directory + file_name_base_snapshot
        file_name = file_name_base + file_extension
        if not os.path.exists(file_name):
            # is a multi-part file?
            file_name = file_name_base + '.0' + file_extension
        if not os.path.exists(file_name):
            # is file name 'snap(snapdir)' instead of 'snapshot'?
            file_name_base = (directory + 'snap_' + snapshot_directory_specific + '_' +
                              snapshot_index_formatted)
            file_name = file_name_base + file_extension
        if not os.path.exists(file_name):
            # is file in snapshot sub-directory? assume this means multi-part files
            file_name_base = (directory + 'snapdir_' + snapshot_index_formatted + '/' +
                              file_name_base_snapshot)
            file_name = file_name_base + '.0' + file_extension
        if not os.path.exists(file_name):
            # give up
            raise ValueError('! cannot find file to read with name = %s*%s' %
                             (file_name_base, file_extension))

        return file_name, file_name_base

Gizmo = GizmoClass()
