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
    def __init__(self, int_dtype=np.int64, float_dtype=np.float64):
        '''
        Parameters
        ----------
        data type for particle int properties: dtype
        data type for particle float properties: dtype
        '''
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.dimension_num = 3    # number of spatial dimensions

    def read_snapshot(
        self, species_types, snapshot_index=440, directory='.', file_name_base='snapshot',
        use_four_character_index=False, get_header_only=False):
        '''
        Read simulation snapshot, return as dictionary.

        Parameters
        ----------
        type[s] of particle species: string or int, or list thereof
            options:
            0 = gas
            1 = dark matter at highest resolution
            2 & 3 = dark matter at lower resolutions for cosmological
            4 = stars
            5 = dark matter at lower res for non-black hole runs (all lower resolution levels)
            5 = black holes, if runs contain thems
            2 = bulge & 3 = disk stars for non-cosmological
        snapshot index: int
        directory: string
        snapshot file name base: string
        whether to use four characters for snapshot index: boolean
        whether to read only header: boolean
        '''
        part = {}    # null dictionary if nothing real to return

        file_extension = '.hdf5'

        species_dict = {
            'gas': 0,
            'dark': 1,
            'dark.2': 2,
            'dark.3': 3,
            #'bulge': 2,
            #'disk': 3,
            'star': 4,
            #'black.hole': 5
            'dark.4': 5
        }

        # ensure is list even if just one species
        if species_types == 'all' or species_types == ['all']:
            species_types = species_dict.keys()
        if np.isscalar(species_types):
            species_types = [species_types]

        species_ids = []
        for spec_type in species_types:
            if isinstance(spec_type, str):
                species_ids.append(species_dict[spec_type])
            elif isinstance(spec_type, int):
                species_ids.append(spec_type)
            else:
                raise ValueError('not recognize particle type = %s' % spec_type)

        species_names = []
        for spec_id in species_ids:
            for spec_name in species_dict:
                if species_dict[spec_name] == spec_id:
                    species_names.append(spec_name)

        if np.min(species_ids) < 0 or np.max(species_ids) > 5:
            print('not recognize particle types = %s' % species_types)
            return part

        file_name, file_name_base, file_name_extension = self.get_file_name(
            directory, snapshot_index, file_name_base=file_name_base, file_extension=file_extension,
            use_four_character_index=use_four_character_index)

        if file_name == 'NULL':
            print('! cannot find file')
            return part

        print('loading file: ' + file_name)

        # open file and parse its header information
        file_in = h5py.File(file_name, 'r')    # open hdf5 snapshot file
        header_toparse = file_in['Header'].attrs    # load header dictionary

        header = {}
        # 6-element array of number of particles of each type in file
        header['particle.numbers.in.file'] = header_toparse['NumPart_ThisFile']
        # 6-element array of total number of particles of each type (across all files at snapshot)
        header['particle.numbers.total'] = header_toparse['NumPart_Total']
        # mass for each particle type, if all are same (0 if they are different, usually true)
        header['mass.array'] = header_toparse['MassTable']
        time = header_toparse['Time']
        header['redshift'] = header_toparse['Redshift']
        box_length = header_toparse['BoxSize']
        # number of output files per snapshot
        header['file.number.per.snapshot'] = header_toparse['NumFilesPerSnapshot']
        header['omega_matter'] = header_toparse['Omega0']
        header['omega_lambda'] = header_toparse['OmegaLambda']
        header['hubble'] = header_toparse['HubbleParam']
        header['sfr.flag'] = header_toparse['Flag_Sfr']
        header['cooling.flag'] = header_toparse['Flag_Cooling']
        header['star.age.flag'] = header_toparse['Flag_StellarAge']
        header['metal.flag'] = header_toparse['Flag_Metals']
        header['feedback.flag'] = header_toparse['Flag_Feedback']
        print('particle number in file: ', header['particle.numbers.in.file'])
        print('particle number total: ', header['particle.numbers.total'])

        # infer whether simulation is cosmological
        if (0 < header_toparse['HubbleParam'] < 1 and 0 < header_toparse['Omega0'] < 1 and
                0 < header_toparse['OmegaLambda'] < 1):
            is_cosmological = True
        else:
            is_cosmological = False
            self.say('assuming that simulation is not cosmological')
            self.say('read h = %.3f, omega_matter_0 = %.3f, omega_lambda_0 = %>3f' %
                     (header_toparse['HubbleParam'], header_toparse['Omega0'],
                      header_toparse['OmegaLambda']))
        header['is.cosmological'] = is_cosmological

        if is_cosmological:
            header['scale.factor'] = time
            header['box.length/h'] = box_length
            header['box.length'] = box_length / header['hubble']    # {kpc comoving}
        else:
            header['time'] = time / header['hubble']    # {Gyr}
            header['box.length'] = box_length    # {kpc / h}

        # check if simulation contains baryons
        if ('gas' not in species_names and 'star' not in species_names and
                'disk' not in species_names and 'bulge' not in species_names):
            header['is.baryonic'] = False
        else:
            header['is.baryonic'] = True

        if get_header_only:
            file_in.close()
            return header

        # check that have some particles
        part_number_min = 0
        for spec_name in species_dict.keys():
            spec_id = species_dict[spec_name]
            if header['particle.numbers.total'][spec_id] <= 0:
                self.say('no particles for species id = %s' % spec_id)
                species_ids.remove(spec_id)
                species_names.remove(spec_name)
            else:
                part_number_min = header['particle.numbers.total'][spec_id]
        if part_number_min == 0:
            file_in.close()
            return part

        # assign cosmological parameters
        if is_cosmological:
            # assume that cosmology parameters not in header are from AGORA - kluge!
            omega_baryon = 0.0455
            sigma_8 = 0.807
            n_s = 0.961
            w = -1.0
            Cosmo = cosmology.CosmologyClass(
                header['hubble'], header['omega_matter'], header['omega_lambda'], omega_baryon,
                sigma_8, n_s, w)

        # initialize variables to read
        for spec_name in species_names:
            spec_id = species_dict[spec_name]

            if spec_id == 0:
                gas = {}
                self.initialize_common_properties(gas, spec_id, header)
                gas['energy.internal'] = np.copy(gas['mass'])    # {physical}
                gas['density'] = np.copy(gas['mass'])    # {physical}
                gas['smooth.length'] = np.copy(gas['mass'])
                if header['cooling.flag']:
                    # average free-electron number per proton (hydrogen nucleon),
                    # averaged over mass of gas particle
                    gas['electron.frac'] = np.copy(gas['mass'])
                    # neutral hydrogen fraction
                    gas['neutral.hydrogen.frac'] = np.copy(gas['mass'])
                if header['sfr.flag']:
                    gas['sfr'] = np.copy(gas['mass'])    # {M_sun / yr}
                if header['metal.flag']:
                    # metallicity {mass fraction} ('solar' would be ~0.02 in total metallicity)
                    # element [i,n] gives the metallicity of the n-th species for the i-th particle
                    # n = 0: "total" metal mass (everything not H, He)
                    # n = 1: He
                    # n = 2: C
                    # n = 3: N
                    # n = 4: O
                    # n = 5: Ne
                    # n = 6: Mg
                    # n = 7: Si
                    # n = 8: S
                    # n = 9: Ca
                    # n = 10: Fe
                    gas['metal'] = np.zeros([header['particle.numbers.total'][spec_id],
                                             header['metal.flag']], dtype=self.float_dtype)
                part[spec_name] = gas

            elif spec_id == 1:
                dark = {}
                self.initialize_common_properties(dark, spec_id, header)
                part[spec_name] = dark

            elif spec_id == 2:
                dark2 = {}
                self.initialize_common_properties(dark2, spec_id, header)
                part[spec_name] = dark2

            elif spec_id == 3:
                dark3 = {}
                self.initialize_common_properties(dark3, spec_id, header)
                part[spec_name] = dark3

            elif spec_id == 4:
                star = {}
                self.initialize_common_properties(star, spec_id, header)
                # star particles retain id of their origin gas particle
                if header['sfr.flag'] and header['star.age.flag']:
                    # 'time' when the star particle formed
                    # for cosmological runs this is the scale factor when star particle formed
                    # for non-cosmological runs this is the time {Gyr / h} when star particle formed
                    star['form.time'] = np.copy(star['mass'])
                if header['metal.flag']:
                    # inherited metallicity from gas particle
                    star['metal'] = np.zeros([header['particle.numbers.total'][spec_id],
                                              header['metal.flag']], dtype=self.float_dtype)
                part[spec_name] = star

            elif spec_id == 5:
                # can be used for black holes or dark matter at lowest resolution
                blackhole = {}
                self.initialize_common_properties(blackhole, spec_id, header)
                if spec_name == 'black.hole':
                    blackhole['bh.mass'] = np.copy(blackhole['mass'])
                    blackhole['dm/dt'] = np.copy(blackhole['mass'])
                part[spec_name] = blackhole

            else:
                raise ValueError('not recognize species id = %d' % spec_id)

        # initial particle indices[s] to start
        part_indices_lo = np.zeros(len(species_ids), self.int_dtype)

        # loop over snapshot parts to get different data pieces
        for file_i in range(header['file.number.per.snapshot']):
            if header['file.number.per.snapshot'] > 1:
                file_in.close()
                file_name = file_name_base + '.' + str(file_i) + file_name_extension
                file_in = h5py.File(file_name, 'r')    # open hdf5 snapshot file

            input_struct = file_in
            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # read actual properties
            for spec_i, spec_name in enumerate(species_names):
                spec_id = species_dict[spec_name]
                spec_input_name = 'PartType' + str(spec_id) + '/'
                part_index_lo = part_indices_lo[spec_i]

                if part_numbers_in_file[spec_id]:
                    part_index_hi = part_index_lo + part_numbers_in_file[spec_id]
                    part[spec_name]['position'][part_index_lo:part_index_hi, :] = \
                        input_struct[spec_input_name + 'Coordinates']
                    part[spec_name]['velocity'][part_index_lo:part_index_hi, :] = \
                        input_struct[spec_input_name + 'Velocities']
                    part[spec_name]['id'][part_index_lo:part_index_hi] = \
                        input_struct[spec_input_name + 'ParticleIDs']
                    part[spec_name]['mass'][part_index_lo:part_index_hi] = \
                        header['mass.array'][spec_id]
                    if header['mass.array'][spec_id] <= 0:
                        part[spec_name]['mass'][part_index_lo:part_index_hi] = \
                            input_struct[spec_input_name + 'Masses']

                    if spec_id == 0:
                        # gas
                        part[spec_name]['energy.internal'][part_index_lo:part_index_hi] = \
                            input_struct[spec_input_name + 'InternalEnergy']
                        part[spec_name]['density'][part_index_lo:part_index_hi] = \
                            input_struct[spec_input_name + 'Density']
                        part[spec_name]['smooth.length'][part_index_lo:part_index_hi] = \
                            input_struct[spec_input_name + 'SmoothingLength']
                        if header['cooling.flag']:
                            part[spec_name]['electron.frac'][part_index_lo:part_index_hi] = \
                                input_struct[spec_input_name + 'ElectronAbundance']
                            part[spec_name]['neutral.hydrogen.frac'][part_index_lo:
                                                                     part_index_hi] = \
                                input_struct[spec_input_name + 'NeutralHydrogenAbundance']
                        if header['sfr.flag']:
                            part[spec_name]['sfr'][part_index_lo:part_index_hi] = \
                                input_struct[spec_input_name + 'StarFormationRate']
                        metals_t = input_struct[spec_input_name + 'Metallicity']
                        if header['metal.flag']:
                            if metals_t.shape[0] != part_numbers_in_file[spec_id]:
                                metals_t = np.transpose(metals_t)
                        else:
                            metals_t = np.reshape(np.array(metals_t), (np.array(metals_t).size, 1))
                        part[spec_name]['metal'][part_index_lo:part_index_hi, :] = metals_t

                    elif spec_id == 4:
                        # stars
                        if header['metal.flag']:
                            metals_t = input_struct[spec_input_name + 'Metallicity']
                            if header['metal.flag'] > 1:
                                if metals_t.shape[0] != part_numbers_in_file[spec_id]:
                                    metals_t = np.transpose(metals_t)
                            else:
                                metals_t = np.reshape(np.array(metals_t),
                                                      (np.array(metals_t).size, 1))
                            part[spec_name]['metal'][part_index_lo:part_index_hi, :] = metals_t

                        if header['sfr.flag'] > 0 and header['star.age.flag'] > 0:
                            part[spec_name]['form.time'][part_index_lo:part_index_hi] = \
                                input_struct[spec_input_name + 'StellarFormationTime']

                    elif spec_id == 5:
                        # black holes or dark matter at lowest-resolution
                        if spec_name == 'black.hole':
                            part[spec_name]['bh.mass'][part_index_lo:part_index_hi] = \
                                input_struct[spec_input_name + 'BH_Mass']
                            part[spec_name]['dm/dt'][part_index_lo:part_index_hi] = \
                                input_struct[spec_input_name + 'BH_Mdot']

                    part_indices_lo[spec_i] = part_index_hi    # set for next iteration

        file_in.close()

        # correct to same ID as original gas particle for new stars, if bit-flip applied
        for spec_name in species_names:
            if len(part[spec_name]['id']):
                if np.min(part[spec_name]['id']) < 0 or np.max(part[spec_name]['id']) > 1e10:
                    masks = part[spec_name]['id'] < 0 or part[spec_name]['id'] > 2e9
                    part[spec_name]['id'][masks] += 1L << 31

        # cosmological conversions on final vectors as needed
        for spec_name in species_names:
            spec_id = species_dict[spec_name]

            part[spec_name]['position'] /= header['hubble']    # {kpc comoving}
            part[spec_name]['mass'] *= 1e10 / header['hubble']    # {M_sun}
            if (np.min(part[spec_name]['mass']) < 10 or np.max(part[spec_name]['mass']) > 2e10):
                self.say('unsure about particle mass units: read min, max = %.3e, %.3e' %
                         (np.min(part[spec_name]['mass']),
                          np.max(part[spec_name]['mass'])))

            # gadget's weird units to {km / s physical}
            part[spec_name]['velocity'] *= np.sqrt(header['scale.factor'])

            if spec_id == 0:
                # gas
                part[spec_name]['density'] *= (1 / header['hubble'] /
                                               (header['scale.factor'] / header['hubble']) ** 3)
                # {kpc physical}
                part[spec_name]['smooth.length'] *= header['scale.factor'] / header['hubble']

            elif spec_id == 4 and header['star.age.flag'] and not is_cosmological:
                # stars
                part[spec_name]['star.form.time'] /= header['hubble']

            elif spec_id == 5:
                # black holes or dark matter at lowest resolution
                if spec_name == 'black.hole':
                    part[spec_name]['bh.mass'] /= header['hubble']

        # convert particle structure to generalied dictinary class to increase flexibility
        part_use = ut.array.DictClass()
        if len(species_names) == 1:
            for k in part[species_names[0]]:
                part_use[k] = part[species_names[0]][k]
        else:
            for k in part:
                part_use[k] = part[k]
        part_use.Cosmo = Cosmo
        part_use.info = header
        part_use.snap = {'redshift': header['redshift'], 'scale.factor': header['scale.factor'],
                         'time': Cosmo.age(header['redshift'])}

        return part_use

    def initialize_common_properties(self, part, species_id, header):
        '''
        Assign common properties to particle type dictionary.

        Parameters
        ----------
        catalog of particles: dictionary
        particle type id: int
        header information: dictionary
        '''
        part['position'] = np.zeros([header['particle.numbers.total'][species_id],
                                     self.dimension_num], self.float_dtype) - 1
        part['velocity'] = np.zeros([header['particle.numbers.total'][species_id],
                                     self.dimension_num], self.float_dtype)
        # initialize so calling an un-itialized value leads to error
        part['id'] = (np.zeros(header['particle.numbers.total'][species_id], self.int_dtype) -
                      header['particle.numbers.total'][species_id] + 1)
        part['mass'] = np.zeros(header['particle.numbers.total'][species_id], self.float_dtype) - 1

    def get_file_name(
        self, directory, snapshot_index, file_name_base='snapshot', file_extension='.hdf5',
        use_four_character_index=False):
        '''
        Get name of file to read in.

        Parameters
        ----------
        directory: string
        index of snapshot: int
        name base of file: string
        extention of file: string
        whether to use four characters in snapshot index: boolean
        '''
        directory = ut.io.get_safe_path(directory)

        s0 = directory.split('/')
        snapshot_directory_specific = s0[len(s0) - 1]
        if len(snapshot_directory_specific) <= 1:
            snapshot_directory_specific = s0[len(s0) - 2]

        for file_extension_found in [file_extension, '.bin', '']:
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
            file_name = file_name_base + file_extension_found
            if not os.path.exists(file_name):
                # is a multi-part file?
                file_name = file_name_base + '.0' + file_extension_found
            if not os.path.exists(file_name):
                # is file name 'snap(snapdir)' instead of 'snapshot'?
                file_name_base = (directory + 'snap_' + snapshot_directory_specific + '_' +
                                  snapshot_index_formatted)
                file_name = file_name_base + file_extension_found
            if not os.path.exists(file_name):
                # is file in snapshot sub-directory? assume this means multi-part files
                file_name_base = (directory + 'snapdir_' + snapshot_index_formatted + '/' +
                                  file_name_base_snapshot)
                file_name = file_name_base + '.0' + file_extension_found
            if not os.path.exists(file_name):
                # give up
                file_name = 'NULL'
                file_name_base = 'NULL'
                file_extension_found = 'NULL'
                continue

            file_name_found = file_name
            file_name_base_found = file_name_base
            break    # filename does exist!

        return file_name_found, file_name_base_found, file_extension_found

Gizmo = GizmoClass()
