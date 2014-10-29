'''
Read gadget/gizmo snapshots.

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
    .
    '''
    def __init__(self):
        self.dimension_num = 3

    def read_snapshot(
        self, directory, snapshot_index, particle_type, file_name_base='snapshot',
        is_cosmological=True, use_four_character_number=False, get_header_only=False):
        '''
        Read simulation snapshot, return as dictionary.

        Parameters
        ----------
        directory: string
        snapshot index: int
        particle kind: string or int
            options:
            0 = gas
            1 = dark matter hat highest resolution,
            4 = stars
            5 = black hole for runs with
            5 = DM at lower resolutions for non-black hole runs (all lower resolution levels)
            2 & 3 = DM at lower resolutions for cosmological
            2 = bulge & 3 = disk stars for non-cosmological
        snapshot file name base: string
        snapshot file extension: string
        '''
        part = {}    # null dictionary if nothing real to return

        file_extension = '.hdf5'

        particle_type_dict = {
            'gas': 0,
            'dark': 1,
            'dark.1': 1,
            'dark.2': 2,
            'dark.3': 3,
            'bulge': 2,
            'disk': 3,
            'star': 4,
            'black.hole': 5
        }

        if isinstance(particle_type, str):
            particle_type = particle_type_dict[particle_type]

        if particle_type < 0 or particle_type > 5:
            print('not recognize particle type = %d' % particle_type)
            return part

        file_name, file_name_base, file_name_extension = self.get_file_name(
            directory, snapshot_index, file_name_base=file_name_base, file_extension=file_extension,
            use_four_character_number=use_four_character_number)

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
        if is_cosmological:
            header['scale.factor'] = header_toparse['Time']
        else:
            header['time'] = header_toparse['Time']    # time {Gyr/h}
        header['redshift'] = header_toparse['Redshift']
        header['box.length'] = header_toparse['BoxSize']
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

        header['box.length'] /= header['hubble']    # {kpc comoving}

        if not is_cosmological:
            header['time'] /= header['hubble']    # {Gyr}

        if get_header_only:
            file_in.close()
            return header

        if header['particle.numbers.total'][particle_type] <= 0:
            file_in.close()
            return part

        if is_cosmological:
            # assume other cosmology parameters from AGORA
            omega_baryon = 0.0455
            sigma_8 = 0.807
            n_s = 0.961
            w = -1.0
            Cosmo = cosmology.CosmologyClass(
                header['hubble'], header['omega_matter'], header['omega_lambda'], omega_baryon,
                sigma_8, n_s, w)

        # initialize variables to read
        poss = np.zeros([header['particle.numbers.total'][particle_type], 3], dtype=float) - 1
        vels = np.copy(poss)
        ids = np.zeros([header['particle.numbers.total'][particle_type]], dtype=long) - 1
        masses = np.zeros([header['particle.numbers.total'][particle_type]], dtype=float) - 1
        if particle_type == 0:
            # gas
            gas_energies_internal = np.copy(masses)
            gas_densities = np.copy(masses)
            gas_smoothing_lengths = np.copy(masses)
            if header['cooling.flag'] > 0:
                gas_electron_fracs = np.copy(masses)
                gas_neutral_hydrogen_fracs = np.copy(masses)
            if header['sfr.flag'] > 0:
                gas_sfrs = np.copy(masses)
        if (particle_type == 0 or particle_type == 4) and header['metal.flag'] > 0:
            # gas or stars
            metals = np.zeros([header['particle.numbers.total'][particle_type],
                               header['metal.flag']], dtype=float)
        if particle_type == 4 and header['sfr.flag'] > 0 and header['star.age.flag'] > 0:
            # stars
            star_form_aexps = np.copy(masses)
        if particle_type == 5:
            # black holes
            bh_masses = np.copy(masses)
            bh_dmdt = np.copy(masses)

        particle_index_lo = 0    # initial particle point to start at

        # loop over snapshot parts to get different data pieces
        for file_i in range(header['file.number.per.snapshot']):
            if header['file.number.per.snapshot'] > 1:
                file_in.close()
                file_name = file_name_base + '.' + str(file_i) + file_name_extension
                file_in = h5py.File(file_name, 'r')    # open hdf5 snapshot file

            input_struct = file_in
            particle_number_in_file = file_in['Header'].attrs['NumPart_ThisFile']
            particle_type_name = 'PartType' + str(particle_type) + '/'

            # do actual reading
            if particle_number_in_file[particle_type] > 0:
                particle_index_hi = particle_index_lo + particle_number_in_file[particle_type]
                poss[particle_index_lo:particle_index_hi, :] = \
                    input_struct[particle_type_name + 'Coordinates']
                vels[particle_index_lo:particle_index_hi, :] = \
                    input_struct[particle_type_name + 'Velocities']
                ids[particle_index_lo:particle_index_hi] = \
                    input_struct[particle_type_name + 'ParticleIDs']
                masses[particle_index_lo:particle_index_hi] = header['mass.array'][particle_type]
                if header['mass.array'][particle_type] <= 0:
                    masses[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'Masses']
                if particle_type == 0:
                    # gas
                    gas_energies_internal[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'InternalEnergy']
                    gas_densities[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'Density']
                    gas_smoothing_lengths[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'SmoothingLength']
                    if header['cooling.flag'] > 0:
                        gas_electron_fracs[particle_index_lo:particle_index_hi] = \
                            input_struct[particle_type_name + 'ElectronAbundance']
                        gas_neutral_hydrogen_fracs[particle_index_lo:particle_index_hi] = \
                            input_struct[particle_type_name + 'NeutralHydrogenAbundance']
                    if header['sfr.flag'] > 0:
                        gas_sfrs[particle_index_lo:particle_index_hi] = \
                            input_struct[particle_type_name + 'StarFormationRate']
                if (particle_type == 0 or particle_type == 4) and header['metal.flag'] > 0:
                    # gas or stars
                    metals_t = input_struct[particle_type_name + 'Metallicity']
                    if header['metal.flag'] > 1:
                        if metals_t.shape[0] != particle_number_in_file[particle_type]:
                            metals_t = np.transpose(metals_t)
                    else:
                        metals_t = np.reshape(np.array(metals_t), (np.array(metals_t).size, 1))
                    metals[particle_index_lo:particle_index_hi, :] = metals_t
                if particle_type == 4 and header['sfr.flag'] > 0 and header['star.age.flag'] > 0:
                    # stars
                    star_form_aexps[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'StellarFormationTime']
                if particle_type == 5:
                    # black holes
                    bh_masses[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'BH_Mass']
                    bh_dmdt[particle_index_lo:particle_index_hi] = \
                        input_struct[particle_type_name + 'BH_Mdot']
                particle_index_lo = particle_index_hi    # set for next iteration

        # correct to same ID as original gas particle for new stars, if bit-flip applied
        if np.min(ids) < 0 or np.max(ids) > 1e9:
            bad = ids < 0 or ids > 1e9
            ids[bad] += 1L << 31

        # do cosmological conversions on final vectors as needed
        poss /= header['hubble']    # {kpc comoving}
        masses *= 1e10 / header['hubble']    # {M_sun}
        if np.min(masses) < 1 or np.max(masses) > 1e10:
            self.say('unsure about particle mass units: read min, max = %.3f, %.3f' %
                     (np.min(masses), np.max(masses)))
        vels *= np.sqrt(header['scale.factor'])    # from gadget's weird units to {km / s physical}
        if particle_type == 0:
            # gas
            gas_densities *= 1 / header['hubble'] / (header['scale.factor'] / header['hubble']) ** 3
            gas_smoothing_lengths *= header['scale.factor'] / header['hubble']    # {kpc physical}
        if particle_type == 4 and header['star.age.flag'] > 0 and not is_cosmological:
            # stars
            star_form_aexps /= header['hubble']
        if particle_type == 5:
            # black holes
            bh_masses /= header['hubble']

        file_in.close()

        if particle_type == 0:
            # gas
            part = {
                'id': ids,    # star particles retain id of their origin gas particle
                'position': poss,
                'velocity': vels,
                'mass': masses,
                'energy.internal': gas_energies_internal,    # {physical}
                'density': gas_densities,
                'smooth.length': gas_smoothing_lengths,
                # average free-electron number per proton (hydrogen nucleon),
                # averaged over mass of gas particle
                'electron.frac': gas_electron_fracs,
                'neutral.hydrogen.frac': gas_neutral_hydrogen_fracs,    # neutral hydrogen fraction
                'sfr': gas_sfrs,    # {M_sun / yr}
                # metallicity {mass fraction} ('solar' would be ~0.02 in total metallicity)
                # element [i,n] gives the metallicity of the n-th species for the i-th particle
                # n=0: "total" metal mass (everything not H, He)
                # n=1: He
                # n=2: C
                # n=3: N
                # n=4: O
                # n=5: Ne
                # n=6: Mg
                # n=7: Si
                # n=8: S
                # n=9: Ca
                # n=10: Fe
                'metal': metals
            }
        elif particle_type == 4:
            # stars
            part = {
                'id': ids,    # star particles retain id of their origin gas particle
                'position': poss,
                'velocity': vels,
                'mass': masses,
                'metal': metals,    # inherited metallicity from gas particle
                # 'time' when the star particle formed
                # for cosmological runs this is the scale factor when star particle formed
                # for non-cosmological runs it is time {Gyr / h} when star particle formed
                'form.time': star_form_aexps
            }
        elif particle_type == 5:
            # black holes
            part = {
                'id': ids,
                'position': poss,
                'velocity': vels,
                'mass': masses,
                'bh.mass': bh_masses,
                'dm/dt': bh_dmdt
            }
        else:
            # dark matter
            part = {
                'id': ids,
                'position': poss,
                'velocity': vels,
                'mass': masses,
            }

        # convert particle structure to generalized dictinary class to increase flexibility
        part_use = ut.array.DictClass()
        for k in part:
            part_use[k] = part[k]
        part_use.Cosmo = Cosmo
        part_use.info = header

        return part_use

    def get_file_name(
        self, directory, snapshot_index, file_name_base='snapshot', file_extension='.hdf5',
        use_four_character_number=False):
        '''
        .
        '''
        for extension_use in [file_extension, '.bin', '']:
            file_name = directory + '/' + file_name_base + '_'
            ext = '00' + str(snapshot_index)
            if snapshot_index >= 10:
                ext = '0' + str(snapshot_index)
            if snapshot_index >= 100:
                ext = str(snapshot_index)
            if use_four_character_number:
                ext = '0' + ext
            if snapshot_index >= 1000:
                ext = str(snapshot_index)
            file_name += ext
            file_name_base = file_name

            s0 = directory.split('/')
            snapshot_directory_specific = s0[len(s0) - 1]
            if len(snapshot_directory_specific) <= 1:
                snapshot_directory_specific = s0[len(s0) - 2]

            # try several common notations for directory/filename structure
            file_name = file_name_base + extension_use
            if not os.path.exists(file_name):
                # is it a multi-part file?
                file_name = file_name_base + '.0' + extension_use
            if not os.path.exists(file_name):
                # is the filename 'snap' instead of 'snapshot'?
                file_name_base = directory + '/snap_' + ext
                file_name = file_name_base + extension_use
            if not os.path.exists(file_name):
                # is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
                file_name = file_name_base + '.0' + extension_use
            if not os.path.exists(file_name):
                # is the filename 'snap(snapdir)' instead of 'snapshot'?
                file_name_base = directory + '/snap_' + snapshot_directory_specific + '_' + ext
                file_name = file_name_base + extension_use
            if not os.path.exists(file_name):
                # is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
                file_name = file_name_base + '.0' + extension_use
            if not os.path.exists(file_name):
                # is it in a snapshot sub-directory? (we assume this means multi-part files)
                file_name_base = directory + '/snapdir_' + ext + '/' + file_name_base + '_' + ext
                file_name = file_name_base + '.0' + extension_use
            if not os.path.exists(file_name):
                # is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
                file_name_base = directory + '/snapdir_' + ext + '/' + 'snap_' + ext
                file_name = file_name_base + '.0' + extension_use
            if not os.path.exists(file_name):
                # give up
                file_name_found = 'NULL'
                file_name_base_found = 'NULL'
                file_name_extension = 'NULL'
                continue
            file_name_found = file_name
            file_name_base_found = file_name_base
            file_name_extension = extension_use
            break    # filename does exist!

        return file_name_found, file_name_base_found, file_name_extension

Gizmo = GizmoClass()
