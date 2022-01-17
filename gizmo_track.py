#!/usr/bin/env python3

'''
Track particles across snapshots in Gizmo simulations.
Using particles tracked back in time, assign the coordinates and properties of the main
progenitor of each host galaxy.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

import os
import sys
import collections
import numpy as np

import utilities as ut
from . import gizmo_default
from . import gizmo_io


# dictionary key of particle id in catalog
ID_NAME = 'id'
ID_CHILD_NAME = 'id.child'


# --------------------------------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------------------------------
class ParticlePointerDictionaryClass(dict, ut.io.SayClass):
    '''
    Dictionary class to store and compute particle pointer indices (and species names),
    for tracking star and gas particles across snapshots.
    '''

    def __init__(self, part_z0=None, part_z=None, species_names=['star', 'gas']):
        '''
        Given input particle catalogs, store summary info about snapshots and particle counts.

        Parameters
        ----------
        part_z0 : dict
            catalog of particles at the reference (later) snapshot
        part_z : dict
            catalog of particles at an earlier snapshot
        species_names : str or list
            name[s] of particle species to track
        id_name : str
            dictionary key of particle id
        '''
        self.id_name = ID_NAME
        self.z0_name = 'z0.'  # prefactor name for reference (latest) snapshot
        self.z_name = 'z.'  # prefactor name for the earlier snapshot
        self.zi_name = 'zi.'  # prefactor name for an intermediate snapshot
        self.pointer_index_name = self.z0_name + 'to.' + self.z_name + 'index'

        # if no input particle catalogs, leave uninitialized
        if part_z0 is not None and part_z is not None:
            self['species'] = species_names

            z0 = self.z0_name
            z = self.z_name

            # initialize particle counters
            self[z0 + 'particle.number'] = 0
            self[z + 'particle.number'] = 0
            for spec_name in species_names:
                self[z0 + spec_name + '.number'] = part_z0[spec_name][self.id_name].size
                self[z0 + spec_name + '.index.limits'] = [
                    self[z0 + 'particle.number'],
                    self[z0 + 'particle.number'] + part_z0[spec_name][self.id_name].size,
                ]
                self[z0 + 'particle.number'] += part_z0[spec_name][self.id_name].size

                # check that species is in particle catalog
                # early snapshots may not have star particles
                if spec_name in part_z and len(part_z[spec_name][self.id_name]) > 0:
                    self[z + spec_name + '.number'] = part_z[spec_name][self.id_name].size
                    self[z + spec_name + '.index.limits'] = [
                        self[z + 'particle.number'],
                        self[z + 'particle.number'] + part_z[spec_name][self.id_name].size,
                    ]
                    self[z + 'particle.number'] += part_z[spec_name][self.id_name].size
                else:
                    self[z + spec_name + '.number'] = 0
                    self[z + spec_name + '.index.limits'] = [0, 0]

            self[z0 + 'snapshot.index'] = part_z0.snapshot['index']
            self[z + 'snapshot.index'] = part_z.snapshot['index']

            # initialize pointer indices
            # set null values safely to negative, so will trip an index error if try to use
            self[self.pointer_index_name] = ut.array.get_array_null(self[z0 + 'particle.number'])

    def get_pointers(
        self,
        species_name_from='star',
        species_names_to='star',
        part_indices=None,
        forward=False,
        intermediate_snapshot=False,
        return_single_array=True,
    ):
        '''
        Get pointer indices (and species) from species_name_from particles at the
        reference (later) snapshot to species_names_to particles the earlier snapshot.
        If enable forward, get pointers going forward in time (from z to z_ref) instead.

        Parameters
        ----------
        species_name_from : str
            name of species at the reference (later, z0) snapshot
        species_names_to : str or list
            name[s] of species to get pointers to at the (earlier, z) snapshot
        part_indices : arr
            indices of particles at the reference (later, z0) snapshot
        forward : bool
            whether to get pointers from the (earlier, z) snapshot to the reference (later, z0)
            snapshot, that is, tracking forward in time default (forward=False) is tracking
            backwards in time
        intermediate_snapshot : bool
            whether to get pointers between z and an intermediate snapshot (at z > 0)
            default (intermediate_snapshot=False) is to get pointers to/from z0
        return_single_array : bool
            if tracking single species at both snapshots, whether to return single array of
            pointer indices (instead of dictionary of pointers that includes species names)

        Returns
        -------
        pointer : arr or dict
            array of pointer indices between snapshots
            OR
            dictionary that contains both pointer indices and species names
        '''
        # parse inputs
        assert np.isscalar(species_name_from)

        if species_names_to is None or len(species_names_to) == 0:
            species_names_to = species_name_from
        elif species_names_to == 'all':
            species_names_to = self['species']
        if np.isscalar(species_names_to):
            species_names_to = [species_names_to]

        if intermediate_snapshot:
            z_ref_name = self.zi_name
            # if self.zi_name + species_name_from + '.number' not in self:
            #    self.add_intermediate_pointers()
        else:
            z_ref_name = self.z0_name

        if forward:
            # track forward in time, from snapshot z to the reference (z0) snapshot
            z_from = self.z_name
            z_to = z_ref_name
            if (
                species_name_from == 'star'
                and len(species_names_to) == 1
                and species_names_to[0] == 'gas'
            ):
                self.say('! gas particles cannot have star particle progenitors')
                return
        else:
            # track backwards in time, from the reference (z0) snapshot to snapshot z
            z_from = z_ref_name
            z_to = self.z_name
            if (
                species_name_from == 'gas'
                and len(species_names_to) == 1
                and species_names_to[0] == 'star'
            ):
                self.say('! gas particles cannot have star particle progenitors')
                return

        pointer_index_name = z_from + 'to.' + z_to + 'index'
        if forward and pointer_index_name not in self:
            self.assign_forward_pointers(intermediate_snapshot)

        if part_indices is None:
            part_indices = ut.array.get_arange(self[z_from + species_name_from + '.number'])

        # if tracking multiple species, adjust input particle indices to be concatenated indices
        part_indices = part_indices + self[z_from + species_name_from + '.index.limits'][0]

        # store as pointer species and indices as dictionary
        pointer = {}

        # get pointer indices (concatenated, if tracking multiple species)
        pointer['index'] = self[pointer_index_name][part_indices]

        # if tracking multiple species, initialize species names
        if len(species_names_to) > 1:
            pointer['species'] = np.zeros(part_indices.size, dtype='<U4')

        for spec_name in species_names_to:
            # get pointer indices for this species
            pis = ut.array.get_indices(
                pointer['index'], self[z_to + spec_name + '.index.limits'], verbose=False
            )
            # adjust back to particle indices
            pointer['index'][pis] -= self[z_to + spec_name + '.index.limits'][0]
            if len(species_names_to) > 1:
                # if tracking multiple species, assign species names
                pointer['species'][pis] = spec_name
            else:
                # tracking single species - set pointers to other species to null (safely negative)
                pis = np.setdiff1d(np.arange(part_indices.size), pis)
                pointer['index'][pis] = -pointer['index'].max() - 1

        # if tracking single species, can return just array of pointer indices
        if len(species_names_to) == 1 and return_single_array:
            pointer = pointer['index']

        return pointer

    def add_intermediate_pointers(self, Pointer):
        '''
        Add pointers between an intermediate snapshot (zi) and the earlier snapshot (z),
        to allow tracking between these 2 snapshots at z > 0.
        The intermediate snapshot (zi) must be between the reference (z0) snapshot and the earlier
        (z) snapshot.

        Parameters
        ----------
        Pointer : dict class
            pointers to an intemediate snapshot (between z0 and z)
        '''
        assert Pointer[Pointer.z_name + 'snapshot.index'] < self[self.z0_name + 'snapshot.index']
        assert Pointer[Pointer.z_name + 'snapshot.index'] > self[self.z_name + 'snapshot.index']

        for prop_name in Pointer:
            if self.z_name in prop_name:
                self[prop_name.replace(self.z_name, self.zi_name)] = Pointer[prop_name]

        z = self.z_name
        z0 = self.z0_name

        if z + 'to.' + z0 + 'index' not in Pointer:
            Pointer.assign_forward_pointers()
        pointer_indices_from = Pointer[z + 'to.' + z0 + 'index']
        pointer_indices_to = self[z0 + 'to.' + z + 'index']
        self[self.zi_name + 'to.' + z + 'index'] = pointer_indices_to[pointer_indices_from]

    def assign_forward_pointers(self, intermediate_snapshot=False):
        '''
        Assign pointer indices going forward in time, from the earlier (z) snapshot to the
        reference (later) snapshot.
        Currently, if gas particles split, assigns only one split gas particle as a descendant.
        TODO: deal with gas particle splitting

        Parameters
        ----------
        intermediate_snapshot : bool
            whether to get pointers between z and an intermediate snapshot (at z > 0)
        '''
        if intermediate_snapshot:
            z_ref = self.zi_name
        else:
            z_ref = self.z0_name

        z = self.z_name

        # get pointers that have valid (non-null) values
        masks_valid = self[z_ref + 'to.' + z + 'index'] >= 0
        pointers_valid = self[z_ref + 'to.' + z + 'index'][masks_valid]

        # sanity check
        if pointers_valid.max() >= self[z + 'particle.number']:
            self.say(
                '! particle dictionary at snapshot {} has {} valid pointers'.format(
                    self[z + 'snapshot.index'], self[z + 'particle.number']
                )
            )
            self.say(f'but {z_ref}->{z} pointer index max = {pointers_valid.max()}')
            self.say(
                'thus, {}->{} pointers do not point to all particles at snapshot {}'.format(
                    z_ref, z, self[z + 'snapshot.index']
                )
            )
            self.say('increasing size of reverse pointer array to accomodate missing particles')
            z_particle_number = pointers_valid.max() + 1
        else:
            z_particle_number = self[z + 'particle.number']

        # initialize pointer indices
        # set null values safely to negative, so will trip an index error if try to use
        self[z + 'to.' + z_ref + 'index'] = ut.array.get_array_null(z_particle_number)
        self[z + 'to.' + z_ref + 'index'][pointers_valid] = ut.array.get_arange(
            self[z_ref + 'to.' + z + 'index'].size
        )[masks_valid]


class ParticlePointerClass(ut.io.SayClass):
    '''
    Read or write particle pointer indicies (and species names), for tracking star and gas particles
    across snapshots.
    '''

    def __init__(
        self,
        species_names=['star', 'gas'],
        simulation_directory=gizmo_default.simulation_directory,
        track_directory=gizmo_default.track_directory,
        snapshot_directory=gizmo_default.snapshot_directory,
        reference_snapshot_index=gizmo_default.snapshot_index,
    ):
        '''
        Parameters
        ----------
        species_names : str or list
            name[s] of particle species to track
        simulation_directory : str
            directory of simulation
        track_directory : str
            directory of files for particle pointers and formation coordinates
        snapshot_directory : str
            directory of snapshot files (within simulation directory)
        reference_snapshot_index : int
            index of reference (later) snapshot to compute particle pointers relative to
        '''
        self.id_name = ID_NAME
        self.id_child_name = ID_CHILD_NAME
        self.properties_read = [self.id_name, self.id_child_name]
        if np.isscalar(species_names):
            species_names = [species_names]  # ensure is list
        self.species_names = species_names
        self.simulation_directory = ut.io.get_path(simulation_directory)
        self.track_directory = ut.io.get_path(track_directory)
        self.snapshot_directory = ut.io.get_path(snapshot_directory)
        self.reference_snapshot_index = reference_snapshot_index

        self.diagnostic = {}

        self.GizmoRead = gizmo_io.ReadClass()

    def io_pointers(
        self,
        part=None,
        snapshot_index=None,
        Pointer=None,
        simulation_directory=None,
        track_directory=None,
        verbose=False,
    ):
        '''
        Read or write, for each star particle at the reference (later, z0) snapshot
        its pointer index (and species name) to the other (earlier, z) snapshot.
        If input particle catalog (part), append pointers as dictionary class to part,
        else return pointers as a dictionary class.

        Parameters
        ----------
        part : dict
            catalog of particles at a the (earlier, z) snapshot
        snapshot_index : int
            index of the (earlier, z) snapshot to read
        Pointer : dict class
            particle pointers (if writing)
        simulation_directory : str
            directory of simulation
        track_directory : str
            directory of files for particle pointers and formation coordinates
        verbose : bool
            whether to print diagnostic information

        Returns
        -------
        Pointer : dict class
            particle pointers
        '''
        if part is not None:
            snapshot_index = part.snapshot['index']
        elif Pointer is not None:
            snapshot_index = Pointer['z.snapshot.index']
        else:
            assert snapshot_index is not None

        file_name = ''
        for spec_name in self.species_names:
            file_name += f'{spec_name}_'
        file_name += 'pointers_{:03d}'.format(snapshot_index)

        if simulation_directory is None:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)

        if track_directory is None:
            track_directory = self.track_directory
        else:
            track_directory = ut.io.get_path(track_directory)

        path_file_name = simulation_directory + track_directory + file_name

        if Pointer is not None:
            # write to file
            track_directory = ut.io.get_path(track_directory, create_path=True)
            for prop_name in Pointer:
                # hdf5 writer needs to receive numpy arrays
                Pointer[prop_name] = np.asarray(Pointer[prop_name])
                if prop_name == 'species':
                    # hdf5 writer does not support unicode
                    Pointer[prop_name] = Pointer[prop_name].astype('|S4')
            ut.io.file_hdf5(path_file_name, Pointer)

        else:
            # read from file
            dict_read = ut.io.file_hdf5(path_file_name, verbose=verbose)

            self.say(
                '* read particle pointers from:  {}.hdf5'.format(
                    simulation_directory.lstrip('./') + track_directory + file_name
                )
            )

            Pointer = ParticlePointerDictionaryClass()
            for k in dict_read:
                if 'number' in k or 'snapshot.index' in k:
                    Pointer[k] = dict_read[k].item()  # convert to float/int
                elif k == 'species':
                    Pointer[k] = dict_read[k].astype('<U4')  # store as unicode
                else:
                    Pointer[k] = dict_read[k]

            if part is None:
                return Pointer
            else:
                part.Pointer = Pointer

    def read_pointers_between_snapshots(
        self, snapshot_index_from, snapshot_index_to, species_name='star', simulation_directory=None
    ):
        '''
        Get particle pointer indices for single species between any two snapshots.
        Given input snapshot indices, get array of pointer indices from snapshot_index_from to
        snapshot_index_to.

        Parameters
        ----------
        snapshot_index_from : int
            snapshot index to get pointers from
        snapshot_index_to : int
            snapshot index to get pointers to
        species_name : str
            name of particle species to track
        simulation_directory : str
            directory of simulation

        Returns
        -------
        part_pointers : array
            particle pointer indices from snapshot_index_from to snapshot_index_to
        '''
        if snapshot_index_from > snapshot_index_to:
            forward = False
        else:
            forward = True

        if simulation_directory is None:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)

        PointerTo = self.io_pointers(
            snapshot_index=snapshot_index_to, simulation_directory=simulation_directory
        )

        if self.reference_snapshot_index in [snapshot_index_to, snapshot_index_from]:
            pointer_indices = PointerTo.get_pointers(
                species_name, species_name, forward=forward, return_single_array=True
            )
        else:
            PointerFrom = self.io_pointers(
                snapshot_index=snapshot_index_from, simulation_directory=simulation_directory
            )

            if species_name == 'star':
                # pointers from z_from to the reference (later, z0) snapshot
                pointer_indices_from = PointerFrom.get_pointers(
                    species_name, species_name, forward=True, return_single_array=True
                )
                # pointers from the reference (later, z0) snapshot to z_to
                pointer_indices_to = PointerTo.get_pointers(
                    species_name, species_name, return_single_array=True
                )
                # pointers from z_from to z_to
                pointer_indices = pointer_indices_to[pointer_indices_from]
            else:
                # trickier case - use internal functions
                if snapshot_index_from > snapshot_index_to:
                    PointerZ1 = PointerFrom
                    PointerZ2 = PointerTo
                else:
                    PointerZ2 = PointerFrom
                    PointerZ1 = PointerTo

                PointerZ2.add_intermediate_pointers(PointerZ1)
                pointer_indices = PointerZ2.get_pointers(
                    species_name,
                    species_name,
                    forward=forward,
                    intermediate_snapshot=True,
                    return_single_array=True,
                )

        return pointer_indices

    def generate_write_pointers(self, snapshot_indices='all', proc_number=1):
        '''
        Assign to each particle a pointer from its index at the reference (later) snapshot
        to its index (and species name) at all other (earlier) snapshots.
        Write particle pointers to file, one file for each snapshot besides the reference snapshot.

        Parameters
        ----------
        snapshot_indices : array-like
            snapshot indices at which to assign pointers
        proc_number : int
            number of parallel processes to run
        '''
        Snapshot = ut.simulation.read_snapshot_times(self.simulation_directory)

        if isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]

        # get list of snapshot indices to assign
        if snapshot_indices == 'all' or snapshot_indices is None or len(snapshot_indices) == 0:
            snapshot_indices = Snapshot['index']
        else:
            snapshot_indices = np.array(snapshot_indices)
            if np.max(snapshot_indices) > self.reference_snapshot_index:
                self.reference_snapshot_index = np.max(snapshot_indices)
                self.say(
                    f'setting reference snapshot to max input index = {np.max(snapshot_indices)}'
                )

        # read particles at the reference snapshot (typically z = 0)
        part_z0 = self.GizmoRead.read_snapshots(
            self.species_names,
            'index',
            self.reference_snapshot_index,
            snapshot_directory=self.snapshot_directory,
            properties=self.properties_read,
            assign_hosts=False,
            check_properties=False,
        )
        for spec_name in self.species_names:
            part_z0[spec_name]._assign_ids_to_indices()

        # skip assigning pointers to this reference snapshot
        snapshot_indices = np.setdiff1d(snapshot_indices, part_z0.snapshot['index'])
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # counters for diagnostics
        self.diagnostic = {
            'no.id.match.number': 0,
            'bad.snapshots': [],
        }

        if proc_number > 1:
            # initiate threads
            from multiprocessing import Pool

            with Pool(proc_number) as pool:
                for snapshot_index in snapshot_indices:
                    # memory errors if try to pass part_z0, so instead re-read part_z0 per thread
                    pool.apply_async(
                        self._generate_write_pointers_to_snapshot, (None, snapshot_index)
                    )
        else:
            for snapshot_index in snapshot_indices:
                self._generate_write_pointers_to_snapshot(part_z0, snapshot_index)

        # print cumulative diagnostics
        print()
        self.say(
            '! {} total particles did not have id match'.format(
                self.diagnostic['no.id.match.number']
            )
        )
        if len(self.diagnostic['bad.snapshots']) > 0:
            self.say(
                '! could not read these snapshots:  {}'.format(self.diagnostic['bad.snapshots'])
            )
            self.say('(missing or corrupt files) so could not assign pointers to those snapshots')

    def _generate_write_pointers_to_snapshot(self, part_z0, snapshot_index):
        '''
        Assign to each particle a pointer from its index at the reference (later, z0) snapshot
        to its index (and species name) at a (earlier, z) snapshot.
        Write the particle pointers to file.

        Parameters
        ----------
        part_z0 : dict
            catalog of particles at the reference (later, z0) snapshot
        snapshot_index : int
            snapshot index to assign pointers to at the (earlier, z) snapshot
        count : dict
            total diagnostic counters across all snapshots
        '''
        # if not input, read particles at reference (z0) snaphsot
        if part_z0 is None:
            part_z0 = self.GizmoRead.read_snapshots(
                self.species_names,
                'index',
                self.reference_snapshot_index,
                snapshot_directory=self.snapshot_directory,
                properties=self.properties_read,
                assign_hosts=False,
                check_properties=False,
            )
            for spec_name in self.species_names:
                part_z0[spec_name]._assign_ids_to_indices()

        # read particles at this snapshot
        try:
            part_z = self.GizmoRead.read_snapshots(
                self.species_names,
                'index',
                snapshot_index,
                snapshot_directory=self.snapshot_directory,
                properties=self.properties_read,
                assign_hosts=False,
                check_properties=False,
            )
        except (IOError, TypeError):
            self.say(f'\n! can not read snapshot {snapshot_index} - missing or corrupt file')
            self.diagnostic['bad.snapshots'].append(snapshot_index)
            return

        # get list of species that have particles at this snapshot
        species_names_z = []
        for spec_name in self.species_names:
            if spec_name in part_z and len(part_z[spec_name][self.id_name]) > 0:
                species_names_z.append(spec_name)
            else:
                self.say(f'! no {spec_name} particles at snapshot {snapshot_index}')
        if len(species_names_z) == 0:
            return

        # initialize dictionary class to store pointers and meta-data
        Pointer = ParticlePointerDictionaryClass(part_z0, part_z, self.species_names)

        for spec_name in species_names_z:
            # get particle index offest (non-zero if concatenating multiple species)
            species_index_offset = Pointer[Pointer.z_name + spec_name + '.index.limits'][0]
            part_z_indices = np.arange(part_z[spec_name][self.id_name].size)
            part_z_total_indices = part_z_indices + species_index_offset
            part_z_ids = part_z[spec_name][self.id_name]
            part_z_cids = part_z[spec_name][self.id_child_name]

            # get particle indices at z0 within each species dictionary
            part_z0_indices, part_z0_species = part_z0.get_pointers_from_ids(
                part_z_ids, part_z_cids
            )

            # convert to total (concatenated) index at z0
            for spec_name in self.species_names:
                indices = np.where(part_z0_species == spec_name)[0]
                species_index_offset = Pointer[Pointer.z0_name + spec_name + '.index.limits'][0]
                part_z0_indices[indices] += species_index_offset

            # assign pointers
            indices = np.where(part_z0_indices >= 0)[0]
            Pointer[Pointer.pointer_index_name][part_z0_indices[indices]] = part_z_total_indices[
                indices
            ]

            no_id_match_number = np.sum(part_z0_indices < 0)
            if no_id_match_number > 0:
                self.say(
                    '! {} (of {}) {} particles at snapshot {} do not have id match'.format(
                        no_id_match_number,
                        Pointer[Pointer.z_name + 'particle.number'],
                        species_names_z,
                        snapshot_index,
                    )
                )
                self.diagnostic['no.id.match.number'] += no_id_match_number

        # write file for this snapshot
        self.io_pointers(Pointer=Pointer)


def test_particle_pointers(part, part_z1, part_z2):
    '''
    .
    '''
    ParticlePointer = ParticlePointerClass()
    ParticlePointer.io_pointers(part_z1)
    ParticlePointer.io_pointers(part_z2)

    part_z2.Pointer.add_intermediate_pointers(part_z1.Pointer)

    for spec_from in ['star', 'gas']:
        pointer_z1 = part_z1.Pointer.get_pointers(spec_from, 'all')
        pointer_z2 = part_z2.Pointer.get_pointers(spec_from, 'all')
        assert part[spec_from]['id'].size == pointer_z1['index'].size
        assert part[spec_from]['id'].size == pointer_z2['index'].size

        for spec_to in ['star', 'gas']:
            pis_z1 = np.where(pointer_z1['species'] == spec_to)[0]
            pis_z2 = np.where(pointer_z2['species'] == spec_to)[0]
            if pis_z1.size:
                masks = (
                    part[spec_from]['id'][pis_z1]
                    != part_z1[spec_to]['id'][pointer_z1['index'][pis_z1]]
                )
                if np.max(masks):
                    print('z0->z1', spec_from, spec_to, np.sum(masks))
            if pis_z2.size:
                masks = (
                    part[spec_from]['id'][pis_z2]
                    != part_z2[spec_to]['id'][pointer_z2['index'][pis_z2]]
                )
                if np.max(masks):
                    print('z0->z2', spec_from, spec_to, np.sum(masks))

        pointer_z1 = part_z1.Pointer.get_pointers(spec_from, 'all', forward=True)
        pointer_z2 = part_z2.Pointer.get_pointers(spec_from, 'all', forward=True)
        assert part_z1[spec_from]['id'].size == pointer_z1['index'].size
        assert part_z2[spec_from]['id'].size == pointer_z2['index'].size

        for spec_to in ['star', 'gas']:
            pis_z1 = np.where(pointer_z1['species'] == spec_to)[0]
            pis_z2 = np.where(pointer_z2['species'] == spec_to)[0]

            if pis_z1.size:
                masks = (
                    part_z1[spec_from]['id'][pis_z1]
                    != part[spec_to]['id'][pointer_z1['index'][pis_z1]]
                )
                if np.max(masks):
                    print('z1->z0', spec_from, spec_to, np.sum(masks))

            if pis_z2.size:
                masks = (
                    part_z2[spec_from]['id'][pis_z2]
                    != part[spec_to]['id'][pointer_z2['index'][pis_z2]]
                )
                if np.max(masks):
                    print('z2->z0', spec_from, spec_to, np.sum(masks))

        pointer_z2 = part_z2.Pointer.get_pointers(spec_from, 'all', intermediate_snapshot=True)
        assert part_z1[spec_from]['id'].size == pointer_z2['index'].size

        for spec_to in ['star', 'gas']:
            pis_z2 = np.where(pointer_z2['species'] == spec_to)[0]

            if pis_z2.size:
                masks = (
                    part_z1[spec_from]['id'][pis_z2]
                    != part_z2[spec_to]['id'][pointer_z2['index'][pis_z2]]
                )
                if np.max(masks):
                    print('z1->z2', spec_from, spec_to, np.sum(masks))

        pointer_z2 = part_z2.Pointer.get_pointers(
            spec_from, 'all', intermediate_snapshot=True, forward=True
        )
        assert part_z2[spec_from]['id'].size == pointer_z2['index'].size

        for spec_to in ['star', 'gas']:
            pis_z2 = np.where(pointer_z2['species'] == spec_to)[0]
            if pis_z2.size:
                masks = (
                    part_z2[spec_from]['id'][pis_z2]
                    != part_z1[spec_to]['id'][pointer_z2['index'][pis_z2]]
                )
                if np.max(masks):
                    print('z2->z1', spec_from, spec_to, np.sum(masks))


class ParticleCoordinateClass(ut.io.SayClass):
    '''
    Select member particles in each host galaxy at the reference snapshot (usually z = 0).
    Tracking back only these particles, compute the position, velocity, and principal axes of each
    host at each previous snapshot.
    Then compute the 3-D distance and 3-D velocity wrt each primary host galaxy for each particle
    at the snapshot after it forms.
    '''

    def __init__(
        self,
        species_name='star',
        simulation_directory=gizmo_default.simulation_directory,
        track_directory=gizmo_default.track_directory,
        snapshot_directory=gizmo_default.snapshot_directory,
        reference_snapshot_index=gizmo_default.snapshot_index,
        host_distance_limits=[0, 30],
        host_edge_percent=90,
    ):
        '''
        Parameters
        ----------
        species : str
            name of particle species to track
        simulation_directory : str
            directory of simulation
        track_directory : str
            directory of files for particle pointers and formation coordinates
        reference_snapshot_index : float
            index of reference (later) snapshot to compute particle pointers from
        snapshot_directory : str
            directory of snapshot files (within simulation directory)
        host_distance_limits : list
            min and max distance [kpc physical] to select particles near each primary host at the
            reference snapshot (usually z = 0).
            Use only these particles to compute host coordinates at earlier snapshots.
        host_edge_percent : float
            percent of species mass (within initial aperture) to define host galaxy radius + height
        '''
        self.id_name = ID_NAME
        self.id_child_name = ID_CHILD_NAME
        self.species_name = species_name
        assert np.isscalar(self.species_name)
        self.simulation_directory = ut.io.get_path(simulation_directory)
        self.track_directory = ut.io.get_path(track_directory)
        self.snapshot_directory = ut.io.get_path(snapshot_directory)
        self.reference_snapshot_index = reference_snapshot_index
        self.host_distance_limits = host_distance_limits
        self.host_edge_percent = host_edge_percent
        self.host_properties = [
            'position',
            'velocity',
            'rotation',
            'axis.ratios',
            f'radius.{self.host_edge_percent}',
            f'height.{self.host_edge_percent}',
            f'mass.{self.host_edge_percent}',
        ]
        # numpy data type to store host and particle coordinates
        self.coordinate_dtype = np.float32
        # names of distances and velocities to write/read
        self.formation_coordiante_kinds = ['form.host.distance', 'form.host.velocity']

        self.GizmoRead = gizmo_io.ReadClass()

    def io_hosts_coordinates(
        self,
        part,
        simulation_directory=None,
        track_directory=None,
        assign_formation_coordinates=False,
        write=False,
        verbose=False,
    ):
        '''
        For each host, read or write its position, velocity, and principal axes at each snapshot,
        computed tracking back only member particles at the reference snapshot (z = 0).
        If formation_coordinates is True, or each particle, read or write its 3-D distance and
        3-D velocity wrt each host galaxy at the first snapshot after it formed,
        aligned with (rotated into) the principal axes of each host at that time.
        If reading, assign to input particle dictionary.

        Parameters
        ----------
        part : dict
            catalog of particles at a snapshot
        simulation_directory : str
            directory of simulation
        track_directory : str
            directory of files for particle pointers and formation coordinates
        assign_formation_coordinates : bool
            whether to read and assign the formation coordinates for each particle
        write : bool
            whether to write to file (instead of read)
        verbose : bool
            whether to print diagnostic information
        '''
        if simulation_directory is None:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)

        if track_directory is None:
            track_directory = self.track_directory
        else:
            track_directory = ut.io.get_path(track_directory)

        path_file_name = (
            simulation_directory + track_directory + gizmo_default.hosts_coordinates_file_name
        )

        if write:
            track_directory = ut.io.get_path(track_directory, create_path=True)
            dict_write = collections.OrderedDict()
            dict_write['snapshot.index'] = np.array(part.snapshot['index'])
            dict_write[self.species_name + '.id'] = part[self.species_name][self.id_name]
            for prop_name in part[self.species_name]:
                if 'form.host' in prop_name:
                    dict_write[self.species_name + '.' + prop_name] = part[self.species_name][
                        prop_name
                    ]
            for prop_name in part[self.species_name].hostz:
                dict_write['host.' + prop_name] = part[self.species_name].hostz[prop_name]

            ut.io.file_hdf5(path_file_name, dict_write)

        else:
            # read
            dict_read = ut.io.file_hdf5(path_file_name, verbose=verbose)

            # initialize dictionaries to store host properties across snapshots
            part.hostz = {}
            for spec_name in part:
                part[spec_name].hostz = {}
            for prop_name in self.host_properties:
                part.hostz[prop_name] = []
                for spec_name in part:
                    part[spec_name].hostz[prop_name] = []

            for prop_name in dict_read:
                if prop_name.lstrip('host.') in part.hostz:
                    # assign hosts' coordinates
                    prop_name_store = prop_name.lstrip('host.')
                    part.hostz[prop_name_store] = dict_read[prop_name]
                    part.host[prop_name_store] = part.hostz[prop_name_store][part.snapshot['index']]
                    for spec_name in part:
                        part[spec_name].hostz[prop_name_store] = dict_read[prop_name]
                        part[spec_name].host[prop_name_store] = part.host[prop_name_store]

            host_number = part.hostz['position'].shape[1]
            host_string = 'host'
            if host_number > 1:
                host_string += 's'
            self.say(
                f'read {host_number} {host_string} (position, velocity, principal axes) from:'
                + '  {}'.format(path_file_name.lstrip('./'))
            )

            for host_i, host_position in enumerate(part.host['position']):
                self.say(f'host{host_i + 1} position = (', end='')
                ut.io.print_array(host_position, '{:.2f}', end='')
                print(') [kpc comoving]')

            for host_i, host_velocity in enumerate(part.host['velocity']):
                self.say(f'host{host_i + 1} velocity = (', end='')
                ut.io.print_array(host_velocity, '{:.1f}', end='')
                print(') [km / s]')

            for host_i, host_axis_ratios in enumerate(part.host['axis.ratios']):
                self.say(f'host{host_i + 1} axis ratios = (', end='')
                ut.io.print_array(host_axis_ratios, '{:.2f}', end='')
                print(')')

            if 'radius.90' in part.host and len(part.host['radius.90']) > 0:
                for host_i, host_radius90 in enumerate(part.host['radius.90']):
                    self.say('host{} R_90 = {:.1f} kpc'.format(host_i + 1, host_radius90))

            if 'height.90' in part.host and len(part.host['height.90']) > 0:
                for host_i, host_height90 in enumerate(part.host['height.90']):
                    self.say('host{} Z_90 = {:.1f} kpc'.format(host_i + 1, host_height90))

            if 'mass.90' in part.host and len(part.host['mass.90']) > 0:
                for host_i, host_mass90 in enumerate(part.host['mass.90']):
                    self.say('host{} M_90 = {:.1e} Msun'.format(host_i + 1, host_mass90))

            if assign_formation_coordinates:
                self.say(
                    f'\n  read formation coordinates for {self.species_name} particles'
                    + ' at snapshot {}'.format(dict_read['snapshot.index'])
                )
                for prop_name in dict_read:
                    if 'form.' in prop_name:
                        # store coordinates at formation
                        prop_name_store = prop_name.lstrip(self.species_name + '.')
                        part[self.species_name][prop_name_store] = dict_read[prop_name]

                    elif '.id' in prop_name:
                        mismatch_id_number = np.sum(
                            part[self.species_name][self.id_name] != dict_read[prop_name]
                        )
                        if mismatch_id_number > 0:
                            self.say(
                                f'! {mismatch_id_number} {prop_name}s are mis-matched between'
                                + ' particles read and input particle dictionary'
                            )
                            self.say(
                                'you likely are assigning formation coordinates to the wrong'
                                + ' simulation or snapshot'
                            )

    def generate_write_hosts_coordinates(
        self, part_z0=None, host_number=1, proc_number=1, simulation_directory=None
    ):
        '''
        Select member particles in each host galaxy at the reference snapshot (usually z = 0).
        Tracking back only these particles, compute the coordinates and principal axes of each host
        at each previous snapshot.
        Also compute the 3-D distance and 3-D velocity wrt each primary host galaxy (rotated into
        its principle axes) for each particle and write to file.
        Work backwards in time and over-write existing values, so for each particle keep only its
        coordinates at the first snapshot after it formed.

        Parameters
        ----------
        part : dict
            catalog of particles at the reference snapshot
        host_number : int
            number of host galaxies to assign and compute coordinates relative to
        proc_number : int
            number of parallel processes to run
        simulation_directory : str
            base directory of simulation
        '''
        # if 'elvis' is in simulation directory name, force 2 hosts
        host_number = ut.catalog.get_host_number_from_directory(host_number, './', os)

        if simulation_directory is None:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)

        if part_z0 is None:
            # read particles at z = 0
            part_z0 = self.GizmoRead.read_snapshots(
                self.species_name,
                'index',
                self.reference_snapshot_index,
                simulation_directory,
                self.snapshot_directory,
                properties=[
                    self.id_name,
                    self.id_child_name,
                    'position',
                    'velocity',
                    'mass',
                    'form.scalefactor',
                ],
                host_number=host_number,
                assign_hosts='mass',
                check_properties=False,
            )

        # get list of snapshots to assign
        snapshot_indices = part_z0.Snapshot['index']
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # initialize and store position, velocity, principal axes rotation tensor + axis ratios
        # of each primary host galaxy at each snapshot
        part_z0[self.species_name].hostz = {}

        for prop_name in self.host_properties:
            if prop_name in ['position', 'velocity', 'axis.ratios']:
                part_z0[self.species_name].hostz[prop_name] = (
                    np.zeros(
                        [part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype
                    )
                    + np.nan
                )
            elif prop_name == 'rotation':
                part_z0[self.species_name].hostz[prop_name] = (
                    np.zeros(
                        [part_z0.Snapshot['index'].size, host_number, 3, 3], self.coordinate_dtype
                    )
                    + np.nan
                )
            elif 'radius' in prop_name or 'height' in prop_name or 'mass' in prop_name:
                part_z0[self.species_name].hostz[prop_name] = (
                    np.zeros([part_z0.Snapshot['index'].size, host_number], self.coordinate_dtype)
                    + np.nan
                )

        # initialize and store indices of particles near all primary hosts at the reference snapshot
        hosts_part_z0_indicess = []

        for host_index in range(host_number):
            host_name = ut.catalog.get_host_name(host_index)

            part_z0_indices = ut.array.get_indices(
                part_z0[self.species_name].prop(host_name + 'distance.total'),
                self.host_distance_limits,
            )
            hosts_part_z0_indicess.append(part_z0_indices)

            # initialize and store particle formation coordinates
            for prop_name in self.formation_coordiante_kinds:
                prop_name = prop_name.replace('host.', host_name)  # update host name (if necessary)
                part_z0[self.species_name][prop_name] = (
                    np.zeros(part_z0[self.species_name]['position'].shape, self.coordinate_dtype)
                    + np.nan
                )

        count = {'id none': 0, 'id wrong': 0, 'bad.snapshots': []}

        # initiate threads, if asking for > 1
        if proc_number > 1:
            from multiprocessing import Pool

            with Pool(proc_number) as pool:
                for snapshot_index in snapshot_indices:
                    pool.apply(
                        self._generate_write_hosts_coordinates_at_snapshot,
                        (part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count),
                    )
        else:
            for snapshot_index in snapshot_indices:
                self._generate_write_hosts_coordinates_at_snapshot(
                    part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count
                )

        # print cumulative diagnostics
        print()
        if len(count['bad.snapshots']) > 0:
            self.say('! could not read these snapshots:  {}'.format(count['bad.snapshots']))
            self.say('(missing or corrupt files) so could not assign pointers to those snapshots')
        if count['id none']:
            self.say('! {} total particles did not have valid id'.format(count['id none']))
        if count['id wrong']:
            self.say('! {} total particles did not have id match'.format(count['id wrong']))

    def _generate_write_hosts_coordinates_at_snapshot(
        self, part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count_tot
    ):
        '''
        Compute the coordinates and principal axes of each host at snapshot_index.
        Also compute the 3-D distance and 3-D velocity wrt each primary host galaxy (rotated into
        its principle axes) for each particle at snapshot_index and write to file.

        Parameters
        ----------
        part_z0 : dict
            catalog of particles at the reference (latest) snapshot
        hosts_part_z0_indices : list of arrays
            indices of particles near each primary host at the reference (latest) snapshot
        host_number : int
            number of host galaxies to assign and compute coordinates relative to
        snapshot_index : int
            snapshot index at which to assign particle pointers to
        count_tot : dict
            diagnostic counters
        '''
        part_z0_indices = ut.array.get_arange(part_z0[self.species_name][self.id_name])

        if snapshot_index == part_z0.snapshot['index']:
            part_pointers = part_z0_indices
        else:
            # read pointer indices from reference snapshot to this snapshot
            ParticlePointer = ParticlePointerClass(
                simulation_directory=self.simulation_directory,
                track_directory=self.track_directory,
                reference_snapshot_index=self.reference_snapshot_index,
            )
            try:
                Pointer = ParticlePointer.io_pointers(snapshot_index=snapshot_index)
                part_pointers = Pointer.get_pointers(
                    self.species_name, self.species_name, return_single_array=True
                )
            except IOError:
                self.say(f'\n! can not read pointers to snapshot {snapshot_index}')
                return

        part_z0_indices = part_z0_indices[part_pointers >= 0]
        self.say(
            f'\n# assigning formation coordinates to {part_z0_indices.size} {self.species_name}'
            + f' particles at snapshot {snapshot_index}'
        )

        count = {'id none': 0, 'id wrong': 0}

        if part_z0_indices.size > 0:
            try:
                part_z = self.GizmoRead.read_snapshots(
                    self.species_name,
                    'index',
                    snapshot_index,
                    snapshot_directory=self.snapshot_directory,
                    properties=[self.id_name, 'position', 'velocity', 'mass', 'form.scalefactor'],
                    assign_hosts=False,
                    check_properties=False,
                )
            except (IOError, TypeError):
                self.say(f'\n! can not read snapshot {snapshot_index} - missing or corrupt file')
                count_tot['bad.snapshots'].append(snapshot_index)
                return

            # use only particles that are near each primary host at the reference snapshot
            # to compute the coordinates of each host progenitor at earlier snapshots
            hosts_part_z_indicess = []
            for host_i in range(host_number):
                hosts_part_z_indices = part_pointers[hosts_part_z0_indicess[host_i]]
                hosts_part_z_indices = hosts_part_z_indices[hosts_part_z_indices >= 0]
                if len(hosts_part_z_indices) == 0:
                    self.say(f'\n! no particles near host{host_i + 1} at snapshot {snapshot_index}')
                    return
                else:
                    hosts_part_z_indicess.append(hosts_part_z_indices)

            try:
                self.GizmoRead.assign_hosts_coordinates(
                    part_z,
                    'mass',
                    self.species_name,
                    hosts_part_z_indicess,
                    host_number=host_number,
                    exclusion_distance=None,
                )
            except Exception:
                # if not enough progenitor star particles near a host galaxy
                self.say(f'\n! can not compute host at snapshot {snapshot_index}')
                return

            if np.isnan(part_z.host['position']).max() or np.isnan(part_z.host['velocity']).max():
                self.say(f'\n! can not compute host at snapshot {snapshot_index}')
                return

            part_z_indices = part_pointers[part_z0_indices]

            # sanity checks
            masks = part_z_indices >= 0
            count['id none'] = part_z_indices.size - np.sum(masks)
            if count['id none']:
                self.say(
                    '! {} particles have no id match at snapshot {}'.format(
                        count['id none'], snapshot_index
                    )
                )
                part_z_indices = part_z_indices[masks]
                part_z0_indices = part_z0_indices[masks]

            masks = (
                part_z0[self.species_name][self.id_name][part_z0_indices]
                == part_z[self.species_name][self.id_name][part_z_indices]
            )
            count['id wrong'] = part_z_indices.size - np.sum(masks)
            if count['id wrong']:
                self.say(
                    '! {} particles have wrong id match at snapshot {}'.format(
                        count['id wrong'], snapshot_index
                    )
                )
                part_z_indices = part_z_indices[masks]
                part_z0_indices = part_z0_indices[masks]

            # assign hosts principal axes rotation tensor
            try:
                self.GizmoRead.assign_hosts_rotation(part_z)
            except ValueError:
                # if not enough progenitor star particles near a host galaxy
                self.say(f'\n! can not compute host (rotation) at snapshot {snapshot_index}')
                return

            # assign hosts size and mass - use host_edge_percent of species mass within an initial
            # radius of 20 kpc comoving, to approximate galaxy size growth
            distance_max = 20 * part_z.snapshot['scalefactor']
            for prop_name in self.host_properties:
                if 'radius' in prop_name or 'height' in prop_name or 'mass' in prop_name:
                    part_z.host[prop_name] = np.zeros(host_number, dtype=self.coordinate_dtype)
            try:
                for host_i in range(host_number):
                    gal = ut.particle.get_galaxy_properties(
                        part_z,
                        self.species_name,
                        edge_value=self.host_edge_percent,
                        axis_kind='both',
                        distance_max=distance_max,
                        host_index=host_i,
                    )
                    part_z.host[f'radius.{self.host_edge_percent}'][host_i] = gal['radius.major']
                    part_z.host[f'height.{self.host_edge_percent}'][host_i] = gal['radius.minor']
                    part_z.host[f'mass.{self.host_edge_percent}'][host_i] = gal['mass']
            except ValueError:
                # if not enough progenitor star particles near a host galaxy
                self.say(f'\n! can not compute host (size + mass) at snapshot {snapshot_index}')
                return

            # store host galaxy properties
            for prop_name in self.host_properties:
                part_z0[self.species_name].hostz[prop_name][snapshot_index] = part_z.host[prop_name]

            for host_i in range(host_number):
                # compute coordinates wrt primary host
                host_name = ut.catalog.get_host_name(host_i)

                for prop_name in self.formation_coordiante_kinds:
                    prop_name = prop_name.replace('host.', host_name)

                    if 'distance' in prop_name:
                        # 3-D distance wrt host in simulation's cartesian coordinates [kpc physical]
                        coordinates = ut.coordinate.get_distances(
                            part_z[self.species_name]['position'][part_z_indices],
                            part_z.host['position'][host_i],
                            part_z.info['box.length'],
                            part_z.snapshot['scalefactor'],
                        )

                    elif 'velocity' in prop_name:
                        # 3-D velocity wrt host in simulation's cartesian coordinates [km / s]
                        coordinates = ut.coordinate.get_velocity_differences(
                            part_z[self.species_name]['velocity'][part_z_indices],
                            part_z.host['velocity'][host_i],
                            part_z[self.species_name]['position'][part_z_indices],
                            part_z.host['position'][host_i],
                            part_z.info['box.length'],
                            part_z.snapshot['scalefactor'],
                            part_z.snapshot['time.hubble'],
                        )

                    # rotate coordinates to align with principal axes
                    coordinates = ut.coordinate.get_coordinates_rotated(
                        coordinates, part_z.host['rotation'][host_i]
                    )

                    # assign 3-D coordinates wrt primary host along principal axes [kpc physical]
                    part_z0[self.species_name][prop_name][part_z0_indices] = coordinates

                for k in count:
                    count_tot[k] += count[k]

            # continuously (re)write as go
            self.io_hosts_coordinates(part_z0, write=True)


# --------------------------------------------------------------------------------------------------
# run from command line
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function: pointer, coordinate, pointer+coordinate')

    function_kind = str(sys.argv[1])

    assert 'pointer' in function_kind or 'coordinate' in function_kind

    if 'pointer' in function_kind:
        ParticlePointer = ParticlePointerClass()
        ParticlePointer.generate_write_pointers()

    if 'coordinate' in function_kind:
        ParticleCoordinate = ParticleCoordinateClass()
        ParticleCoordinate.generate_write_hosts_coordinates()
