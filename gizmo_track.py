#!/usr/bin/env python3

'''
Track particles across snapshots in Gizmo simulations.

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
        part_z0 : dict : catalog of particles at the reference (later) snapshot
        part_z : dict : catalog of particles at the earlier snapshot
        species_names : str or list : name[s] of particle species to track
        id_name : str : dictionary key of particle id
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
        return_array=True,
    ):
        '''
        Get pointer indices (and species) from species_name_from particles at the
        reference (later) snapshot to species_names_to particles the earlier snapshot.
        If enable forward, get pointers going forward in time (from z to z_ref) instead.

        Parameters
        ----------
        species_name_from : str : name of species at the reference (later, z0) snapshot
        species_names_to : str or list :
            name[s] of species to get pointers to at the (earlier, z) snapshot
        part_indices : arr : indices of particles at the reference (later, z0) snapshot
        forward : bool : whether to get pointers from the (earlier, z) snapshot to the
            reference (later, z0) snapshot, that is, tracking forward in time
            default (forward=False) is tracking backwards in time
        intermediate_snapshot : bool :
            whether to get pointers between z and an intermediate snapshot (at z > 0)
            default (intermediate_snapshot=False) is to get pointers to/from z0
        return_array : bool : if tracking single species at both snapshots, return just array of
            pointer indices (and not a pointer dictionary that includes species names)

        Returns
        -------
        pointer : arr or dict :
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
        if len(species_names_to) == 1 and return_array:
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
        Pointer : dict class : pointers to an intemediate snapshot (between z0 and z)
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
        intermediate_snapshot : bool :
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
                '! particle catalog at snapshot {} has {} valid pointers'.format(
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
    Read/write particle pointer indicies (and species names), for tracking star and gas particles
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
        species_names : str or list : name[s] of particle species to track
        simulation_directory : str : directory of simulation
        track_directory : str : directory of files for particle pointers and formation coordinates
        snapshot_directory : str : directory of snapshot files (within simulation directory)
        reference_snapshot_index : int :
            index of reference (later) snapshot to compute particle pointers relative to
        '''
        self.id_name = ID_NAME
        self.species_names = species_names
        if np.isscalar(self.species_names):
            self.species_names = [self.species_names]  # ensure is list
        self.simulation_directory = ut.io.get_path(simulation_directory)
        self.track_directory = ut.io.get_path(track_directory)
        self.snapshot_directory = ut.io.get_path(snapshot_directory)
        self.reference_snapshot_index = reference_snapshot_index

        self.GizmoRead = gizmo_io.ReadClass()

        self.match_property = None
        self.match_propery_tolerance = None
        self.test_property = None

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
        part : dict : catalog of particles at a the (earlier, z) snapshot
        snapshot_index : int : index of the (earlier, z) snapshot to read
        Pointer : dict class : particle pointers (if writing)
        simulation_directory : str : directory of simulation
        track_directory : str : directory of files for particle pointers and formation coordinates
        verbose : bool : whether to print diagnostic information

        Returns
        -------
        Pointer : dict class : particle pointers
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

    def io_pointers_old_file_format(
        self,
        part=None,
        snapshot_index=None,
        part_pointers=None,
        simulation_directory=None,
        track_directory=None,
    ):
        '''
        This reads the old pointer files (star_indices_*.hdf5) and converts them into the new
        dictionary class format.

        Read or write, for each star particle at the reference snapshot (z0, usually z = 0),
        its pointer index to another snapshot (z).
        If input particle catalog (part), append pointers as dictionary class to part,
        else return pointers as a dictionary class.

        Parameters
        ----------
        part : dict : catalog of particles at a (non-reference) snapshot (z)
        snapshot_index : int : index of other (non-reference) snapshot to read
        Pointer : dict class : particle pointers (if writing)
        simulation_directory : str : directory of simulation
        track_directory : str: directory of files for particle pointers and formation coordinates

        Returns
        -------
        Pointer : dict class : particle pointers
        '''
        species_name = 'star'
        hdf5_dict_name = 'indices'

        if part is not None:
            snapshot_index = part.snapshot['index']
        elif not snapshot_index:
            raise ValueError('! need to input either particle catalog or snapshot_index')

        file_name = '{}_indices_{:03d}'.format(species_name, snapshot_index)

        if simulation_directory is None:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)

        if track_directory is None:
            track_directory = self.track_directory
        else:
            track_directory = ut.io.get_path(track_directory)

        if part_pointers is not None:
            # write to file
            track_directory = ut.io.get_path(track_directory, create_path=True)
            ut.io.file_hdf5(
                simulation_directory + track_directory + file_name, {hdf5_dict_name: part_pointers}
            )
        else:
            # read from file
            dict_read = ut.io.file_hdf5(simulation_directory + track_directory + file_name)

            Pointer = ParticlePointerDictionaryClass()
            particle_index_name = Pointer.pointer_index_name
            z0 = Pointer.z0_name
            z = Pointer.z_name
            Pointer[particle_index_name] = dict_read[hdf5_dict_name]

            part_z0_number = Pointer[particle_index_name].size
            Pointer['species'] = [species_name]
            Pointer[z0 + 'particle.number'] = part_z0_number
            Pointer[z0 + species_name + '.number'] = part_z0_number
            Pointer[z0 + species_name + '.index.limits'] = [0, part_z0_number]
            Pointer[z0 + 'snapshot.index'] = gizmo_default.snapshot_index

            part_z_number = np.sum(Pointer[particle_index_name] >= 0)
            Pointer[z + 'particle.number'] = part_z_number
            Pointer[z + species_name + '.number'] = part_z_number
            Pointer[z + species_name + '.index.limits'] = [0, part_z_number]
            Pointer[z + 'snapshot.index'] = snapshot_index

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
        snapshot_index_from : int : snapshot index to get pointers from
        snapshot_index_to : int : snapshot index to get pointers to
        species_name : str : name of particle species to track
        simulation_directory : str : directory of simulation

        Returns
        -------
        part_pointers : array :
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
                species_name, species_name, forward=forward, return_array=True
            )
        else:
            PointerFrom = self.io_pointers(
                snapshot_index=snapshot_index_from, simulation_directory=simulation_directory
            )

            if species_name == 'star':
                # pointers from z_from to the reference (later, z0) snapshot
                pointer_indices_from = PointerFrom.get_pointers(
                    species_name, species_name, forward=True, return_array=True
                )
                # pointers from the reference (later, z0) snapshot to z_to
                pointer_indices_to = PointerTo.get_pointers(
                    species_name, species_name, return_array=True
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
                    return_array=True,
                )

        return pointer_indices

    def write_pointers_to_snapshots(
        self,
        part_z0=None,
        match_property='id.child',
        match_propery_tolerance=1e-6,
        test_property='form.scalefactor',
        snapshot_indices=[],
        proc_number=1,
    ):
        '''
        Assign to each particle a pointer from its index at the reference (later) snapshot
        to its index (and species name) at all other (earlier) snapshots,
        to track particles across time.
        Write particle pointers to file, one file for each snapshot besides the reference snapshot.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at reference (later, z0) snapshot
        match_property : str :
            some particles have the same id, so this is the property to use to match them.
            options (in order of preference): 'id.child', 'form.scalefactor', 'massfraction.metals'
        match_propery_tolerance : float : fractional tolerance for matching via match_property
        test_property : str : additional property to use to test matching
        snapshot_indices : array-like : snapshot indices at which to assign pointers
        proc_number : int : number of parallel processes to run
        '''
        assert match_property in ['id.child', 'massfraction.metals', 'form.scalefactor']

        if part_z0 is None:
            # read particles at reference snapshot (typically z = 0)
            # get list of properties relevant to use in matching
            properties_read = [self.id_name, 'id.child']
            if match_property not in properties_read:
                properties_read.append(match_property)
            if test_property and test_property not in properties_read:
                properties_read.append(test_property)
            part_z0 = self.GizmoRead.read_snapshots(
                self.species_names,
                'index',
                self.reference_snapshot_index,
                snapshot_directory=self.snapshot_directory,
                properties=properties_read,
                element_indices=[0],
                assign_hosts=False,
                check_properties=False,
            )

        # older simulations do not have id.child - use abundance of total metals instead
        if match_property == 'id.child' and 'id.child' not in part_z0[self.species_names[0]]:
            self.say(
                'input match_property = {} does not exist in snapshot {}'.format(
                    match_property, part_z0.snapshot['index']
                )
            )
            match_property = 'massfraction.metals'
            self.say(f'instead, using: {match_property}')
            if match_property not in properties_read:
                properties_read.append(match_property)
                part_z0 = self.GizmoRead.read_snapshots(
                    self.species_names,
                    'index',
                    self.reference_snapshot_index,
                    snapshot_directory=self.snapshot_directory,
                    properties=properties_read,
                    element_indices=[0],
                    assign_hosts=False,
                    check_properties=False,
                )

        for spec_name in self.species_names:
            assert spec_name in part_z0
            assert part_z0[spec_name].prop(match_property) is not None
            if test_property and spec_name == 'star':
                assert part_z0[spec_name].prop(test_property) is not None

        # get list of snapshot indices to assign
        if snapshot_indices is None or len(snapshot_indices) == 0:
            snapshot_indices = np.arange(
                min(part_z0.Snapshot['index']), max(part_z0.Snapshot['index']) + 1
            )
        snapshot_indices = np.setdiff1d(snapshot_indices, part_z0.snapshot['index'])  # skip current
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # diagnostic
        pindices_mult = ut.particle.get_indices_by_id_uniqueness(
            part_z0, self.species_names, self.id_name, 'multiple'
        )
        species_names_print = self.species_names[0]
        if len(self.species_names) > 1:
            for spec_name in self.species_names[1:]:
                species_names_print += f' + {spec_name}'
        self.say(
            '* {} {} particles have redundant id at (reference) snapshot {}'.format(
                pindices_mult.size, species_names_print, self.reference_snapshot_index
            )
        )

        # assign pointers at reference snapshot from particle id to index in catalog
        ut.particle.assign_id_to_index(
            part_z0, self.species_names, self.id_name, store_as_dict=True, verbose=False
        )
        self.say(f'assigned id->index pointers at snapshot {self.reference_snapshot_index}')

        self.match_property = match_property
        self.match_propery_tolerance = match_propery_tolerance
        self.test_property = test_property

        # counters for sanity checks
        count = {
            'id no match': 0,
            'match prop no match': 0,
            'match prop redundant': 0,
            'test prop offset': 0,
        }

        # initiate threads, if asking for > 1
        if proc_number > 1:
            from multiprocessing import Pool

            with Pool(proc_number) as pool:
                for snapshot_index in snapshot_indices:
                    # memory errors if try to pass part_z0, so instead re-read part_z0 per thread
                    pool.apply_async(
                        self._write_pointers_to_snapshot, (None, snapshot_index, count)
                    )
        else:
            for snapshot_index in snapshot_indices:
                self._write_pointers_to_snapshot(part_z0, snapshot_index, count)

        # print cumulative diagnostics
        print()
        if count['id no match']:
            self.say('! {} total not have id match!'.format(count['id no match']))
        if count['match prop no match']:
            self.say(
                '! {} total not have {} match!'.format(count['match prop no match'], match_property)
            )
        if count['match prop redundant']:
            self.say(
                '! {} total have redundant {}!'.format(
                    count['match prop redundant'], match_property
                )
            )
        if count['test prop offset']:
            self.say('! {} total have offset {}'.format(count['test prop offset'], test_property))

    def _write_pointers_to_snapshot(self, part_z0, snapshot_index, count_tot={}):
        '''
        Assign to each particle a pointer from its index at the reference (later, z0) snapshot
        to its index (and species name) at a (earlier, z) snapshot.
        Write the particle pointers to file.

        Parameters
        ----------
        part_z0 : dict : catalog of particles at the reference (later, z0) snapshot
        snapshot_index : int : snapshot index to assign pointers to at the (earlier, z) snapshot
        count_tot : dict : diagnostic counters
        '''
        properties_read = [self.id_name, self.match_property, self.test_property]

        # if not input, read particles at reference (z0) snaphsot
        if part_z0 is None:
            part_z0 = self.GizmoRead.read_snapshots(
                self.species_names,
                'index',
                self.reference_snapshot_index,
                snapshot_directory=self.snapshot_directory,
                properties=properties_read,
                element_indices=[0],
                assign_hosts=False,
                check_properties=False,
            )
            ut.particle.assign_id_to_index(
                part_z0, self.species_names, self.id_name, store_as_dict=True, verbose=False
            )

        # read particles at this snapshot
        # need to do this to get the exact scale-factor of snapshot
        part_z = self.GizmoRead.read_snapshots(
            self.species_names,
            'index',
            snapshot_index,
            snapshot_directory=self.snapshot_directory,
            properties=properties_read,
            element_indices=[0],
            assign_hosts=False,
            check_properties=False,
        )

        # diagnostic
        species_names_print = self.species_names[0]
        if len(self.species_names) > 1:
            for spec_name in self.species_names[1:]:
                species_names_print += f' + {spec_name}'

        spec_count = 0
        species_names_z = []  # species that are in catalog at this snapshot
        for spec_name in self.species_names:
            if spec_name in part_z and len(part_z[spec_name][self.id_name]) > 0:
                spec_count += 1
                species_names_z.append(spec_name)
            else:
                self.say(f'! no {spec_name} particles at snapshot {snapshot_index}')
        if not spec_count:
            return

        pindices_mult = ut.particle.get_indices_by_id_uniqueness(
            part_z, species_names_z, self.id_name, 'multiple'
        )

        self.say(
            '* {} {} particles have redundant id at snapshot {}'.format(
                pindices_mult.size, species_names_print, snapshot_index
            )
        )

        count = {
            'id no match': 0,
            'match prop redundant': 0,
            'match prop no match': 0,
            'test prop offset': 0,
        }

        # dictionary class to store pointers and meta-data
        Pointer = ParticlePointerDictionaryClass(part_z0, part_z, self.species_names)
        pointer_index_name = Pointer.pointer_index_name
        z = Pointer.z_name

        for spec_name in species_names_z:
            # get particle index offest (non-zero if concatenating multiple species)
            total_index_offset = Pointer[z + spec_name + '.index.limits'][0]

            id_no_match_number = 0
            match_prop_redundant_number = 0
            match_prop_no_match_number = 0
            for part_z_index, part_z_id in enumerate(part_z[spec_name][self.id_name]):
                try:
                    # can point to multiple particles at reference (later, z0) snapshot
                    part_z0_list = part_z0.id_to_index[part_z_id]
                except (IndexError, KeyError):
                    id_no_match_number += 1
                    continue

                # get index in concatenated list
                part_z_total_index = part_z_index + total_index_offset

                if np.ndim(part_z0_list) == 1:
                    # particle id is unique - easy case
                    part_z0_total_index = part_z0_list[2]
                    Pointer[pointer_index_name][part_z0_total_index] = part_z_total_index
                else:
                    # particle id is redundant - tricky case
                    # loop through particles with this id, use match_property to match
                    # sanity check
                    match_props = [
                        part_z0[z0_spec_name].prop(self.match_property, z0_index)
                        for z0_spec_name, z0_index, z0_total_index in part_z0_list
                    ]
                    if np.unique(match_props).size != len(part_z0_list):
                        match_prop_redundant_number += 1

                    z_match_prop = part_z[spec_name].prop(self.match_property, part_z_index)

                    for z0_spec_name, z0_index, z0_total_index in part_z0_list:
                        z0_match_prop = part_z0[z0_spec_name].prop(self.match_property, z0_index)

                        if self.match_property == 'id.child':
                            if z0_match_prop == z_match_prop:
                                Pointer[pointer_index_name][z0_total_index] = part_z_total_index
                                break
                        else:
                            frac_dif = np.abs((z0_match_prop - z_match_prop) / z_match_prop)
                            if frac_dif < self.match_propery_tolerance:
                                Pointer[pointer_index_name][z0_total_index] = part_z_total_index
                                break
                    else:
                        match_prop_no_match_number += 1

            if id_no_match_number:
                self.say(
                    '! {} {} particles not have id match at snapshot {}'.format(
                        id_no_match_number, spec_name, snapshot_index
                    )
                )
                count['id no match'] += id_no_match_number
            if match_prop_redundant_number:
                self.say(
                    '! {} {} particles have redundant {} at snapshot {}'.format(
                        match_prop_redundant_number, spec_name, self.match_property, snapshot_index
                    )
                )
                count['match prop redundant'] += match_prop_redundant_number
            if match_prop_no_match_number:
                self.say(
                    '! {} {} particles not have {} match at snapshot {}'.format(
                        match_prop_no_match_number, spec_name, self.match_property, snapshot_index
                    )
                )
                count['match prop no match'] += match_prop_no_match_number

        # sanity checks
        part_z0_total_indices = np.where(Pointer[pointer_index_name] >= 0)[0]
        # ensure same number of pointers from z0 to z as particles in snapshot at z
        if part_z0_total_indices.size != Pointer[z + 'particle.number']:
            self.say(
                '! {} {} particles at snapshot {},'.format(
                    Pointer[z + 'particle.number'], species_names_z, snapshot_index
                )
            )
            self.say(
                'but matched to only {} particles at snapshot {}'.format(
                    part_z0_total_indices.size, part_z0.snapshot['index']
                )
            )
        else:
            # check using test property - only valid for stars!
            if (
                self.test_property
                and self.test_property != self.match_property
                and 'star' in species_names_z
                and count['id no match'] == count['match prop no match'] == 0
            ):

                z_star_indices = Pointer.get_pointers('star', 'star', return_array=True)
                z0_star_indices = np.where(z_star_indices >= 0)[0]
                z_star_indices = z_star_indices[z0_star_indices]

                prop_difs = np.abs(
                    (
                        part_z['star'].prop(self.test_property, z_star_indices)
                        - part_z0['star'].prop(self.test_property, z0_star_indices)
                    )
                    / part_z['star'].prop(self.test_property, z_star_indices)
                )
                count['test prop offset'] = np.sum(prop_difs > self.match_propery_tolerance)

                if count['test prop offset']:
                    self.say(
                        '! {} matched particles have different {} at snapshot {} v {}'.format(
                            count['test prop offset'],
                            self.test_property,
                            snapshot_index,
                            part_z0.snapshot['index'],
                        )
                    )

        for k in count:
            count_tot[k] += count[k]

        # write file for this snapshot
        self.io_pointers(Pointer=Pointer)


ParticlePointer = ParticlePointerClass()


def test_particle_pointers(part, part_z1, part_z2):
    '''
    .
    '''
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
    Compute coordinates (3-D distances and 3-D velocities) wrt each primary host galaxy for each
    particle at the snapshot immediately after it forms.
    In computing the coordinates of each primary host at each previous snapshots, use only star
    particles within host_distance_limits of each primary host at the reference (latest) snapshot.
    '''

    def __init__(
        self,
        species_name='star',
        simulation_directory=gizmo_default.simulation_directory,
        track_directory=gizmo_default.track_directory,
        snapshot_directory=gizmo_default.snapshot_directory,
        reference_snapshot_index=gizmo_default.snapshot_index,
        host_distance_limits=[0, 50],
    ):
        '''
        Parameters
        ----------
        species : str : name of particle species to track
        simulation_directory : str : directory of simulation
        track_directory : str : directory of files for particle pointers and formation coordinates
        reference_snapshot_index : float :
            index of reference (later) snapshot to compute particle pointers from
        snapshot_directory : str : directory of snapshot files (within simulation directory)
        host_distance_limits : list :
            min and max distance [kpc physical] to select particles near each primary host at the
            reference snapshot; use only these particles to compute host coordinates at earlier
            snapshots
        '''
        self.id_name = ID_NAME
        self.species_name = species_name
        assert np.isscalar(self.species_name)
        self.simulation_directory = ut.io.get_path(simulation_directory)
        self.track_directory = ut.io.get_path(track_directory)
        self.snapshot_directory = ut.io.get_path(snapshot_directory)
        self.reference_snapshot_index = reference_snapshot_index
        self.host_distance_limits = host_distance_limits

        self.GizmoRead = gizmo_io.ReadClass()

        # set numpy data type to store coordinates
        self.coordinate_dtype = np.float32

        # names of distances and velocities to write/read
        self.form_host_coordiante_kinds = ['form.host.distance', 'form.host.velocity']

    def io_formation_coordinates(
        self, part, simulation_directory=None, track_directory=None, write=False
    ):
        '''
        Read or write, for each particle, at the first snapshot after it formed,
        its coordinates (3-D distances and 3-D velocities) wrt the host galaxy center,
        aligned with the principal axes of the host galaxy at that time.
        If reading, assign to input particle dictionary.

        Parameters
        ----------
        part : dict : catalog of particles at a snapshot
        simulation_directory : str : directory of simulation
        track_directory : str : directory of files for particle pointers and formation coordinates
        write : bool : whether to write to file (instead of read)
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
            dict_out = collections.OrderedDict()
            dict_out['species'] = self.species_name
            dict_out['snapshot.index'] = part.snapshot['index']
            dict_out['id'] = part[self.species_name][self.id_name]
            for prop_name in part[self.species_name]:
                if 'form.host' in prop_name:
                    dict_out[prop_name] = part[self.species_name][prop_name]
            for prop_name in ['position', 'velocity', 'rotation', 'axis.ratios']:
                dict_out['host.' + prop_name] = part[self.species_name].hostz[prop_name]

            ut.io.file_hdf5(path_file_name, dict_out)

        else:
            # read in
            # backwards compatibility with old file name
            try:
                dict_read = ut.io.file_hdf5(path_file_name)
            except OSError:
                path_file_name = path_file_name.replace(
                    gizmo_default.hosts_coordinates_file_name, 'star_form_coordinates_600.hdf5'
                )
                dict_read = ut.io.file_hdf5(path_file_name)

            # sanity check
            bad_id_number = np.sum(part[self.species_name][self.id_name] != dict_read['id'])
            if bad_id_number:
                self.say(f'! {bad_id_number} particles have mismatched id, this is bad!')

            # initialize dictionaries to store host properties across
            part.hostz = {
                'position': [],
                'velocity': [],
                'rotation': [],
                'axis.ratios': [],
            }
            for spec_name in part:
                part[spec_name].hostz = {
                    'position': [],
                    'velocity': [],
                    'rotation': [],
                    'axis.ratios': [],
                }

            for prop_name in dict_read:
                if prop_name == 'species':
                    self.say(f'reading formation coordinates of {dict_read[prop_name]} particles')
                    continue

                if prop_name == 'snapshot.index':
                    self.say(f'reading particles at snapshot {dict_read[prop_name]}')
                    continue

                if prop_name == 'id':
                    mismatch_ids = part[self.species_name][self.id_name] != dict_read[prop_name]
                    if np.sum(mismatch_ids) > 0:
                        self.say(
                            f'! {np.sum(mismatch_ids)} ids are mis-matched between'
                            + ' formation particles read in and input particle dictionary'
                        )
                    continue

                elif 'form.' in prop_name:
                    # store coordinates at formation
                    part[self.species_name][prop_name] = dict_read[prop_name]
                    continue

                elif 'host.position' in prop_name or 'center.position' in prop_name:
                    if np.ndim(dict_read[prop_name]) == 2:
                        dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                            (dict_read[prop_name].shape[0], 1, dict_read[prop_name].shape[1])
                        )  # update from old file format
                    prop_name_store = 'position'

                elif 'host.velocit' in prop_name or 'center.velocit' in prop_name:
                    if np.ndim(dict_read[prop_name]) == 2:
                        dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                            (dict_read[prop_name].shape[0], 1, dict_read[prop_name].shape[1])
                        )  # update from old file format
                    prop_name_store = 'velocity'

                elif 'host.rotation' in prop_name or prop_name == 'principal.axes.vectors':
                    # compatible with all naming conventions
                    if np.ndim(dict_read[prop_name]) == 3:
                        dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                            (
                                dict_read[prop_name].shape[0],
                                1,
                                dict_read[prop_name].shape[1],
                                dict_read[prop_name].shape[2],
                            )
                        )  # update from old file format
                    prop_name_store = 'rotation'

                elif 'host.axis.ratios' in prop_name:
                    prop_name_store = 'axis.ratios'

                else:
                    self.say(f'! not sure how to parse {prop_name}')
                    continue

                part.hostz[prop_name_store] = dict_read[prop_name]
                for spec_name in part:
                    part[spec_name].hostz[prop_name_store] = dict_read[prop_name]

    def read_hosts(self, part, simulation_directory=None, track_directory=None, verbose=True):
        '''
        Read the position, velocity, and principal axes ratios + rotation tensor of each host
        galaxy, computed via particle tracking and stored in the file for particle formation
        coordinates.
        Assign to input particle dictionary.

        Parameters
        ----------
        part : dict : catalog of particles at a snapshot
        simulation_directory : str : directory of simulation
        track_directory : str : directory of files for particle pointers and formation coordinates
        verbose : bool : whether to print diagnostic information

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

        # self.say(
        #    '* reading hosts position, velocity, rotation, axis ratios from:  {}'.format(
        #        path_file_name.lstrip('./')
        #    )
        # )

        # backwards compatibility with old file name
        try:
            dict_read = ut.io.file_hdf5(path_file_name, verbose=False)
        except OSError:
            path_file_name = path_file_name.replace(
                gizmo_default.hosts_coordinates_file_name, 'star_form_coordinates_600.hdf5'
            )
            dict_read = ut.io.file_hdf5(path_file_name, verbose=False)

        snapshot_index = part.snapshot['index']

        # initialize dictionaries to store host properties across
        part.hostz = {
            'position': [],
            'velocity': [],
            'rotation': [],
            'axis.ratios': [],
        }
        for spec_name in part:
            part[spec_name].hostz = {
                'position': [],
                'velocity': [],
                'rotation': [],
                'axis.ratios': [],
            }

        for prop_name in dict_read:
            if 'form.' in prop_name or prop_name in ['id', 'snapshot.index', 'species']:
                continue

            elif 'host.position' in prop_name or 'center.position' in prop_name:
                if np.ndim(dict_read[prop_name]) == 2:
                    dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                        (dict_read[prop_name].shape[0], 1, dict_read[prop_name].shape[1])
                    )  # update from old file format
                prop_name_store = 'position'

            elif 'host.velocit' in prop_name or 'center.velocit' in prop_name:
                if np.ndim(dict_read[prop_name]) == 2:
                    dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                        (dict_read[prop_name].shape[0], 1, dict_read[prop_name].shape[1])
                    )  # update from old file format
                prop_name_store = 'velocity'

            elif 'host.rotation' in prop_name or prop_name == 'principal.axes.vectors':
                if np.ndim(dict_read[prop_name]) == 3:
                    dict_read[prop_name] = np.array([dict_read[prop_name]]).reshape(
                        (
                            dict_read[prop_name].shape[0],
                            1,
                            dict_read[prop_name].shape[1],
                            dict_read[prop_name].shape[2],
                        )
                    )  # update from old file format
                prop_name_store = 'rotation'

            elif 'host.axis.ratios' in prop_name:
                prop_name_store = 'axis.ratios'

            else:
                self.say(f'! not sure how to parse {prop_name}')
                continue

            part.hostz[prop_name_store] = dict_read[prop_name]
            part.host[prop_name_store] = part.hostz[prop_name_store][snapshot_index]
            for spec_name in part:
                part[spec_name].hostz[prop_name_store] = part.hostz[prop_name_store]
                part[spec_name].host[prop_name_store] = part.host[prop_name_store]

        if 'hostz' in part.__dict__:
            host_number = part.hostz['position'].shape[1]
            self.say(
                f'read position, velocity, rotation, axis ratios for {host_number} host[s]'
                + f' in:  {path_file_name}'
            )

        if verbose:
            for host_i, host_position in enumerate(part.host['position']):
                self.say(f'host{host_i + 1} position = (', end='')
                ut.io.print_array(host_position, '{:.3f}', end='')
                print(') [kpc comoving]')

            for host_i, host_velocity in enumerate(part.host['velocity']):
                self.say(f'host{host_i + 1} velocity = (', end='')
                ut.io.print_array(host_velocity, '{:.1f}', end='')
                print(') [km / s]')

            # print()

    def write_formation_coordinates(
        self, part_z0=None, host_number=1, proc_number=1, simulation_directory=None
    ):
        '''
        Assign to each particle its coordiates (3-D distance and 3-D velocity) wrt each primary
        host galaxy at the snapshot after it formed.

        Parameters
        ----------
        part : dict : catalog of particles at the reference snapshot
        host_number : int : number of host galaxies to assign and compute coordinates relative to
        proc_number : int : number of parallel processes to run
        simulation_directory : str : directory of simulation
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
                    'position',
                    'velocity',
                    'mass',
                    'id.child',
                    'form.scalefactor',
                ],
                element_indices=[0],
                host_number=host_number,
                assign_hosts='mass',
                check_properties=False,
            )

        # get list of snapshots to assign
        snapshot_indices = np.arange(
            min(part_z0.Snapshot['index']), max(part_z0.Snapshot['index']) + 1
        )
        snapshot_indices = np.sort(snapshot_indices)[::-1]  # work backwards in time

        # initialize and store position, velocity, principal axes rotation tensor + axis ratios
        # of each primary host galaxy at each snapshot
        part_z0[self.species_name].hostz = {}

        part_z0[self.species_name].hostz['position'] = (
            np.zeros([part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype)
            + np.nan
        )
        part_z0[self.species_name].hostz['velocity'] = (
            np.zeros([part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype)
            + np.nan
        )

        part_z0[self.species_name].hostz['rotation'] = (
            np.zeros([part_z0.Snapshot['index'].size, host_number, 3, 3], self.coordinate_dtype)
            + np.nan
        )

        part_z0[self.species_name].hostz['axis.ratios'] = (
            np.zeros([part_z0.Snapshot['index'].size, host_number, 3], self.coordinate_dtype)
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
            for prop_name in self.form_host_coordiante_kinds:
                prop_name = prop_name.replace('host.', host_name)  # update host name (if necessary)
                part_z0[self.species_name][prop_name] = (
                    np.zeros(part_z0[self.species_name]['position'].shape, self.coordinate_dtype)
                    + np.nan
                )

        count = {'id none': 0, 'id wrong': 0}

        # initiate threads, if asking for > 1
        if proc_number > 1:
            from multiprocessing import Pool

            with Pool(proc_number) as pool:
                for snapshot_index in snapshot_indices:
                    pool.apply(
                        self._write_formation_coordinates,
                        (part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count),
                    )
        else:
            for snapshot_index in snapshot_indices:
                self._write_formation_coordinates(
                    part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count
                )

        # print cumulative diagnostics
        print()
        if count['id none']:
            self.say('! {} total do not have valid id!'.format(count['id none']))
        if count['id wrong']:
            self.say('! {} total not have id match!'.format(count['id wrong']))

    def _write_formation_coordinates(
        self, part_z0, hosts_part_z0_indicess, host_number, snapshot_index, count_tot
    ):
        '''
        Assign to each particle its coordinates (position and velocity) wrt the primary host.
        Write to file.

        TODO: add + store host axis ratios

        Parameters
        ----------
        part_z0 : dict : catalog of particles at the reference (latest) snapshot
        hosts_part_z0_indices : list of arrays :
            indices of particles near each primary host at the reference (latest) snapshot
        host_number : int : number of host galaxies to assign and compute coordinates relative to
        snapshot_index : int : snapshot index at which to assign particle pointers to
        count_tot : dict : diagnostic counters
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
                    self.species_name, self.species_name, return_array=True
                )
            except IOError:
                self.say(f'! can not read pointers to snapshot {snapshot_index}')
                return

        part_z0_indices = part_z0_indices[part_pointers >= 0]
        self.say(f'\n# {part_z0_indices.size} to assign at snapshot {snapshot_index}')

        count = {'id none': 0, 'id wrong': 0}

        if part_z0_indices.size > 0:
            part_z = self.GizmoRead.read_snapshots(
                self.species_name,
                'index',
                snapshot_index,
                snapshot_directory=self.snapshot_directory,
                properties=[self.id_name, 'position', 'velocity', 'mass', 'form.scalefactor'],
                assign_hosts=False,
                check_properties=True,
            )

            # only use the particles that are near each primary host at the reference snapshot
            # to compute the coordinates of host progenitors at earlier snapshots
            hosts_part_z_indicess = []
            for host_i in range(host_number):
                hosts_part_z_indices = part_pointers[hosts_part_z0_indicess[host_i]]
                hosts_part_z_indicess.append(hosts_part_z_indices[hosts_part_z_indices >= 0])

            self.GizmoRead.assign_hosts_coordinates(
                part_z,
                self.species_name,
                hosts_part_z_indicess,
                'mass',
                host_number,
                exclusion_distance=None,
            )

            part_z_indices = part_pointers[part_z0_indices]

            # sanity checks
            masks = part_z_indices >= 0
            count['id none'] = part_z_indices.size - np.sum(masks)
            if count['id none']:
                self.say(
                    '! {} have no id match at snapshot {}!'.format(count['id none'], snapshot_index)
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
                    '! {} have wrong id match at snapshot {}!'.format(
                        count['id wrong'], snapshot_index
                    )
                )
                part_z_indices = part_z_indices[masks]
                part_z0_indices = part_z0_indices[masks]

            # compute rotation vectors for principal axes from young stars within R_90
            self.GizmoRead.assign_hosts_rotation(part_z)

            # store host galaxy properties
            for prop_name in ['position', 'velocity', 'rotation', 'axis.ratios']:
                part_z0[self.species_name].hostz[prop_name][snapshot_index] = part_z.host[prop_name]

            for host_i in range(host_number):
                # compute coordinates wrt primary host
                host_name = ut.catalog.get_host_name(host_i)

                for prop_name in self.form_host_coordiante_kinds:
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
            self.io_formation_coordinates(part_z0, write=True)


ParticleCoordinate = ParticleCoordinateClass()


# --------------------------------------------------------------------------------------------------
# run from command line
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function: pointer, coordinate, pointer+coordinate')

    function_kind = str(sys.argv[1])

    assert 'pointer' in function_kind or 'coordinate' in function_kind

    if 'pointer' in function_kind:
        ParticlePointer.write_pointers_to_snapshots()

    if 'coordinate' in function_kind:
        ParticleCoordinate.write_formation_coordinates()
