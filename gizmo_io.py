#!/usr/bin/env python3

'''
Read Gizmo snapshots, intended for use with FIRE-2 simulations.

@author: Andrew Wetzel <arwetzel@gmail.com>


Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
    elemental abundance [mass fraction]


Reading a snapshot

Within a simulation directory, read all particles in a snapshot at redshift 0 via:
    part = gizmo.io.Read.read_snapshots('all', 'redshift', 0)
part is a dictionary, with a key for each particle species. So, access star particle dictionary via:
    part['star']
part['star'] is dictionary, with each property of particles as a key. For example:
    part['star']['mass']
returns a numpy array of masses, one for each star particle, while
    part['star']['position']
returns a numpy array of positions, of dimension particle_number x 3.

If you want the code to compute and store the principal axes (via the moment of inertia tensor),
computed using the stellar distribution (disk) of the host galaxy[s]:
    part = gizmo.io.Read.read_snapshots('all', 'redshift', 0, assign_host_principal_axes=True)


Particle species

The available particle species in a cosmological simulation are:
    part['dark'] : dark matter at the highest resolution
    part['dark2'] : dark matter at lower resolution (outside of the zoom-in region)
    part['gas'] : gas
    part['star'] : stars
    part['blackhole'] : black holes (if the simulation contains them)


Default/stored particle properties

Access these via:
    part[species_name][property_name]
For example:
    part['star']['position']

All particle species have the following properties:
    'id' : ID (indexing starts at 0)
    'position' : 3-D position, along simulations's (arbitrary) x,y,z grid [kpc comoving]
    'velocity' : 3-D velocity, along simulations's (arbitrary) x,y,z grid [km / s peculiar]
    'mass' : mass [M_sun]
    'potential' : potential (computed via all particles in the box) [km^2 / s^2 physical]

Star and gas particles also have additional IDs (because gas can split):
    'id.child' : child ID
    'id.generation' : generation ID
These are initialized to 0 for all gas particles.
Each time a gas particle splits into 2, the 'self' particle retains id.child, while the other
particle gets id.child += 2 ^ id.generation.
Both particles then get id.generation += 1.
Star particles inherit these from their progenitor gas particles.
Caveat: this allows a maximum of 30 generations, then its resets to 0.
Thus, particles with id.generation > 30 are not unique anymore.

Star and gas particles also have:
    'massfraction' : fraction of the mass that is in different elemental abundances,
        stored as an array for each particle, with indexes as follows:
        0 = all metals (everything not H, He)
        1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe

Star particles also have:
  'form.scalefactor' : expansion scale-factor when the star particle formed [0 to 1]

Gas particles also have:
    'temperature' : [K]
    'density' : [M_sun / kpc^3]
    'smooth.length' : smoothing/kernel length, stored as Plummer-equivalent
        (for consistency with force softening) [kpc physical]
    'electron.fraction' : free-electron number per proton, averaged over mass of gas particle
    'hydrogen.neutral.fraction' : fraction of hydrogen that is neutral (not ionized)
    'sfr' : instantaneous star formation rate [M_sun / yr]


Derived properties

part is a ParticleDictionaryClass that can compute derived properties on the fly.
Call derived (or stored) properties via:
    part[species_name].prop(property_name)
For example:
    part['star'].prop('metallicity.fe')
You also can call stored properties via part[species_name].prop(property_name).
It will know that it is a stored property and return as is.
For example, part['star'].prop('position') is the same as part['star']['position'].

See ParticleDictionaryClass.prop() for full options for parsing of derived properties.
Some useful examples:

    part[species_name].prop('host.distance') :
        3-D distance from primary galaxy center along simulation's (arbitrary) x,y,z [kpc physical]
    part[species_name].prop('host.distance.total') : total (scalar) distance [kpc physical]
    part[species_name].prop('host.distance.principal') :
        3-D distance aligned with the galaxy principal (major, intermed, minor) axes [kpc physial]
    part[species_name].prop('host.distance.principal.cylindrical') :
        same, but in cylindrical coordinates [kpc physical]:
            along the major axes (R, positive definite)
            vertical height wrt the disk (Z, signed)
            azimuthal angle (phi, 0 to 2 * pi)

    part[species_name].prop('host.velocity') :
        3-D velocity wrt primary galaxy center along simulation's (arbitrary) x,y,z axes [km / s]
    part[species_name].prop('host.velocity.total') : total (scalar) velocity [km / s]
    part[species_name].prop('host.velocity.principal') :
        3-D velocity aligned with the galaxy principal (major, intermed, minor) axes [km / s]
    part[species_name].prop('host.distance.principal.cylindrical') :
        same, but in cylindrical coordinates [km / s]:
            along the major axes (v_R, signed)
            along the vertical wrt the disk (v_Z, signed)
            along the azimuth (phi, signed)

    part['star'].prop('form.time') : time of the Universe when star particle formed [Gyr]
    part['star'].prop('age') :
        age of star particle at current snapshot (current_time - formation_time) [Gyr]

    part['star'].prop('form.mass') : mass of star particle when it formed [M_sun]
    part['star'].prop('mass.loss') : mass loss since formation of star particle [M_sun]

    part['gas'].prop('number.density') :
        gas number density, assuming solar metallicity [hydrogen atoms / cm^3]

    part['gas' or 'star'].prop('metallicity.iron') :
        iron abundance [Fe/H] :=
            log10((mass_iron / mass_hydrogen)_particle / (mass_iron / mass_hydrogen)_sun)
        as scaled to Solar (Asplund et al 2009)
        this works for all abundances: 'metallicity.carbon', 'metallicity.magnesium', etc
    part['gas' or 'star'].prop('metallicity.magnesium - metallicity.iron') : [Mg/Fe]
        also can compute arithmetic combinations

    part['gas' or 'star'].prop('mass.hydrogen') : total hydrogen mass in particle [M_sun]
    part['gas' or 'star'].prop('mass.oxygen') : total oxygen mass in particle [M_sun]
    etc
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatibility
import os
import collections
import h5py
import numpy as np
# local ----
import utilities as ut
from . import gizmo_star


#===================================================================================================
# particle dictionary class
#===================================================================================================
class ParticleDictionaryClass(dict):
    '''
    Dictionary class to store particle data.
    This functions like a normal dictionary in terms of storing default properties of particles,
    but it also allows greater flexibility, storing additional meta-data (such as snapshot
    information and cosmological parameters) and calling derived quantities via .prop().
    '''

    def __init__(self):
        # use to translate between element name and index in element table
        self.element_dict = collections.OrderedDict()
        self.element_dict['metals'] = self.element_dict['total'] = 0
        self.element_dict['helium'] = self.element_dict['he'] = 1
        self.element_dict['carbon'] = self.element_dict['c'] = 2
        self.element_dict['nitrogen'] = self.element_dict['n'] = 3
        self.element_dict['oxygen'] = self.element_dict['o'] = 4
        self.element_dict['neon'] = self.element_dict['ne'] = 5
        self.element_dict['magnesium'] = self.element_dict['mg'] = 6
        self.element_dict['silicon'] = self.element_dict['si'] = 7
        self.element_dict['sulphur'] = self.element_dict['s'] = 8
        self.element_dict['calcium'] = self.element_dict['ca'] = 9
        self.element_dict['iron'] = self.element_dict['fe'] = 10

        # to use if read only subset of elements
        self.element_pointer = np.arange(len(self.element_dict) // 2)

        self.MassLoss = None

    def prop(self, property_name='', indices=None, dict_only=False):
        '''
        Get property, either from self dictionary or derive.
        Can compute basic mathematical manipulations, for example:
            'log temperature', 'temperature / density', 'abs position'

        Parameters
        ----------
        property_name : string : name of property
        indices : array : indices of particles to select
        dict_only : bool : require property_name to be in self's dict - avoids endless recursion

        Returns
        -------
        values : float or array : depending on dimensionality of input indices
        '''
        ## parsing general to all catalogs ----------
        property_name = property_name.strip()  # strip white space

        # if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]
        elif dict_only:
            raise KeyError('property = {} is not in self\'s dictionary'.format(property_name))

        # math relation, combining more than one property
        if ('/' in property_name or '*' in property_name or '+' in property_name or
                '-' in property_name):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise KeyError('not sure how to parse property = {}'.format(property_name))

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
                        prop_values[masks] = (
                            prop_values[masks] / self.prop(prop_name, indices)[masks])
                        masks = self.prop(prop_name, indices) == 0
                        prop_values[masks] = np.nan
                if '*' in property_name:
                    prop_values = prop_values * self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values = prop_values + self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values = prop_values - self.prop(prop_name, indices)

            if prop_values.size == 1:
                prop_values = np.float(prop_values)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        ## parsing specific to this catalog ----------
        # stellar mass loss
        if ('mass' in property_name and 'form' in property_name) or 'mass.loss' in property_name:
            if self.MassLoss is None:
                self.MassLoss = gizmo_star.MassLossClass()

            # fractional mass loss since formation
            values = self.MassLoss.get_mass_loss_fraction_from_spline(
                self.prop('age', indices) * 1000,
                metal_mass_fractions=self.prop('massfraction.metals', indices))

            if 'mass.loss' in property_name:
                if 'fraction' in property_name:
                    pass
                else:
                    values *= self.prop('mass', indices, dict_only=True) / (1 - values)  # mass loss
            elif 'mass' in property_name and 'form' in property_name:
                values = self.prop('mass', indices, dict_only=True) / (1 - values)  # formation mass

            return values

        # mass of element
        if 'mass.' in property_name:
            # mass from individual element
            values = (self.prop('mass', indices, dict_only=True) *
                      self.prop(property_name.replace('mass.', 'massfraction.'), indices))

            if property_name == 'mass.hydrogen.neutral':
                # mass from neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices, dict_only=True)

            return values

        # elemental abundance
        if 'massfraction.' in property_name or 'metallicity.' in property_name:
            # special cases
            if 'massfraction.hydrogen' in property_name or property_name == 'massfraction.h':
                # special case: mass fraction of hydrogen (excluding helium and metals)
                values = (1 - self.prop('massfraction.total', indices) -
                          self.prop('massfraction.helium', indices))

                if (property_name == 'massfraction.hydrogen.neutral' or
                        property_name == 'massfraction.h.neutral'):
                    # mass fraction of neutral hydrogen (excluding helium, metals, and ionized)
                    values = values * self.prop('hydrogen.neutral.fraction', indices)

                return values

            elif 'alpha' in property_name:
                return np.mean(
                    [self.prop('metallicity.o', indices),
                     self.prop('metallicity.mg', indices),
                     self.prop('metallicity.si', indices),
                     self.prop('metallicity.ca', indices),
                     ], 0)

            # normal cases
            element_index = None
            for prop_name in property_name.split('.'):
                if prop_name in self.element_dict:
                    element_index = self.element_pointer[self.element_dict[prop_name]]
                    element_name = prop_name
                    break

            if element_index is None:
                raise KeyError('not sure how to parse property = {}'.format(property_name))

            if indices is None:
                values = self['massfraction'][:, element_index]
            else:
                values = self['massfraction'][indices, element_index]

            if 'metallicity.' in property_name:
                values = ut.math.get_log(
                    values / ut.constant.sun_composition[element_name]['massfraction'])

            return values

        if 'number.density' in property_name:
            values = (self.prop('density', indices, dict_only=True) * ut.constant.proton_per_sun *
                      ut.constant.kpc_per_cm ** 3)

            if '.hydrogen' in property_name:
                # number density of hydrogen, using actual hydrogen mass of each particle [cm ^ -3]
                values = values * self.prop('massfraction.hydrogen', indices)
            else:
                # number density of 'hydrogen', assuming solar metallicity for particles [cm ^ -3]
                values = values * ut.constant.sun_hydrogen_mass_fraction

            return values

        if 'kernel.length' in property_name:
            # gaussian standard-deviation length (for cubic kernel) = inter-particle spacing [pc]
            return 1000 * (self.prop('mass', indices, dict_only=True) /
                           self.prop('density', indices, dict_only=True)) ** (1 / 3)

        # formation time or coordinates
        if (('form.' in property_name or property_name == 'age') and
                'host' not in property_name and 'distance' not in property_name and
                'velocity' not in property_name):
            if property_name == 'age' or ('time' in property_name and 'lookback' in property_name):
                # look-back time (stellar age) to formation
                values = self.snapshot['time'] - self.prop('form.time', indices)
            elif 'time' in property_name:
                # time (age of universe) of formation
                values = self.Cosmology.get_time(
                    self.prop('form.scalefactor', indices, dict_only=True), 'scalefactor')
            elif 'redshift' in property_name:
                # redshift of formation
                values = 1 / self.prop('form.scalefactor', indices, dict_only=True) - 1
            elif 'snapshot' in property_name:
                # snapshot index immediately after formation
                # increase formation scale-factor slightly for safety, because scale-factors of
                # written snapshots do not exactly coincide with input scale-factors
                padding_factor = (1 + 1e-7)
                values = self.Snapshot.get_snapshot_indices(
                    'scalefactor',
                    np.clip(self.prop('form.scalefactor', indices, dict_only=True) *
                            padding_factor, 0, 1), round_kind='up')

            return values

        # distance or velocity wrt the host galaxy/halo
        if ('host' in property_name and
                ('distance' in property_name or 'velocity' in property_name or
                 'acceleration' in property_name)):
            if 'host.near.' in property_name:
                host_name = 'host.near.'
                host_index = 0
            elif 'host.' in property_name or 'host1.' in property_name:
                host_name = 'host.'
                host_index = 0
            elif 'host2.' in property_name:
                host_name = 'host2.'
                host_index = 1
            elif 'host3.' in property_name:
                host_name = 'host3.'
                host_index = 2
            else:
                raise ValueError('could not identify host name in {}'.format(property_name))

            if 'form.' in property_name:
                # special case: coordinates wrt primary host *at formation*
                if 'distance' in property_name:
                    # 3-D distance vector wrt primary host at formation
                    values = self.prop('form.' + host_name + 'distance', indices, dict_only=True)
                elif 'velocity' in property_name:
                    # 3-D velocity vectory wrt host at formation
                    values = self.prop('form.' + host_name + 'velocity', indices, dict_only=True)
            else:
                # general case: coordinates wrt primary host at current snapshot
                if 'distance' in property_name:
                    # 3-D distance vector wrt primary host at current snapshot
                    values = ut.coordinate.get_distances(
                        self.prop('position', indices, dict_only=True),
                        self.host_positions[host_index],
                        self.info['box.length'], self.snapshot['scalefactor'])  # [kpc physical]
                elif 'velocity' in property_name:
                    # 3-D velocity, includes the Hubble flow
                    values = ut.coordinate.get_velocity_differences(
                        self.prop('velocity', indices, dict_only=True),
                        self.host_velocities[host_index],
                        self.prop('position', indices, dict_only=True),
                        self.host_positions[host_index],
                        self.info['box.length'], self.snapshot['scalefactor'],
                        self.snapshot['time.hubble'])
                elif 'acceleration' in property_name:
                    # 3-D acceleration
                    values = self.prop('acceleration', indices, dict_only=True)

                if 'principal' in property_name:
                    # align with host principal axes
                    values = ut.coordinate.get_coordinates_rotated(
                        values, self.host_rotation_tensors[host_index])

            if 'cylindrical' in property_name or 'spherical' in property_name:
                # convert to cylindrical or spherical coordinates
                if 'cylindrical' in property_name:
                    coordinate_system = 'cylindrical'
                elif 'spherical' in property_name:
                    coordinate_system = 'spherical'

                if 'distance' in property_name:
                    values = ut.coordinate.get_positions_in_coordinate_system(
                        values, 'cartesian', coordinate_system)
                elif 'velocity' in property_name or 'acceleration' in property_name:
                    if 'form.' in property_name:
                        # special case: coordinates wrt primary host *at formation*
                        distance_vectors = self.prop(
                            'form.' + host_name + 'distance', indices, dict_only=True)
                    elif 'principal' in property_name:
                        distance_vectors = self.prop(host_name + 'distance.principal', indices)
                    else:
                        distance_vectors = self.prop(
                            host_name + 'distance', indices, dict_only=True)
                    values = ut.coordinate.get_velocities_in_coordinate_system(
                        values, distance_vectors, 'cartesian', coordinate_system)

            # compute total (scalar) quantity
            if '.total' in property_name:

                if len(values.shape) == 1:
                    shape_pos = 0
                else:
                    shape_pos = 1
                values = np.sqrt(np.sum(values ** 2, shape_pos))

            return values

        if '.total' in property_name:
            # compute total (scalar) quantity (for velocity, acceleration)
            prop_name = property_name.replace('.total', '')
            try:
                values = self.prop(prop_name, indices)
                values = np.sqrt(np.sum(values ** 2, 1))
                return values
            except ValueError:
                pass

        # should not get this far without a return
        raise KeyError('not sure how to parse property = {}'.format(property_name))


#===================================================================================================
# read
#===================================================================================================
class ReadClass(ut.io.SayClass):
    '''
    Read Gizmo snapshot[s].
    '''

    def __init__(self):
        '''
        Set properties for snapshot files.
        '''
        # this format avoids accidentally reading text file that contains snapshot indices
        self.snapshot_name_base = 'snap*[!txt]'
        self.file_extension = '.hdf5'

        self.gas_eos = 5 / 3  # assumed equation of state of gas

        # create ordered dictionary to convert particle species name to its id,
        # set all possible species, and set the order in which to read species
        self.species_dict = collections.OrderedDict()
        # dark-matter species
        self.species_dict['dark'] = 1  # dark matter at highest resolution
        self.species_dict['dark2'] = 2  # dark matter at all lower resolutions
        # baryon species
        self.species_dict['gas'] = 0
        self.species_dict['star'] = 4
        self.species_dict['blackhole'] = 5

        self.species_all = tuple(self.species_dict.keys())
        self.species_read = list(self.species_all)

    def read_snapshots(
        self, species='all',
        snapshot_value_kind='index', snapshot_values=600,
        simulation_directory='.', snapshot_directory='output/', simulation_name='',
        properties='all', element_indices=None, particle_subsample_factor=None,
        separate_dark_lowres=False, sort_dark_by_id=False, convert_float32=False,
        host_number=1, assign_host_coordinates=True,
        assign_host_principal_axes=False, assign_host_orbits=False,
        assign_formation_coordinates=False, assign_index_pointers=False,
        check_properties=True):
        '''
        Read given properties for given particle species from simulation snapshot file[s].
        Can read single snapshot or multiple snapshots.
        If single snapshot, return as dictionary class;
        if multiple snapshots, return as list of dictionary classes.

        Parameters
        ----------
        species : string or list : name[s] of particle species:
            'all' = all species in file
            'dark' = dark matter at highest resolution
            'dark2' = dark matter at lower resolution
            'gas' = gas
            'star' = stars
            'blackhole' = black holes, if run contains them
        snapshot_value_kind : string :
            input snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof :
            index[s] or redshift[s] or scale-factor[s] of snapshot[s]
        simulation_directory : string : directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_dict
        element_indices : int or list : indices of elemental abundances to keep
            note: 0 = total metals, 1 = helium, 10 = iron, None or 'all' = read all elements
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        separate_dark_lowres : boolean :
            whether to separate low-resolution dark matter into separate dicts according to mass
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        convert_float32 : boolean : whether to convert all floats to 32 bit to save memory
        host_number : int : number of hosts to assign and compute coordinates relative to
        assign_host_coordinates : boolean :
            whether to assign position[s] and velocity[s] of host galaxy/halo[s]
        assign_host_principal_axes : boolean :
            whether to assign principal axes rotation tensor[s] of host galaxy[s]/halo[s]
        assign_host_orbits : booelan :
            whether to assign orbital properties wrt host galaxy[s]/halo[s]
        assign_formation_coordinates : boolean :
            whether to assign coordindates wrt the host galaxy at formation to stars
        assign_index_pointers : boolean :
            whether to assign index pointers from particles at z = 0 to particles in this snapshot
        check_properties : boolean : whether to check sanity of particle properties after read in

        Returns
        -------
        parts : dictionary or list thereof :
            if single snapshot, return as dictionary, else if multiple snapshots, return as list
        '''
        # parse input species to read
        if species == 'all' or species == ['all'] or not species:
            # read all species in snapshot
            species = self.species_all
        else:
            # read subsample of species in snapshot
            if np.isscalar(species):
                species = [species]  # ensure is list
            # check if input species names are valid
            for spec_name in list(species):
                if spec_name not in self.species_dict:
                    species.remove(spec_name)
                    self.say('! not recognize input species = {}'.format(spec_name))
        self.species_read = list(species)

        # read information about snapshot times
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        # if 'elvis' is in simulation directory name, force 2 hosts
        host_number = ut.catalog.get_host_number_from_directory(
            host_number, simulation_directory, os)

        Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
        snapshot_values = ut.array.arrayize(snapshot_values)

        parts = []  # list to store particle dictionaries

        # read all input snapshots
        for snapshot_value in snapshot_values:
            snapshot_index = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_value)

            # read header from snapshot file
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory, simulation_name)

            # read particles from snapshot file[s]
            part = self.read_particles(
                'index', snapshot_index, simulation_directory, snapshot_directory, properties,
                element_indices, convert_float32, header)

            # read/get (additional) cosmological parameters
            if header['is.cosmological']:
                part.Cosmology = self.get_cosmology(
                    simulation_directory, header['omega_lambda'], header['omega_matter'],
                    hubble=header['hubble'])
                for spec_name in part:
                    part[spec_name].Cosmology = part.Cosmology

            # adjust properties for each species
            self.adjust_particle_properties(
                part, header, particle_subsample_factor, separate_dark_lowres, sort_dark_by_id)

            # check sanity of particle properties read in
            if check_properties:
                self.check_properties(part)

            # assign auxilliary information to particle dictionary class
            # store header dictionary
            part.info = header
            for spec_name in part:
                part[spec_name].info = part.info

            # store information about snapshot time
            time = part.Cosmology.get_time(header['redshift'], 'redshift')
            part.snapshot = {
                'index': snapshot_index,
                'redshift': header['redshift'],
                'scalefactor': header['scalefactor'],
                'time': time,
                'time.lookback': part.Cosmology.get_time(0) - time,
                'time.hubble': (ut.constant.Gyr_per_sec /
                                part.Cosmology.get_hubble_parameter(header['redshift'])),
            }
            for spec_name in part:
                part[spec_name].snapshot = part.snapshot

            # store information on all snapshot times - may or may not be initialized
            part.Snapshot = Snapshot
            for spec_name in part:
                part[spec_name].Snapshot = part.Snapshot

            # initialize arrays to store position[s] and velocity[s] of host galaxy[s]/halo[s]
            # store both single 'default' host and array of hosts (for LG-like pairs)
            part.host_positions = []
            part.host_velocities = []
            for spec_name in part:
                part[spec_name].host_positions = []
                part[spec_name].host_velocities = []

            if assign_host_coordinates:
                self.assign_host_coordinates(part, host_number=host_number)

            # initialize arrays to store rotation tensor[s] that define principal axes of host[s]
            part.host_rotation_tensors = []
            for spec_name in part:
                part[spec_name].host_rotation_tensors = []
            if assign_host_coordinates and assign_host_principal_axes:
                self.assign_host_principal_axes(part)

            # store orbital properties wrt host galaxy[s]/halo[s]
            if assign_host_orbits and ('velocity' in properties or properties is 'all'):
                self.assign_host_orbits(part, 'star', part.host_positions, part.host_velocities)

            if 'star' in species and (assign_formation_coordinates or assign_index_pointers):
                from . import gizmo_track
                if assign_formation_coordinates:
                    # assign coordinates wrt host galaxy at formation
                    ParticleCoordinate = gizmo_track.ParticleCoordinateClass(
                        'star', simulation_directory + gizmo_track.TRACK_DIRECTORY)
                    ParticleCoordinate.io_formation_coordinates(part)

                elif assign_index_pointers:
                    # assign particle index pointers from z = 0 to this snapshot
                    ParticleIndex = gizmo_track.ParticleIndexPointerClass(
                        'star', simulation_directory + gizmo_track.TRACK_DIRECTORY)
                    ParticleIndex.io_pointers(part)

            # if read only 1 snapshot, return as particle dictionary instead of list
            if len(snapshot_values) == 1:
                parts = part
            else:
                parts.append(part)
                print()

        return parts

    def read_snapshots_simulations(
        self, species='all', snapshot_value_kind='index', snapshot_value=600,
        simulation_directories=[], snapshot_directory='output/',
        properties='all', element_indices=[0, 1, 6, 10], assign_host_principal_axes=False):
        '''
        Read snapshots at the same redshift from different simulations.
        Return as list of dictionaries.

        Parameters
        ----------
        species : string or list : name[s] of particle species to read
        snapshot_value_kind : string :
            input snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_value : int or float : index or redshift or scale-factor of snapshot
        simulation_directories : list or list of lists :
            list of simulation directories, or list of pairs of directory + simulation name
        snapshot_directory: string : directory of snapshot files within simulation_directory
        properties : string or list : name[s] of properties to read
        element_indices : int or list : indices of elements to read
        assign_host_principal_axes : boolean :
            whether to assign principal axes rotation tensor[s] of host galaxy[s]/halo[s]

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
        bad_snapshot_value = 0
        for simulation_directory, simulation_name in simulation_directories:
            try:
                _header = self.read_header(
                    snapshot_value_kind, snapshot_value, simulation_directory, snapshot_directory,
                    simulation_name)
            except Exception:
                self.say('! could not read snapshot header at {} = {:.3f} in {}'.format(
                         snapshot_value_kind, snapshot_value, simulation_directory))
                bad_snapshot_value += 1

        if bad_snapshot_value:
            self.say('\n! could not read {} snapshots'.format(bad_snapshot_value))
            return

        parts = []
        directories_read = []
        for directory, simulation_name in simulation_directories:
            try:
                part = self.read_snapshots(
                    species, snapshot_value_kind, snapshot_value, directory,
                    snapshot_directory, simulation_name, properties, element_indices,
                    assign_host_principal_axes=assign_host_principal_axes)

                if 'velocity' in properties:
                    self.assign_host_orbits(part, 'gas')

                parts.append(part)
                directories_read.append(directory)

            except Exception:
                self.say('! could not read snapshot at {} = {} in {}'.format(
                         snapshot_value_kind, snapshot_value, directory))

        if not len(parts):
            self.say('! could not read any snapshots at {} = {}'.format(
                     snapshot_value_kind, snapshot_value))
            return

        if 'mass' in properties and 'star' in part:
            for part, directory in zip(parts, directories_read):
                print('{}: star.mass = {:.3e}'.format(directory, part['star']['mass'].sum()))

        return parts

    def read_header(
        self, snapshot_value_kind='index', snapshot_value=600, simulation_directory='.',
        snapshot_directory='output/', simulation_name='', verbose=True):
        '''
        Read header from snapshot file.

        Parameters
        ----------
        snapshot_value_kind : string : input snapshot number kind: 'index', 'redshift'
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        simulation_name : string : name to store for future identification
        verbose : boolean : whether to print number of particles in snapshot

        Returns
        -------
        header : dictionary class : header dictionary
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
            # number of file blocks per snapshot
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
            # level of compression of snapshot file
            'CompactLevel': 'compression.level',
            'Compactify_Version': 'compression.version',
            'ReadMe': 'compression.readme',
        }

        header = {}  # dictionary to store header information

        # parse input values
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        if snapshot_value_kind != 'index':
            Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_value)
        else:
            snapshot_index = snapshot_value

        path_file_name = self.get_snapshot_file_names_indices(snapshot_directory, snapshot_index)

        self._is_first_print = True
        self.say('* reading header from:  {}'.format(path_file_name.strip('./')), end='\n')

        # open snapshot file
        with h5py.File(path_file_name, 'r') as file_in:
            header_in = file_in['Header'].attrs  # load header dictionary

            for prop_in in header_in:
                prop = header_dict[prop_in]
                header[prop] = header_in[prop_in]  # transfer to custom header dict

        # determine whether simulation is cosmological
        if (0 < header['hubble'] < 1 and 0 < header['omega_matter'] <= 1 and
                0 < header['omega_lambda'] <= 1):
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

        if verbose:
            self.say('snapshot contains the following number of particles:')
            # keep only species that have any particles
            read_particle_number = 0
            for spec_name in ut.array.get_list_combined(self.species_all, self.species_read):
                spec_id = self.species_dict[spec_name]
                self.say('  {:9s} (id = {}): {} particles'.format(
                         spec_name, spec_id, header['particle.numbers.total'][spec_id]))

                if header['particle.numbers.total'][spec_id] > 0:
                    read_particle_number += header['particle.numbers.total'][spec_id]
                elif spec_name in self.species_read:
                    self.species_read.remove(spec_name)

        if read_particle_number <= 0:
            raise OSError(
                'snapshot file[s] contain no particles of species = {}'.format(self.species_read))

        # check if simulation contains baryons
        header['has.baryons'] = False
        for spec_name in self.species_all:
            if 'dark' not in spec_name:
                if header['particle.numbers.total'][self.species_dict[spec_name]] > 0:
                    header['has.baryons'] = True
                    break

        # assign simulation name
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.split('/')[-2]
            simulation_name = simulation_name.replace('_', ' ')
            simulation_name = simulation_name.replace('res', 'r')
        header['simulation.name'] = simulation_name

        header['catalog.kind'] = 'particle'

        print()

        return header

    def read_particles(
        self, snapshot_value_kind='index', snapshot_value=600, simulation_directory='.',
        snapshot_directory='output/', properties='all', element_indices=None,
        convert_float32=False, header=None):
        '''
        Read particles from snapshot file[s].

        Parameters
        ----------
        snapshot_value_kind : string : input snapshot number kind: 'index', 'redshift'
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory: string : directory of snapshot files within simulation_directory
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, choose subset from among property_dict
        element_indices : int or list : indices of elements to keep
            note: 0 = total metals, 1 = helium, 10 = iron, None or 'all' = read all elements
        convert_float32 : boolean : whether to convert all floats to 32 bit to save memory

        Returns
        -------
        part : dictionary class : catalog of particles
        '''
        # convert name in snapshot's particle dictionary to custon name preference
        # if comment out any prop, will not read it
        property_dict = {
            ## all particles ----------
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',
            'Acceleration': 'acceleration',  # from grav for DM and stars, from grav + hydro for gas
            ## particles with adaptive smoothing
            #'AGS-Softening': 'smooth.length',  # for gas, this is same as SmoothingLength

            ## gas particles ----------
            'InternalEnergy': 'temperature',
            'Density': 'density',
            # stored in snapshot file as maximum distance to neighbor (radius of compact support)
            # but here convert to Plummer-equivalent length (for consistency with force softening)
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
            'BH_Mdot': 'accretion.rate',
            'BH_Mass_AlphaDisk': 'disk.mass',
            'BH_AccretionLength': 'accretion.length',
            'BH_NProgs': 'prog.number',
        }

        part = ut.array.DictClass()  # dictionary class to store properties for particle species

        # parse input list of properties to read
        if 'all' in properties or not properties:
            properties = list(property_dict.keys())
        else:
            if np.isscalar(properties):
                properties = [properties]  # ensure is list
            # make safe list of properties to read
            properties_temp = []
            for prop in list(properties):
                prop = str.lower(prop)
                if 'massfraction' in prop or 'metallicity' in prop:
                    prop = 'massfraction'  # this has several aliases, so ensure default name
                for prop_in in property_dict:
                    if prop in [str.lower(prop_in), str.lower(property_dict[prop_in])]:
                        properties_temp.append(prop_in)
            properties = properties_temp
            del(properties_temp)

        if 'InternalEnergy' in properties:
            # need helium mass fraction and electron fraction to compute temperature
            for prop in np.setdiff1d(['ElectronAbundance', 'Metallicity'], properties):
                properties.append(prop)

        # parse other input values
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        if snapshot_value_kind != 'index':
            Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
            snapshot_index = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_value)
        else:
            snapshot_index = snapshot_value

        if not header:
            header = self.read_header(
                'index', snapshot_index, simulation_directory, snapshot_directory)

        path_file_name = self.get_snapshot_file_names_indices(snapshot_directory, snapshot_index)

        self.say('* reading species: {}'.format(self.species_read))

        # open snapshot file
        with h5py.File(path_file_name, 'r') as file_in:
            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # initialize arrays to store each prop for each species
            for spec_name in self.species_read:
                spec_id = self.species_dict[spec_name]
                part_number_tot = header['particle.numbers.total'][spec_id]

                # add species to particle dictionary
                part[spec_name] = ParticleDictionaryClass()

                # set element pointers if reading only subset of elements
                if (element_indices is not None and len(element_indices) and
                        element_indices != 'all'):
                    if np.isscalar(element_indices):
                        element_indices = [element_indices]
                    for element_i, element_index in enumerate(element_indices):
                        part[spec_name].element_pointer[element_index] = element_i

                # check if snapshot file happens not to have particles of this species
                if part_numbers_in_file[spec_id] > 0:
                    part_in = file_in['PartType' + str(spec_id)]
                else:
                    # this scenario should occur only for multi-file snapshot
                    if header['file.number.per.snapshot'] == 1:
                        raise OSError('no {} particles in snapshot file'.format(spec_name))

                    # need to read in other snapshot files until find one with particles of species
                    for file_i in range(1, header['file.number.per.snapshot']):
                        file_name_i = path_file_name.replace('.0.', '.{}.'.format(file_i))
                        # try each snapshot file
                        file_in_i = h5py.File(file_name_i, 'r')
                        part_numbers_in_file_i = file_in_i['Header'].attrs['NumPart_ThisFile']
                        if part_numbers_in_file_i[spec_id] > 0:
                            # found one
                            part_in = file_in_i['PartType' + str(spec_id)]
                            break
                    else:
                        # tried all files and still did not find particles of species
                        raise OSError('no {} particles in any snapshot file'.format(spec_name))

                props_print = []
                ignore_flag = False  # whether ignored any properties in the file
                for prop_in in part_in.keys():
                    if prop_in in properties:
                        prop = property_dict[prop_in]

                        # determine shape of prop array
                        if len(part_in[prop_in].shape) == 1:
                            prop_shape = part_number_tot
                        elif len(part_in[prop_in].shape) == 2:
                            prop_shape = [part_number_tot, part_in[prop_in].shape[1]]
                            if (prop_in == 'Metallicity' and element_indices is not None and
                                    element_indices != 'all'):
                                prop_shape = [part_number_tot, len(element_indices)]

                        # determine data type to store
                        prop_in_dtype = part_in[prop_in].dtype
                        if convert_float32 and prop_in_dtype == 'float64':
                            prop_in_dtype = np.float32

                        # initialize to -1's
                        part[spec_name][prop] = np.zeros(prop_shape, prop_in_dtype) - 1

                        if prop == 'id':
                            # initialize so calling an un-itialized value leads to error
                            part[spec_name][prop] -= part_number_tot

                        if prop_in in property_dict:
                            props_print.append(property_dict[prop_in])
                        else:
                            props_print.append(prop_in)
                    else:
                        ignore_flag = True

                if ignore_flag:
                    props_print.sort()
                    self.say('* reading {} properties: {}'.format(spec_name, props_print))

                # special case: particle mass is fixed and given in mass array in header
                if 'Masses' in properties and 'Masses' not in part_in:
                    prop = property_dict['Masses']
                    part[spec_name][prop] = np.zeros(part_number_tot, dtype=np.float32)

        ## read properties for each species ----------
        # initial particle indices to assign to each species from each file
        part_indices_lo = np.zeros(len(self.species_read), dtype=np.int64)

        if header['file.number.per.snapshot'] == 1:
            self.say('* reading particles from:\n    {}'.format(path_file_name.strip('./')))
        else:
            self.say('* reading particles from:')

        # loop over all files at given snapshot
        for file_i in range(header['file.number.per.snapshot']):
            # open i'th of multiple files for snapshot
            file_name_i = path_file_name.replace('.0.', '.{}.'.format(file_i))

            # open snapshot file
            with h5py.File(file_name_i, 'r') as file_in:
                if header['file.number.per.snapshot'] > 1:
                    self.say('  ' + file_name_i.split('/')[-1])

                part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

                # read particle properties
                for spec_i, spec_name in enumerate(self.species_read):
                    spec_id = self.species_dict[spec_name]
                    if part_numbers_in_file[spec_id] > 0:
                        part_in = file_in['PartType' + str(spec_id)]

                        part_index_lo = part_indices_lo[spec_i]
                        part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                        # check if mass of species is fixed, according to header mass array
                        if 'Masses' in properties and header['particle.masses'][spec_id] > 0:
                            prop = property_dict['Masses']
                            part[spec_name][prop][
                                part_index_lo:part_index_hi] = header['particle.masses'][spec_id]

                        for prop_in in part_in.keys():
                            if prop_in in properties:
                                prop = property_dict[prop_in]
                                if len(part_in[prop_in].shape) == 1:
                                    part[spec_name][prop][part_index_lo:part_index_hi] = (
                                        part_in[prop_in])
                                elif len(part_in[prop_in].shape) == 2:
                                    if (prop_in == 'Metallicity' and element_indices is not None and
                                            element_indices != 'all'):
                                        prop_in = part_in[prop_in][:, element_indices]
                                    else:
                                        prop_in = part_in[prop_in]

                                    part[spec_name][prop][part_index_lo:part_index_hi, :] = prop_in

                        part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        print()

        return part

    def adjust_particle_properties(
        self, part, header, particle_subsample_factor=None, separate_dark_lowres=True,
        sort_dark_by_id=False):
        '''
        Adjust properties for each species, including unit conversions, separating dark species by
        mass, sorting by id, and subsampling.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        header : dict : header dictionary
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        separate_dark_lowres : boolean :
            whether to separate low-resolution dark matter into separate dicts according to mass
        sort_dark_by_id : boolean : whether to sort dark-matter particles by id
        '''
        # if dark2 contains different masses (refinements), split into separate dicts
        species_name = 'dark2'

        if species_name in part and 'mass' in part[species_name]:
            dark_lowres_masses = np.unique(part[species_name]['mass'])
            if dark_lowres_masses.size > 9:
                self.say('! warning: {} different masses of low-resolution dark matter'.format(
                         dark_lowres_masses.size))

            if separate_dark_lowres and dark_lowres_masses.size > 1:
                self.say('* separating low-resolution dark matter by mass into dictionaries')
                dark_lowres = {}
                for prop in part[species_name]:
                    dark_lowres[prop] = np.array(part[species_name][prop])

                for dark_i, dark_mass in enumerate(dark_lowres_masses):
                    spec_indices = np.where(dark_lowres['mass'] == dark_mass)[0]
                    spec_name = 'dark{}'.format(dark_i + 2)

                    part[spec_name] = ParticleDictionaryClass()

                    for prop in dark_lowres:
                        part[spec_name][prop] = dark_lowres[prop][spec_indices]
                    self.say('{}: {} particles'.format(spec_name, spec_indices.size))

                del(spec_indices)
                print()

        if sort_dark_by_id:
            # order dark-matter particles by id - should be conserved across snapshots
            self.say('* sorting the following dark particles by id:')
            for spec_name in part:
                if 'dark' in spec_name and 'id' in part[spec_name]:
                    indices_sorted = np.argsort(part[spec_name]['id'])
                    self.say('{}: {} particles'.format(spec_name, indices_sorted.size))
                    for prop in part[spec_name]:
                        part[spec_name][prop] = part[spec_name][prop][indices_sorted]
            del(indices_sorted)
            print()

        # apply unit conversions
        for spec_name in part:
            if 'position' in part[spec_name]:
                # convert to [kpc comoving]
                part[spec_name]['position'] /= header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to [km / s physical]
                part[spec_name]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'mass' in part[spec_name]:
                # convert to [M_sun]
                part[spec_name]['mass'] *= 1e10 / header['hubble']

            if 'bh.mass' in part[spec_name]:
                # convert to [M_sun]
                part[spec_name]['bh.mass'] *= 1e10 / header['hubble']

            if 'density' in part[spec_name]:
                # convert to [M_sun / kpc^3 physical]
                part[spec_name]['density'] *= (
                    1e10 / header['hubble'] / (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to [pc physical]
                part[spec_name]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                # convert to Plummer softening - 2.8 is valid for cubic spline
                # alternately, to convert to Gaussian scale length, divide by 2
                part[spec_name]['smooth.length'] /= 2.8

            if 'form.scalefactor' in part[spec_name]:
                if header['is.cosmological']:
                    pass
                else:
                    part[spec_name]['form.scalefactor'] /= header['hubble']  # convert to [Gyr]

            if 'temperature' in part[spec_name]:
                # convert from [(km / s) ^ 2] to [Kelvin]
                # ignore small corrections from elements beyond He
                helium_mass_fracs = part[spec_name]['massfraction'][:, 1]
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + part[spec_name]['electron.fraction'])
                molecular_weights = mus * ut.constant.proton_mass
                part[spec_name]['temperature'] *= (
                    ut.constant.centi_per_kilo ** 2 * (self.gas_eos - 1) * molecular_weights /
                    ut.constant.boltzmann)
                del(helium_mass_fracs, ys_helium, mus, molecular_weights)

            if 'potential' in part[spec_name]:
                # convert to [km^2 / s^2 physical]
                # TO DO: check if Gizmo writes potential as m / r, in raw units?
                # might need to add:
                # M *= 1e10 / header['hubble'] to get Msun
                # r /= header['hubble'] to get kpc physical
                # G conversion?
                part[spec_name]['potential'] /= header['scalefactor']

            if 'acceleration' in part[spec_name]:
                # convert to [km / s^2 physical]
                # consistent with v^2 / r at z = 0.5, TO DO check at z = 0
                part[spec_name]['acceleration'] *= header['hubble']

        # renormalize so potential max = 0
        renormalize_potential = False
        if renormalize_potential:
            potential_max = 0
            for spec_name in part:
                if part[spec_name]['potential'].max() > potential_max:
                    potential_max = part[spec_name]['potential'].max()
            for spec_name in part:
                part[spec_name]['potential'] -= potential_max

        # sub-sample particles, for smaller memory
        if particle_subsample_factor is not None and particle_subsample_factor > 1:
            self.say('* periodically subsampling all particles by factor = {}'.format(
                     particle_subsample_factor), end='\n\n')
            for spec_name in part:
                for prop in part[spec_name]:
                    part[spec_name][prop] = part[spec_name][prop][::particle_subsample_factor]

    def get_snapshot_file_names_indices(self, directory, snapshot_index=None):
        '''
        Get name of file or directory (with relative path) and index for all snapshots in directory.
        If input valid snapshot_index, get its file name (if multiple files per snapshot, get name
        of 0th one).
        If input snapshot_index as None or 'all', get name of file/directory and index for each
        snapshot file/directory.

        Parameters
        ----------
        directory : string : directory to check for files
        snapshot_index : int : index of snapshot: if None or 'all', get all snapshots in directory

        Returns
        -------
        path_file_name[s] : string or list thereof : (relative) path + name of file[s]
        [file_indices : list of ints : indices of snapshot files]
        '''
        directory = ut.io.get_path(directory)

        # get names and indices of all snapshot files in directory
        path_file_names, file_indices = ut.io.get_file_names(
            directory + self.snapshot_name_base, (int, float))

        # if ask for all snapshots, return all files/directories and indices
        if snapshot_index is None or snapshot_index == 'all':
            return path_file_names, file_indices

        # else get file name for single snapshot
        if snapshot_index < 0:
            snapshot_index = file_indices[snapshot_index]  # allow negative indexing of snapshots
        elif snapshot_index not in file_indices:
            raise OSError(
                'cannot find snapshot index = {} in:  {}'.format(snapshot_index, path_file_names))

        path_file_name = path_file_names[np.where(file_indices == snapshot_index)[0][0]]

        if self.file_extension not in path_file_name:
            # got snapshot directory with multiple files, return only 0th one
            path_file_names = ut.io.get_file_names(path_file_name + '/' + self.snapshot_name_base)
            if len(path_file_names) and '.0.' in path_file_names[0]:
                path_file_name = path_file_names[0]
            else:
                raise OSError('cannot find 0th snapshot file in:  {}'.format(path_file_names))

        return path_file_name

    def get_cosmology(
        self, directory='.', omega_lambda=None, omega_matter=None, omega_baryon=None, hubble=None,
        sigma_8=None, n_s=None):
        '''
        Get cosmological parameters, stored in Cosmology class.
        Read cosmological parameters from MUSIC initial condition config file.
        If cannot find file, assume AGORA cosmology as default.

        Parameters
        ----------
        directory : string : directory of simulation (where directory of initial conditions is)

        Returns
        -------
        Cosmology : class : stores cosmological parameters and functions
        '''

        def get_check_value(line, value_test=None):
            frac_dif_max = 0.01
            value = float(line.split('=')[-1].strip())
            if 'h0' in line:
                value /= 100
            if value_test is not None:
                frac_dif = np.abs((value - value_test) / value)
                if frac_dif > frac_dif_max:
                    print('! read {}, but previously assigned = {}'.format(line, value_test))
            return value

        if directory:
            # find MUSIC file, assuming named *.conf
            file_name_find = ut.io.get_path(directory) + '*/*.conf'
            path_file_names = ut.io.get_file_names(file_name_find, verbose=False)
            if len(path_file_names):
                path_file_name = path_file_names[0]
                self.say('* reading cosmological parameters from:  {}'.format(
                         path_file_name.strip('./')), end='\n\n')
                # read cosmological parameters
                with open(path_file_name, 'r') as file_in:
                    for line in file_in:
                        line = line.lower().strip().strip('\n')  # ensure lowercase for safety
                        if 'omega_l' in line:
                            omega_lambda = get_check_value(line, omega_lambda)
                        elif 'omega_m' in line:
                            omega_matter = get_check_value(line, omega_matter)
                        elif 'omega_b' in line:
                            omega_baryon = get_check_value(line, omega_baryon)
                        elif 'h0' in line:
                            hubble = get_check_value(line, hubble)
                        elif 'sigma_8' in line:
                            sigma_8 = get_check_value(line, sigma_8)
                        elif 'nspec' in line:
                            n_s = get_check_value(line, n_s)
            else:
                self.say('! cannot find MUSIC config file:  {}'.format(file_name_find.strip('./')))

        # AGORA box (use as default, if cannot find MUSIC config file)
        if omega_baryon is None or sigma_8 is None or n_s is None:
            self.say('! missing cosmological parameters, assuming the following (from AGORA box):')
            if omega_baryon is None:
                omega_baryon = 0.0455
                self.say('assuming omega_baryon = {}'.format(omega_baryon))
            if sigma_8 is None:
                sigma_8 = 0.807
                self.say('assuming sigma_8 = {}'.format(sigma_8))
            if n_s is None:
                n_s = 0.961
                self.say('assuming n_s = {}'.format(n_s))
            self.say('')

        Cosmology = ut.cosmology.CosmologyClass(
            omega_lambda, omega_matter, omega_baryon, hubble, sigma_8, n_s)

        return Cosmology

    def check_properties(self, part):
        '''
        Checks sanity of particle properties, print warning if they are outside given limits.

        Parameters
        ----------
        part : dictionary class : catalog of particles
        '''
        # limits of sanity
        prop_limit_dict = {
            'id': [0, 4e9],
            'id.child': [0, 4e9],
            'id.generation': [0, 4e9],
            'position': [0, 1e6],  # [kpc comoving]
            'velocity': [-1e5, 1e5],  # [km / s]
            'mass': [9, 1e11],  # [M_sun]
            'potential': [-1e9, 1e9],  # [km^2 / s^2]
            'temperature': [3, 1e9],  # [K]
            'density': [0, 1e14],  # [M_sun/kpc^3]
            'smooth.length': [0, 1e9],  # [kpc physical]
            'hydrogen.neutral.fraction': [0, 1],
            'sfr': [0, 1000],  # [M_sun/yr]
            'massfraction': [0, 1],
            'form.scalefactor': [0, 1],
        }

        mass_factor_wrt_median = 4  # mass should not vary by more than this!

        self.say('* checking sanity of particle properties')

        for spec_name in part:
            for prop in [k for k in prop_limit_dict if k in part[spec_name]]:
                if (part[spec_name][prop].min() < prop_limit_dict[prop][0] or
                        part[spec_name][prop].max() > prop_limit_dict[prop][1]):
                    self.say(
                        '! warning: {} {} [min, max] = [{}, {}]'.format(
                            spec_name, prop,
                            ut.io.get_string_from_numbers(part[spec_name][prop].min(), 3),
                            ut.io.get_string_from_numbers(part[spec_name][prop].max(), 3))
                    )
                elif prop is 'mass' and spec_name in ['star', 'gas', 'dark']:
                    m_min = np.median(part[spec_name][prop]) / mass_factor_wrt_median
                    m_max = np.median(part[spec_name][prop]) * mass_factor_wrt_median
                    if part[spec_name][prop].min() < m_min or part[spec_name][prop].max() > m_max:
                        self.say(
                            '! warning: {} {} [min, med, max] = [{}, {}, {}]'.format(
                                spec_name, prop,
                                ut.io.get_string_from_numbers(part[spec_name][prop].min(), 3),
                                ut.io.get_string_from_numbers(np.median(part[spec_name][prop]), 3),
                                ut.io.get_string_from_numbers(part[spec_name][prop].max(), 3))
                        )

        print()

    def assign_host_coordinates(
        self, part, species_name='', part_indices=None, method='center-of-mass', host_number=1):
        '''
        Assign center position[s] [kpc comoving] and velocity[s] [km / s] wrt host
        galaxy[s]/halo[s].
        Use species_name, if defined, else default to stars for baryonic simulation or
        dark matter for dark matter-only simulation.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        species_name : string : which particle species to use to define center
        part_indices : array : list of indices of particle to use to define center
            use this to exclude particles that you know are not relevant
        method : string : method of centering: 'center-of-mass', 'potential'
        host_number : int : number of hosts to assign
        '''
        if (species_name in part and 'position' in part[species_name] and
                len(part[species_name]['position'])):
            pass
        elif 'star' in part and 'position' in part['star'] and len(part['star']['position']):
            species_name = 'star'
        elif 'dark' in part and 'position' in part['dark'] and len(part['dark']['position']):
            species_name = 'dark'
        else:
            self.say('! catalog not contain star or dark particles, cannot assign host coordinates')
            return

        if species_name is 'star':
            velocity_radius_max = 15
        elif species_name is 'dark':
            velocity_radius_max = 30

        self.say('* assigning coordinates for {} host galaxy/halo[s]:'.format(host_number))

        if 'position' in part[species_name]:
            # assign to overall dictionary
            part.host_positions = ut.particle.get_center_positions(
                part, species_name, part_indices, method, host_number, return_array=False)
            # assign to each species dictionary
            for spec_name in part:
                part[spec_name].host_positions = part.host_positions

            for host_position in part.host_positions:
                self.say('position = (', end='')
                ut.io.print_array(host_position, '{:.3f}', end='')
                print(') [kpc comoving]')

        if 'velocity' in part[species_name]:
            # assign to overall dictionary
            part.host_velocities = ut.particle.get_center_velocities(
                part, species_name, part_indices, velocity_radius_max, part.host_positions,
                return_array=False)
            # assign to each species dictionary
            for spec_name in part:
                part[spec_name].host_velocities = part.host_velocities

            for host_velocity in part.host_velocities:
                self.say('velocity = (', end='')
                ut.io.print_array(host_velocity, '{:.1f}', end='')
                print(') [km / s]')

        print()

    def assign_host_principal_axes(self, part, distance_max=15, mass_percent=90, age_percent=30):
        '''
        Assign rotation vectors of principal axes (via moment of inertia tensor) of host
        galaxy[s]/halo[s], using stars for baryonic simulations.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        distance_max : float : maximum distance to select particles [kpc physical]
        mass_percent : float : keep particles within the distance that encloses mass percent
            [0, 100] of all particles within distance_max
        age_percent : float : keep youngest age_percent of particles within distance cut
        '''
        spec_name = 'star'

        if spec_name not in part or not len(part[spec_name]['position']):
            self.say('! catalog not contain star particles, so cannot assign principal axes')
            return

        self.say('* assigning principal axes of host galaxy[s]/halo[s]:')
        self.say('using {} particles at distance < {} kpc'.format(spec_name, distance_max))

        if mass_percent:
            self.say('using distance that encloses {}% of mass'.format(mass_percent))

        if age_percent:
            if ('form.scalefactor' not in part[spec_name] or
                    not len(part[spec_name]['form.scalefactor'])):
                self.say('! catalog not contain {} ages'.format(spec_name))
                self.say('so assigning principal axes using all {} particles'.format(spec_name))
            else:
                self.say('using youngest {}% of {} particles'.format(age_percent, spec_name))

        principal_axes = ut.particle.get_principal_axes(
            part, spec_name, distance_max, mass_percent, age_percent,
            center_positions=part.host_positions, return_array=False, print_results=False)

        part.host_rotation_tensors = principal_axes['rotation.tensor']
        for spec_name in part:
            part[spec_name].host_rotation_tensors = part.host_rotation_tensors

        for center_i in range(part.host_positions.shape[0]):
            self.say('axis ratios: min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
                     principal_axes['axis.ratios'][center_i, 0],
                     principal_axes['axis.ratios'][center_i, 1],
                     principal_axes['axis.ratios'][center_i, 2]))

        print()

    def assign_host_orbits(self, part, species=[], host_positions=None, host_velocities=None):
        '''
        Assign derived orbital properties wrt single center to species.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        species : string or list : particle species to compute
        center_positions : array or array of arrays : center position[s] to use
        center_velocities : array or array of arrays : center velocity[s] to use
        '''
        if not species:
            species = ['star', 'gas', 'dark']
        species = ut.particle.parse_species(part, species)

        self.say('* assigning orbital properties wrt galaxy/halo to {}'.format(species))

        if host_positions is None:
            host_positions = part.host_positions
        if host_velocities is None:
            host_velocities = part.host_velocities

        for center_i, center_position in enumerate(host_positions):
            center_velocity = host_velocities[center_i]

            orb = ut.particle.get_orbit_dictionary(
                part, species, None, center_position, center_velocity, return_single=False)

            host_name = ut.catalog.get_host_name(center_i)

            for spec_name in species:
                for prop in orb[spec_name]:
                    part[spec_name][host_name + prop] = orb[spec_name][prop]

    # write to file ----------
    def rewrite_snapshot(
        self, species='gas', action='delete', value_adjust=None,
        snapshot_value_kind='redshift', snapshot_value=0,
        simulation_directory='.', snapshot_directory='output/'):
        '''
        Read snapshot file[s].
        Rewrite, deleting given species.

        Parameters
        ----------
        species : string or list : name[s] of particle species to delete:
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark2' = dark matter at lower resolution
            'star' = stars
            'blackhole' = black holes
        action : string : what to do to snapshot file: 'delete', 'velocity'
        value_adjust : float : value by which to adjust property (if not deleting)
        snapshot_value_kind : string : input snapshot number kind: 'index', 'redshift'
        snapshot_value : int or float : index (number) of snapshot file
        simulation_directory : root directory of simulation
        snapshot_directory : string : directory of snapshot files within simulation_directory
        '''
        if np.isscalar(species):
            species = [species]  # ensure is list

        ## read information about snapshot times ----------
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = simulation_directory + ut.io.get_path(snapshot_directory)

        Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
        snapshot_index = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_value)

        path_file_name = self.get_snapshot_file_names_indices(snapshot_directory, snapshot_index)
        self.say('* reading header from:  {}'.format(path_file_name.strip('./')), end='\n\n')

        ## read header ----------
        # open snapshot file and parse header
        with h5py.File(path_file_name, 'r+') as file_in:
            header = file_in['Header'].attrs  # load header dictionary

            ## read and delete input species ----------
            for file_i in range(header['NumFilesPerSnapshot']):
                # open i'th of multiple files for snapshot
                file_name_i = path_file_name.replace('.0.', '.{}.'.format(file_i))
                file_in = h5py.File(file_name_i, 'r+')

                self.say('reading particles from: ' + file_name_i.split('/')[-1])

                if 'delete' in action:
                    part_number_in_file = header['NumPart_ThisFile']
                    part_number = header['NumPart_Total']

                # read and delete particle properties
                for _spec_i, spec_name in enumerate(species):
                    spec_id = self.species_dict[spec_name]
                    spec_in = 'PartType' + str(spec_id)
                    self.say('adjusting species = {}'.format(spec_name))

                    if 'delete' in action:
                        self.say('deleting species = {}'.format(spec_name))

                        # zero numbers in header
                        part_number_in_file[spec_id] = 0
                        part_number[spec_id] = 0

                        # delete properties
                        #for prop in file_in[spec_in]:
                        #    del(file_in[spec_in + '/' + prop])
                        #    self.say('  deleting {}'.format(prop))

                        del(file_in[spec_in])

                    elif 'velocity' in action and value_adjust:
                        dimension_index = 2  # boost velocity along z-axis
                        self.say('  boosting velocity along axis.{} by {:.1f} km/s'.format(
                                 dimension_index, value_adjust))
                        velocities = file_in[spec_in + '/' + 'Velocities']
                        scalefactor = 1 / (1 + header['Redshift'])
                        velocities[:, 2] += value_adjust / np.sqrt(scalefactor)
                        #file_in[spec_in + '/' + 'Velocities'] = velocities

                    print()

                if 'delete' in action:
                    header['NumPart_ThisFile'] = part_number_in_file
                    header['NumPart_Total'] = part_number


Read = ReadClass()


#===================================================================================================
# write snapshot text file
#===================================================================================================
def write_snapshot_text(part):
    '''
    Write snapshot to text file, one file per species.

    Parameters
    ----------
    part : dictionary class : catalog of particles at snapshot
    '''
    spec_name = 'dark'
    file_name = 'snapshot_{}_{}.txt'.format(part.snapshot['index'], spec_name)
    part_spec = part[spec_name]

    with open(file_name, 'w') as file_out:
        file_out.write(
            '# id mass[M_sun] distance_wrt_host(x,y,z)[kpc] velocity_wrt_host(x,y,z)[km/s]\n')

        for pi in range(len(part_spec['id'])):
            file_out.write(
                '{} {:.3e} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}\n'.format(
                    part_spec['id'][pi], part_spec['mass'][pi],
                    part_spec.prop('host.distance', pi)[0], part_spec.prop('host.distance', pi)[1],
                    part_spec.prop('host.distance', pi)[2],
                    part_spec.prop('host.velocity', pi)[0], part_spec.prop('host.velocity', pi)[1],
                    part_spec.prop('host.velocity', pi)[2],
                )
            )

    spec_name = 'gas'
    file_name = 'snapshot_{}_{}.txt'.format(part.snapshot['index'], spec_name)
    part_spec = part[spec_name]

    with open(file_name, 'w') as file_out:
        file_out.write(
            '# id mass[M_sun] distance_wrt_host(x,y,z)[kpc] velocity_wrt_host(x,y,z)[km/s] ' +
            'density[M_sun/kpc^3] temperature[K]\n')

        for pi in range(len(part_spec['id'])):
            file_out.write(
                '{} {:.3e} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f} {:.2e} {:.2e}\n'.format(
                    part_spec['id'][pi], part_spec['mass'][pi],
                    part_spec.prop('host.distance', pi)[0], part_spec.prop('host.distance', pi)[1],
                    part_spec.prop('host.distance', pi)[2],
                    part_spec.prop('host.velocity', pi)[0], part_spec.prop('host.velocity', pi)[1],
                    part_spec.prop('host.velocity', pi)[2],
                    part_spec['density'][pi], part_spec['temperature'][pi],
                )
            )

    spec_name = 'star'
    file_name = 'snapshot_{}_{}.txt'.format(part.snapshot['index'], spec_name)
    part_spec = part[spec_name]

    with open(file_name, 'w') as file_out:
        file_out.write(
            '# id mass[M_sun] distance_wrt_host(x,y,z)[kpc] velocity_wrt_host(x,y,z)[km/s] ' +
            'age[Gyr]\n')

        for pi in range(len(part_spec['id'])):
            file_out.write(
                '{} {:.3e} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(
                    part_spec['id'][pi], part_spec['mass'][pi],
                    part_spec.prop('host.distance', pi)[0], part_spec.prop('host.distance', pi)[1],
                    part_spec.prop('host.distance', pi)[2],
                    part_spec.prop('host.velocity', pi)[0], part_spec.prop('host.velocity', pi)[1],
                    part_spec.prop('host.velocity', pi)[2],
                    part_spec.prop('age', pi),
                )
            )
