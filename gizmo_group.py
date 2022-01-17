'''
Generate FoF group catalogs of particles, to identify star clusters or gas clouds (GMCs).
Read and write them to HDF5 files.
Analyze/plot them.

@author:
    Andrew Wetzel <arwetzel@gmail.com>

----------
Units
    unless otherwise noted, this package converts all quantities to (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance [kpc physical]
    radius [pc physical]
    velocity [km / s]
    time [Gyr]

----------
Reading a group catalog

Within a simulation directory, read all groups in the snapshot at redshift 0 via:
    grp = gizmo.group.IO.read_group_catalogs(species_name)
grp is a dictionary, with a key for each property. So, access via:
    grp[property_name]
For example:
    grp['mass']
returns a numpy array of masses, one for each group, while
    grp['position']
returns a numpy array of positions (of dimension particle_number x 3)

----------
Default/stored properties in group catalog
    'indices' : indices (within the particle catalog) of member partices
    'number' : total number of member particles [M_sun]
    'mass' : total mass of members particles [M_sun]
    'position' or 'host.distance.principal' :
        3-D COM artesian distance wrt center of primary host along its principal axes [kpc physical]
    'velocity' or 'host.velocity.principal' :
        3-D COM velocity wrt center of primary host along its principal axes [km / s]
    'radius.100' : maximum distance of member particles [pc physical]
    'radius.90' : distance that encloses 90% of member mass [pc physical]
    'radius.50' : istance that encloses 50% of member mass [pc physical]
    'vel.std' : 68% width of the mass-weighted velocity distribution of members [km / s]
    'temperature' : mass-weighted temperature of members [K]

----------
Derived properties

grp is a GroupDictionaryClass that can compute derived properties on the fly.
Call derived (or stored) properties via:
    grp.prop(property_name)
Examples:
    grp.prop('density.50')

You also can call stored properties via grp.prop(property_name).
The package will know that it is a stored property and return it as is.
For example, grp.prop('position') is the same as grp['position'].

See GroupDictionaryClass.prop() for full option for parsing of derived properties.
'''

from matplotlib import pyplot as plt
import numpy as np

import utilities as ut
from . import gizmo_default


# --------------------------------------------------------------------------------------------------
# dictionary class to store groups of particles
# --------------------------------------------------------------------------------------------------
class GroupDictionaryClass(dict,):
    '''
    Dictionary class to store catalog of FoF groups.
    Allows production of derived quantities.
    '''

    def __init__(self):
        self.info = {}
        self.snapshot = {}
        self.Snapshot = None
        self.Cosmology = None
        self.host = None

    def prop(self, property_name, indices=None, _dict_only=False):
        '''
        Get property, either from self dictionary or derive.
        If several properties, need to provide mathematical relationship.

        Parameters
        ----------
        property_name : str
            name of property
        indices : array
            list of indices to select on (of arbitrary dimensions)
        _dict_only : bool
            require property_name to be in self's dict - avoids endless recursion

        Returns
        -------
        values : float or array
            depending on dimensionality of input indices
        '''
        # parsing general to all catalogs ----------
        property_name = property_name.strip()  # strip white space

        # if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]
        elif _dict_only:
            raise KeyError(f'property = {property_name} is not in self\'s dictionary')

        # math relation, combining more than one property
        if (
            '/' in property_name
            or '*' in property_name
            or '+' in property_name
            or '-' in property_name
        ):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise ValueError(f'not sure how to parse property = {property_name}')

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
                            prop_values[masks] / self.prop(prop_name, indices)[masks]
                        )
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

        # parsing specific to FoF group catalog ----------
        # default radius
        if property_name == 'radius':
            return self.prop('radius.100', indices, _dict_only=True)

        # mass density or virial parameter (alpha_vir)
        if 'density' in property_name or 'virial.parameter' in property_name:
            if property_name == 'density' or property_name == 'virial.parameter':
                property_name += '.90'  # use R_90 as default radius if not input

            radius_percent = property_name.split('.')[-1]
            radiuss = self.prop('radius.' + radius_percent, indices, _dict_only=True)
            masses = float(radius_percent) / 100 * self.prop('mass', indices, _dict_only=True)

            if 'density' in property_name:
                # mean mass density [Msun / pc ^ 2]
                values = masses / (4 / 3 * np.pi * radiuss ** 3)
            elif 'virial.parameter' in property_name:
                # alpha_vir := 5 R sigma_vel,1D ^ 2 / GM [dimensionless]
                # convert to 1-D dispersion, by convention
                vel_stds = self.prop('vel.std', indices, _dict_only=True) / np.sqrt(3)
                values = (
                    5
                    * (radiuss * ut.constant.km_per_pc)
                    * vel_stds ** 2
                    / ut.constant.grav_km_msun_sec
                    / masses
                )

            return values

        # velocity dispersion along 1 dimension
        if 'vel.' in property_name and '.1d' in property_name:
            values = self.prop(property_name.replace('.1d', ''), indices) / np.sqrt(3)

        # distance/velocity wrt center of a primary host
        if 'host' in property_name:
            if 'host.' in property_name or 'host1.' in property_name:
                host_name = 'host.'
            elif 'host2.' in property_name:
                host_name = 'host2.'
            elif 'host3.' in property_name:
                host_name = 'host3.'
            else:
                raise ValueError(f'cannot identify host name in {property_name}')

            if host_name + 'distance' in property_name:
                values = self.prop(host_name + 'distance', indices, _dict_only=True)
            elif host_name + 'velocity' in property_name:
                values = self.prop(host_name + 'velocity', indices, _dict_only=True)

            if '.spher' in property_name:
                # convert to spherical coordinates
                if 'distance' in property_name:
                    # along R (positive definite), theta [0, pi), phi [0, 2 * pi)
                    values = ut.coordinate.get_positions_in_coordinate_system(
                        values, 'cartesian', 'spherical'
                    )
                if 'velocity' in property_name:
                    # along v_R, v_theta, v_phi
                    if 'principal' in property_name:
                        distance_vectors = self.prop(host_name + 'distance.principal', indices)
                    else:
                        distance_vectors = self.prop(
                            host_name + 'distance', indices, _dict_only=True
                        )
                    values = ut.coordinate.get_velocities_in_coordinate_system(
                        values, distance_vectors, 'cartesian', 'spherical'
                    )

            if 'total' in property_name:
                # compute total (scalar) distance / velocity
                if len(values.shape) == 1:
                    shape_pos = 0
                else:
                    shape_pos = 1
                values = np.sqrt(np.sum(values ** 2, shape_pos))

            return values

        # should not get this far without a return
        raise KeyError(f'not sure how to parse property = {property_name}')

    def get_indices(
        self,
        particle_number_min=6,
        mass_limits=[1, None],
        density_limits=[None, None],
        host_distance_limits=None,
        object_kind='',
        prior_indices=None,
    ):
        '''
        Get indices of groups that satisfy input selection limits.

        Parameters
        ----------
        particle_number_min : int
            minimum number of member particles
        mass_limits : list
            min and max limits for total mass [M_sun]
        density_limits : list
            min and max limits for average mass density within R_50 [M_sun / kpc^3]
        host_distance_limits : list
            min and max limits for distance to host [kpc physical]
        object_kind : str
            [unused placeholder for now]
        prior_indices : array
            prior indices of groups to impose

        Returns
        -------
        gindices : array
            indices of groups
        '''
        if object_kind:
            print(f'GroupDictionary.get_indices() does not support input object_kind={object_kind}')

        gindices = prior_indices
        if gindices is None or len(gindices) == 0:
            gindices = ut.array.get_arange(self['mass'])

        if 'mass' in self and np.max(self['mass']) > 0:
            if particle_number_min > 0:
                gindices = ut.array.get_indices(
                    self['number'], [particle_number_min, np.Inf], gindices
                )

            if mass_limits is not None and len(mass_limits) > 0:
                gindices = ut.array.get_indices(self['mass'], mass_limits, gindices)

            if density_limits is not None and len(density_limits) > 0:
                gindices = ut.array.get_indices(self.prop('density.50'), density_limits, gindices)

        # properties for satellites of host(s)
        if host_distance_limits is not None and len(host_distance_limits) > 0:
            gindices = ut.array.get_indices(
                self.prop('host.distance.total'), host_distance_limits, gindices
            )

        return gindices


class IOClass(ut.io.SayClass):
    '''
    Read or write group catalog HDF5 files, or generate groups from a particle catalog.
    '''

    def __init__(
        self,
        simulation_directory=gizmo_default.simulation_directory,
        group_directory=gizmo_default.group_directory,
    ):
        '''
        '''
        self.file_name_base = '{}_group.fof.{:.0f}pc_{:03d}'
        self.species_names = ['gas', 'star', 'dark']
        self.particle_properties_read = [
            'position',
            'mass',
            'velocity',
            'density',
            'temperature',
            'form.scalefactor',
        ]
        self.property_name_default = 'mass'  # default property for iterating

        # set directories
        self.simulation_directory = ut.io.get_path(simulation_directory)
        self.group_directory = ut.io.get_path(group_directory)

    def generate_group_catalog(
        self,
        part,
        species_name='gas',
        linking_length=20,
        particle_number_min=10,
        property_select={'number.density': [10, np.Inf], 'temperature': [0, 1e4]},
        dimension_indices=None,
        host_index=0,
        part_indices=None,
    ):
        '''
        Generate and get dictionary catalog of FoF groups of particles of input species.

        Parameters
        ----------
        part : dict
            catalog of particles at snapshot
        species_name : str
            name of particle species to use
        linking_length : float
            maximum distance to link neighbors [pc physical]
        particle_number_min : int
            minimum number of member particles to keep a group
        property_select : dict
            properties to select particles on: names as keys and limits as values
        dimension_indices : list
            which dimensions of positions to use (use this to run in 2-D)
        host_index : int
            index of host galaxy to use to get positions and velocities around
        part_indices : array
            prior indices[s] of particles to select
        '''
        grp_dict = ut.particle.get_fof_group_catalog(
            part,
            species_name,
            linking_length,
            particle_number_min,
            dimension_indices,
            host_index,
            property_select,
            part_indices,
        )

        if grp_dict is None:
            return

        part_spec = part[species_name]

        # transfer group to dictionary class
        grp = GroupDictionaryClass()
        for prop_name in grp_dict:
            grp[prop_name] = grp_dict[prop_name]

        # transfer meta-data from particle catalog
        grp.info = dict(part_spec.info)
        grp.info['catalog.kind'] = f'{species_name}.group.fof'
        grp.info['fof.species'] = species_name
        grp.info['fof.linking.length'] = linking_length
        for prop_name in property_select:
            grp.info['fof.' + prop_name] = property_select[prop_name]
        grp.snapshot = dict(part_spec.snapshot)
        grp.Snapshot = part_spec.Snapshot
        grp.Cosmology = part_spec.Cosmology
        grp.host = dict(part_spec.host)

        return grp

    def read_group_catalogs(
        self,
        species_name='gas',
        linking_length=20,
        snapshot_value_kind='redshift',
        snapshot_values=0,
        simulation_directory=None,
        group_directory=None,
        all_snapshot_list=True,
        simulation_name='',
    ):
        '''
        Read catalog[s] of groups at snapshot[s].
        Return as dictionary or list of dictionaries.

        Parameters
        ----------
        species_name : str
            name of particle species of group catalog: 'gas', 'star', 'dark'
        linking_length : float
            FoF linking length [pc physical]
        snapshot_value_kind : string
            snapshot value kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
            if 'all' or None, read all snapshots with group catalogs
        simulation_directory : string
            base directory of simulation
        group_directory : string
            directory  of group catalog files
        host_number : int
            number of hosts to assign and compute coordinates relative to
            if 0 or None, skip host assignment
        all_snapshot_list : bool
            if reading multiple snapshots, whether to create a list of group catalogs of length
            equal to all snapshots in simulation (so group catalog index = snapsht index)
        simulation_name : string
            name of simulation to store for future identification

        Returns
        -------
        grps : dictionary or list of dictionaries
            catalog[s] of groups at snapshot[s]
        '''
        # parse inputs
        assert species_name in self.species_names
        assert linking_length > 0

        if isinstance(snapshot_values, int):
            snapshot_values = [snapshot_values]  # ensure is list

        if simulation_directory is None or len(simulation_directory) == 0:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)
        if group_directory is None or len(group_directory) == 0:
            group_directory = self.group_directory
        else:
            group_directory = ut.io.get_path(group_directory)

        Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        grps = [[] for _ in Snapshot['index']]  # list of group catalogs across all snapshots

        if snapshot_values == 'all' or snapshot_values is None or len(snapshot_values) == 0:
            # read/write all snapshots
            snapshot_indices = Snapshot['index']
        else:
            # get snapshot index[s] corresponding to input snapshot values
            snapshot_indices = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_values)

        # get names of all existing group files
        path_file_names, file_snapshot_indices = self._get_group_file_names_and_indices(
            species_name, linking_length, simulation_directory + group_directory, snapshot_indices
        )

        if len(path_file_names) == 0:
            raise OSError(
                'cannot find any group catalog files in:  {}'.format(
                    simulation_directory + group_directory.lstrip('./')
                )
            )

        # read group catalog at all input snapshots
        for _path_file_name, snapshot_index in zip(path_file_names, file_snapshot_indices):
            grp = self._io_group_catalog(
                species_name,
                snapshot_index,
                linking_length,
                None,
                simulation_directory,
                group_directory,
            )

            if simulation_name:
                grp.info['simulation.name'] = simulation_name

            # assign information about all snapshots
            grp.Snapshot = Snapshot

            # if read single snapshot, return as dictionary instead of list
            if len(file_snapshot_indices) == 1:
                grps = grp
            else:
                grps[snapshot_index] = grp
                if snapshot_index != file_snapshot_indices[-1]:
                    print()

        if len(file_snapshot_indices) > 1 and not all_snapshot_list:
            grps = [grp for grp in grps if len(grp)]

        return grps

    def _io_group_catalog(
        self,
        species_name='gas',
        snapshot_index=600,
        linking_length=20,
        grp=None,
        simulation_directory=None,
        group_directory=None,
        verbose=True,
    ):
        '''
        Read/write a catalog of groups at a snapshot to/from HDF5 file.
        If reading, return as dictionary.

        Parameters
        ----------
        species_name : str
            name of particle species: 'gas', 'star', 'dark'
        snapshot_index : int
            index of snapshot
        linking_length : float
            FoF linking length [pc physical]
        grp : dictionary
            catalog of groups at a snapshot, if writing
        simulation_directory : string
            base directory of simulation
        group_directory : str
            directory (within a simulation_directory) of group catalog files
        verbose : bool
            whether to print each property read/written

        Returns
        -------
        grp : class
            catalog of groups at a snapshot
        '''
        # parse inputs
        if grp is None:
            # reading
            assert species_name in self.species_names
            assert snapshot_index
            assert linking_length > 0
        else:
            # writing
            species_name = grp.info['fof.species']
            snapshot_index = grp.snapshot['index']
            linking_length = grp.info['fof.linking.length']

        if simulation_directory is None or len(simulation_directory) == 0:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)
        if group_directory is None or len(group_directory) == 0:
            group_directory = self.group_directory
        else:
            group_directory = ut.io.get_path(group_directory)

        file_path = ut.io.get_path(simulation_directory) + ut.io.get_path(group_directory)
        if not snapshot_index:
            snapshot_index = grp.snapshot['index']

        file_name = self.file_name_base.format(species_name, linking_length, snapshot_index)
        path_file_name = file_path + file_name

        if grp is not None:
            # write to file
            file_path = ut.io.get_path(file_path, create_path=True)

            properties_add = []
            for prop_name in grp.info:
                if not isinstance(grp.info[prop_name], str) and grp.info[prop_name] is not None:
                    grp['info:' + prop_name] = np.array(grp.info[prop_name])
                    properties_add.append('info:' + prop_name)

            for prop_name in grp.snapshot:
                grp['snapshot:' + prop_name] = np.array(grp.snapshot[prop_name])
                properties_add.append('snapshot:' + prop_name)

            for prop_name in grp.Cosmology:
                grp['cosmology:' + prop_name] = np.array(grp.Cosmology[prop_name])
                properties_add.append('cosmology:' + prop_name)

            for prop_name in grp.host:
                grp['host:' + prop_name] = np.array(grp.host[prop_name])
                properties_add.append('host:' + prop_name)

            ut.io.file_hdf5(path_file_name, grp, verbose)

            for prop_name in properties_add:
                del grp[prop_name]

        else:
            # read from file

            # store as dictionary class
            grp = GroupDictionaryClass()
            grp.info = {}
            grp.snapshot = {}
            grp.Cosmology = None
            grp.host = {}
            header = {}

            try:
                # try to read from file
                grp_read = ut.io.file_hdf5(path_file_name, verbose=False)

            except OSError as exc:
                raise OSError(
                    '! cannot find FoF group catalog file:  {}'.format(path_file_name.lstrip('./'))
                ) from exc

            for prop_name in grp_read:
                if 'info:' in prop_name:
                    grp_prop_name = prop_name.split(':')[-1]
                    if grp_read[prop_name].size == 1:
                        grp.info[grp_prop_name] = float(grp_read[prop_name])
                    else:
                        grp.info[grp_prop_name] = grp_read[prop_name]
                elif 'snapshot:' in prop_name:
                    grp_prop_name = prop_name.split(':')[-1]
                    if grp_prop_name == 'index':
                        grp.snapshot[grp_prop_name] = int(grp_read[prop_name])
                    else:
                        grp.snapshot[grp_prop_name] = float(grp_read[prop_name])
                elif 'cosmology:' in prop_name:
                    grp_prop_name = prop_name.split(':')[-1]
                    header[grp_prop_name] = float(grp_read[prop_name])
                elif 'host:' in prop_name:
                    grp_prop_name = prop_name.split(':')[-1]
                    grp.host[grp_prop_name] = grp_read[prop_name]
                    # grp.host[grp_prop_name] = np.array(grp_read[prop_name], dtype=np.float32)
                else:
                    grp[prop_name] = grp_read[prop_name]

            grp.Cosmology = ut.cosmology.CosmologyClass(
                header['omega_lambda'],
                header['omega_matter'],
                header['omega_baryon'],
                header['hubble'],
                header['sigma_8'],
                header['n_s'],
                header['w'],
            )

            grp.info['catalog.kind'] = f'{species_name}.group.fof'
            grp.info['fof.species'] = species_name
            grp.info['simulation.name'] = ''

            self.say(
                '* read {} FoF groups from:  {}.hdf5'.format(
                    grp[self.property_name_default].size, path_file_name.lstrip('./')
                )
            )

            return grp

    def generate_write_group_catalogs(
        self,
        species_name='gas',
        linking_length=20,
        particle_number_min=10,
        property_select={'number.density': [10, np.Inf], 'temperature': [0, 1e4]},
        snapshot_value_kind='redshift',
        snapshot_values=0,
        simulation_directory=None,
        group_directory=None,
        proc_number=1,
        verbose=True,
    ):
        '''
        Read particles from snapshot[s], generate FoF group catalog of particle species
        write FoF group catalog to HDF5 file.
        By default, set up to run from within the base directory of a simulation.

        Parameters
        ----------
        species_name : str
            name of particle species to generate groups of
        linking_length : float
            FoF linking length [pc physical]
        particle_number_min : int
            minimum number of member particles to keep a group
        property_select : dict
            properties to select particles on: names as keys and limits as values
        snapshot_value_kind : str
            snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : string
            base directory of simulation
        group_directory : str
            directory (within a simulation_directory) of group catalog files
        proc_number : int
            number of parallel processes to run
        verbose : bool
            whether to print each property read/written
        '''
        # parse inputs
        assert species_name in self.species_names
        assert linking_length > 0
        assert particle_number_min > 1

        if isinstance(snapshot_values, int):
            snapshot_values = [snapshot_values]  # ensure is list

        if simulation_directory is None or len(simulation_directory) == 0:
            simulation_directory = self.simulation_directory
        else:
            simulation_directory = ut.io.get_path(simulation_directory)
        if group_directory is None or len(group_directory) == 0:
            group_directory = self.group_directory
        else:
            group_directory = ut.io.get_path(group_directory)

        # read list of all snapshots
        Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        if snapshot_values == 'all' or snapshot_values is None or len(snapshot_values) == 0:
            # read all snapshots
            snapshot_indices = Snapshot['index']
        else:
            # get snapshot index[s] corresponding to input snapshot values
            snapshot_indices = Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_values)

        args_list = []
        for snapshot_index in snapshot_indices:
            args_list.append(
                (
                    species_name,
                    linking_length,
                    particle_number_min,
                    property_select,
                    snapshot_index,
                    simulation_directory,
                    group_directory,
                    verbose,
                )
            )

        ut.io.run_in_parallel(
            self._generate_write_group_catalog, args_list, proc_number=proc_number, verbose=verbose,
        )

    def _generate_write_group_catalog(
        self,
        species_name,
        linking_length,
        particle_number_min,
        property_select,
        snapshot_index,
        simulation_directory,
        group_directory,
        verbose,
    ):
        '''
        Utility function.

        Parameters
        ----------
        species_name : str
            name of particle species to generate groups of
        linking_length : float
            FoF linking length [pc physical]
        particle_number_min : int
            minimum number of member particles to keep a group
        property_select : dict
            properties to select particles on: names as keys and limits as values
        snapshot_index : int
            index of snapshot
        simulation_directory : string
            base directory of simulation
        group_directory : str
            directory (within a simulation_directory) of group catalog files
        verbose : bool
            whether to print each property read/written
        '''
        from gizmo_analysis import gizmo_io

        if np.isscalar(species_name):
            species_names = [species_name]
        else:
            species_names = species_name

        # ensure that read star particles, to define host galaxy
        if 'star' not in species_names:
            species_names.append('star')

        # read particles
        part = gizmo_io.Read.read_snapshots(
            species_names,
            'index',
            snapshot_index,
            simulation_directory,
            properties=self.particle_properties_read,
            assign_hosts=True,
            assign_hosts_rotation=True,
            check_properties=False,
        )

        if (
            part is None
            or species_name not in part
            or self.property_name_default not in part[species_name]
            or len(part[species_name][self.property_name_default]) == 0
        ):
            self.say(f'! no snapshot or no {species_name} particles at snapshot {snapshot_index}')
            return

        # generate FoF group catalog of species
        grp = self.generate_group_catalog(
            part, species_name, linking_length, particle_number_min, property_select,
        )

        if grp is None:
            return

        # write group catalog to HDF5 file
        self._io_group_catalog(
            grp=grp,
            simulation_directory=simulation_directory,
            group_directory=group_directory,
            verbose=verbose,
        )

    def _get_group_file_names_and_indices(
        self, species_name, linking_length, simulation_and_group_directory, snapshot_indices
    ):
        '''
        Get name[s] and snapshot index[s] of group catalog file[s].

        Parameters
        ----------
        species_name : str
            name of particle species: 'gas', 'star', 'dark'
        linking_length : float
            FoF linking length [pc physical]
        simulation_and_group_directory : string
            directory (including path of base simulation_directory) of group catalog files
        snapshot_indices : int or array thereof
            index of snapshot[s]

        Returns
        -------
        path_file_names : list
            path + name[s] of group catalog file[s]
        file_indices : list
            snapshot index[s] of group catalog file[s]
        '''
        file_name_base = self.file_name_base.replace('{:03d}', '*.hdf5')
        file_name_base = file_name_base.format(species_name, linking_length)
        directory = ut.io.get_path(simulation_and_group_directory)

        # get names and indices of all group catalog files that match name base
        path_file_names_all, file_indices_all = ut.io.get_file_names(
            directory + file_name_base, int, verbose=False
        )

        if snapshot_indices is not None:
            path_file_names = []
            file_indices = []
            for file_i, file_index in enumerate(file_indices_all):
                if file_index in snapshot_indices:
                    path_file_names.append(path_file_names_all[file_i])
                    file_indices.append(file_index)

            if np.isscalar(snapshot_indices):
                snapshot_indices = [snapshot_indices]
            if len(snapshot_indices) > 1 and len(snapshot_indices) != len(path_file_names):
                self.say(
                    '! input {} snapshot indices but found only {} group catalog files'.format(
                        len(snapshot_indices), len(path_file_names)
                    )
                )
        else:
            # return all file names and snapshot indices that found
            path_file_names = path_file_names_all
            file_indices = file_indices_all

        if len(path_file_names) == 0:
            self.say('! cannot find group catalog files in:  {}'.format(directory.lstrip('./')))

        return path_file_names, file_indices


# --------------------------------------------------------------------------------------------------
# analysis and plotting
# --------------------------------------------------------------------------------------------------
def print_properties(grp, grp_indices, properties=None, digits=3):
    '''
    Print useful properties of groups[s].

    Parameters
    ----------
    grp : dict
        catalog of groups
    grp_indices : int or array
        index[s] of groups[s]
    properties : str or list
        name[s] of properties to print
    digits : int
        number of digits after period
    '''
    Say = ut.io.SayClass(print_properties)

    grp_indices = ut.array.arrayize(grp_indices)

    if properties:
        # print input properties
        for gi in grp_indices:
            print(f'group index = {gi}')
            for prop_name in properties:
                string = ut.io.get_string_from_numbers(grp[prop_name][gi], digits)
                print(f'{prop_name} = {string}')
            print()
    else:
        # print default properties
        for gi in grp_indices:
            Say.say('host distance = {:.1f} kpc'.format(grp.prop('host.distance.total', gi)))
            print()

            Say.say('group:')
            Say.say('* M = {} Msun'.format(ut.io.get_string_from_numbers(grp.prop('mass', gi), 2)))
            Say.say('* N_part = {:d}'.format(grp.prop('number', gi)))
            Say.say(
                '* V_std = {} km/s'.format(
                    ut.io.get_string_from_numbers(grp.prop('vel.std', gi), 1)
                )
            )
            Say.say(
                '* R_max = {} kpc'.format(
                    ut.io.get_string_from_numbers(grp.prop('radius.100', gi), 1)
                )
            )
            Say.say(
                '* R_50 = {}, R_90 = {} kpc'.format(
                    ut.io.get_string_from_numbers(grp.prop('radius.50', gi), 2),
                    ut.io.get_string_from_numbers(grp.prop('radius.90', gi), 2),
                )
            )
            print()


# --------------------------------------------------------------------------------------------------
# number of groups versus mass (or other property) or distance
# --------------------------------------------------------------------------------------------------
def plot_number_v_mass(
    grps=None,
    mass_name='mass',
    mass_limits=[3e4, 1e8],
    mass_width=0.1,
    mass_log_scale=True,
    host_distance_limitss=[[2, 20]],
    grp_indicess=None,
    number_kind='number.cum',
    number_limits=[0.8, None],
    number_log_scale=True,
    include_above_limits=True,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot number (cumulative or differential) of groups v mass_name.

    Parameters
    ----------
    grps : dict or list
        catalog[s] of groups at a snapshot
    mass_name : str or list
        mass kind[s] to plot
    mass_limits : list
        min and max limits for mass_name
    mass_width : float
        width of mass_name bin
    mass_log_scale : bool
        whether to use logarithmic scaling for mass_name bins
    host_distance_limitss : list or list of lists
        min and max limits of distance to host [kpc physical]
    grp_indicess : array or list of arrays
        group indices to plot
    number_kind : str
        mass function kind to plot: 'number',  'number.dif', 'number.cum'
    number_limits : list
        min and max limits to impose on y-axis
    number_log_scale : bool
        whether to use logarithmic scaling for y axis
    include_above_limits : bool
        whether to include mass_name values above limits for cumulative
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        index of figure for matplotlib
    '''
    if isinstance(grps, dict):
        grps = [grps]

    if host_distance_limitss is not None:
        host_distance_limitss = np.array(host_distance_limitss)
        if np.ndim(host_distance_limitss) == 1:
            host_distance_limitss = np.array([host_distance_limitss])
        host_distance_bin_number = host_distance_limitss.shape[0]
    else:
        host_distance_bin_number = 1

    if not isinstance(grp_indicess, list):
        grp_indicess = [grp_indicess for _ in grps]

    assert number_kind in ['number', 'number.dif', 'number.cum']

    MassBin = ut.binning.BinClass(
        mass_limits, mass_width, include_max=True, log_scale=mass_log_scale
    )

    grp_number_values, grp_number_uncs = np.zeros(
        [2, len(grps), host_distance_bin_number, MassBin.number]
    )

    # get counts for groups
    for grp_i, grp in enumerate(grps):

        if grp_indicess[grp_i] is None or len(grp_indicess[grp_i]) == 0:
            grp_indices = ut.array.get_arange(grp.prop(mass_name))
        else:
            grp_indices = grp_indices[grp_i]

        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            gis_d = grp.get_indices(
                host_distance_limits=host_distance_limits, prior_indices=grp_indices,
            )

            if len(gis_d) > 0:
                grp_number_d = MassBin.get_distribution(
                    grp.prop(mass_name, gis_d), include_above_limits=include_above_limits
                )
                grp_number_values[grp_i, dist_i] = grp_number_d[number_kind]
                grp_number_uncs[grp_i, dist_i] = grp_number_d[number_kind + '.err']

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    mass_funcs_all = []
    if grp_number_values.size:
        mass_funcs_all.append(grp_number_values)

    ut.plot.set_axes_scaling_limits(
        subplot, mass_log_scale, mass_limits, None, number_log_scale, number_limits, mass_funcs_all
    )

    # increase minor ticks along y-axis
    if not number_log_scale:
        subplot.yaxis.set_minor_locator(plt.MultipleLocator(2))

    if not mass_log_scale:
        subplot.xaxis.set_minor_locator(plt.MultipleLocator(1))

    # set x-axis label
    axis_x_label = ut.plot.Label.get_label(mass_name)
    subplot.set_xlabel(axis_x_label)

    # set y-axis label
    mass_label = ut.plot.Label.get_label(mass_name, get_units=False).strip('$')
    if 'dif' in number_kind:
        axis_y_label = f'${{\\rm d}}n / {{\\rm d}}log({mass_label})$'
    elif 'cum' in number_kind:
        axis_y_label = f'$N(> {mass_label})$'
    else:
        axis_y_label = f'$N({mass_label})$'
    subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(len(grps))
    line_styles = ut.plot.get_line_styles(host_distance_bin_number)

    x_values = MassBin.get_bin_values(number_kind)

    # plot groups
    for grp_i, grp in enumerate(grps):
        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            linewidth = 3.0
            alpha = 0.9
            color = colors[grp_i]

            label = grp.info['simulation.name']

            # ensure n = 1 is clear on log plot
            y_values = grp_number_values[grp_i, dist_i]
            if number_log_scale:
                y_values = np.clip(y_values, 0.5, np.Inf)

            subplot.plot(
                x_values,
                y_values,
                # grp_number_uncs[grp_i],
                color=color,
                linestyle=line_styles[dist_i],
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

    if len(grps) > 1 or len(host_distance_limitss) > 1:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    if plot_file_name is True or plot_file_name == '':
        redshift_label = ''
        if len(grps) > 0:
            redshift_label = ut.plot.get_time_label('redshift', grps[0].snapshot)
        plot_file_name = f'{number_kind}_v_{mass_name}{redshift_label}'
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_number_v_distance(
    grps=None,
    mass_name='mass',
    mass_limitss=[[]],
    distance_limits=[1, 1000],
    distance_bin_width=0.1,
    distance_log_scale=True,
    grp_indicess=None,
    number_kind='sum',
    number_limits=None,
    number_log_scale=True,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot number (cumulative or differential) of groups v host distance.

    Parameters
    ----------
    grps : dict or list
        catalog[s] of groups at a snapshot
    mass_name : str
        group mass kind to plot
    mass_limitss : list or list of lists
        min and max limits of mass_name
    distance_limits : list
        min and max distance from host [kpc physical]
    distance_bin_width : float
        width of distance bin
    distance_log_scale : bool
        whether to use logarithmic scaling for distance bins
    grp_indicess : array or list of arrays
        prior indices of groups
    number_kind : str
         number kind to plot: 'sum', 'sum.cum', 'fraction', 'fraction.cum', 'density'
    number_limits : list
        min and max limits to impose on y-axis
    number_log_scale : bool
        whether to use logarithmic scaling for y axis
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        index of figure for matplotlib
    '''
    dimension_number = 3

    if isinstance(grps, dict):
        grps = [grps]

    if mass_limitss is not None:
        mass_limitss = np.array(mass_limitss)
        if np.ndim(mass_limitss) == 1:
            mass_limitss = np.array([mass_limitss])
        mass_number = mass_limitss.shape[0]

    if not isinstance(grp_indicess, list):
        grp_indicess = [grp_indicess for _ in grps]

    DistanceBin = ut.binning.DistanceBinClass(
        distance_limits,
        distance_bin_width,
        include_max=True,
        log_scale=distance_log_scale,
        dimension_number=dimension_number,
    )

    grp_number = {}

    for grp_i, grp in enumerate(grps):
        if grp_indicess[grp_i] is None or len(grp_indicess[grp_i]) == 0:
            grp_indices = ut.array.get_arange(grp.prop(mass_name))
        else:
            grp_indices = grp_indicess[grp_i]

        grp_number_h = {}

        for _m_i, mass_limits in enumerate(mass_limitss):
            gis_m = ut.array.get_indices(grp.prop(mass_name), mass_limits, grp_indices)
            grp_number_m = DistanceBin.get_sum_profile(grp.prop('host.distance.total', gis_m))
            ut.array.append_dictionary(grp_number_h, grp_number_m)
        ut.array.append_dictionary(grp_number, grp_number_h)
    ut.array.arrayize_dictionary(grp_number)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot,
        distance_log_scale,
        distance_limits,
        None,
        number_log_scale,
        number_limits,
        grp_number[number_kind],
    )

    # if distance_log_scale:
    #    subplot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))

    species_name = grps[0].info['fof.species']
    if 'gas' in species_name:
        object_kind = 'GMC'
    elif 'star' in species_name:
        object_kind = 'cluster'

    subplot.set_xlabel('host distance $\\left[ {\\rm kpc} \\right]$')

    if '.cum' in number_kind:
        axis_y_label = f'$N_{{\\rm {object_kind}}}(< d)$'
    else:
        if distance_log_scale:
            axis_y_label = '${\\rm d}n/{\\rm d}log(d) \, \\left[ {\\rm kpc^{-3}} \\right]$'
        else:
            axis_y_label = '${\\rm d}n/{\\rm d}d \, \\left[ {\\rm kpc^{-2}} \\right]$'
    subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(len(grps))
    line_styles = ut.plot.get_line_styles(mass_number)

    distance_kind = 'distance.mid'
    if '.cum' in number_kind:
        distance_kind = 'distance.cum'

    if grp is not None and len(grps) > 0:
        for grp_i, grp in enumerate(grps):
            for m_i, _mass_limits in enumerate(mass_limitss):
                linewidth = 3.0
                alpha = 0.9
                color = colors[grp_i]

                label = grp.info['simulation.name']
                y_values = grp_number[number_kind][grp_i, m_i]
                # ensure n = 1 is clear on log plot
                if 'sum' in number_kind and number_log_scale:
                    y_values = np.clip(y_values, 0.5, np.Inf)

                masks = np.where(y_values > -1)[0]
                subplot.plot(
                    grp_number[distance_kind][grp_i, m_i][masks],
                    y_values[masks],
                    color=color,
                    linestyle=line_styles[m_i],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )

    if len(grps) > 1:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    if plot_file_name is True or plot_file_name == '':
        redshift_label = ''
        if len(grps) > 0:
            redshift_label = ut.plot.get_time_label('redshift', grps[0].snapshot)
        plot_file_name = f'{species_name}.group.number.{number_kind}_v_distance{redshift_label}'
    ut.plot.parse_output(plot_file_name, plot_directory)


# --------------------------------------------------------------------------------------------------
# group property versus property or distance
# --------------------------------------------------------------------------------------------------
def plot_property_v_property(
    grps=None,
    x_property_name='mass',
    x_property_limits=[],
    x_property_log_scale=True,
    y_property_name='vel.std',
    y_property_limits=[],
    y_property_log_scale=True,
    host_distance_limitss=None,
    grp_indicess=None,
    plot_kind='point',
    histogram_bin_number=100,
    statistic_bin_width=0.1,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot property v property.

    Parameters
    ----------
    grps : dict
        catalog[s] of groups at snapshot
    x_property_name : str
        name of property for x-axis
    x_property_limits : list
        min and max limits to impose on x_property_name
    x_property_log_scale : bool
        whether to use logarithmic scaling for x axis
    y_property_name : str
        name of property for y-axis
    y_property_limits : list
        min and max limits to impose on y_property_name
    y_property_log_scale : bool
        whether to use logarithmic scaling for y axis
    host_distance_limitss : list
        min and max limits for distance from galaxy
    grp_indicess : array or list of arrays
        prior indices of groups to select
    plot_kind : str
        what kind of plot to make (can include multiple): 'point', 'histogram', 'statistic'
    histogram_bin_number : int
        number of bins along each axis (if histogram)
    statistic_bin_width : float
        width of x_property_name bin for computing statistics
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        ndex of figure for matplotlib
    '''
    Say = ut.io.SayClass(plot_property_v_property)

    assert 'point' in plot_kind or 'histogram' in plot_kind or 'statistic' in plot_kind

    if isinstance(grps, dict):
        grps = [grps]

    if host_distance_limitss is not None:
        host_distance_limitss = np.array(host_distance_limitss)
        if np.ndim(host_distance_limitss) == 1:
            host_distance_limitss = np.array([host_distance_limitss])
        host_distance_bin_number = host_distance_limitss.shape[0]
    else:
        host_distance_bin_number = 1
        host_distance_limitss = [None]

    if not isinstance(grp_indicess, list):
        grp_indicess = [grp_indicess for _ in grps]

    if 'statistic' in plot_kind:
        Bin = ut.binning.BinClass(
            x_property_limits, statistic_bin_width, log_scale=x_property_log_scale
        )

    x_property_values = []
    y_property_values = []
    y_property_statistics = []

    for grp_i, grp in enumerate(grps):
        gis = grp_indicess[grp_i]
        if gis is None:
            gis = ut.array.get_arange(grp['mass'])

        x_props_g = []
        y_props_g = []
        y_stats_g = []

        for host_distance_limits in host_distance_limitss:
            if host_distance_limits is not None and len(host_distance_limits) > 0:
                gis_d = ut.array.get_indices(
                    grp.prop('host.distance.total'), host_distance_limits, gis
                )
            else:
                gis_d = gis

            x_props_d = grp.prop(x_property_name, gis_d)
            y_props_d = grp.prop(y_property_name, gis_d)

            Say.say(
                '{} range = [{:.3e}, {:.3e}], med = {:.3e}'.format(
                    x_property_name, x_props_d.min(), x_props_d.max(), np.median(x_props_d),
                )
            )
            Say.say(
                '{} range = [{:.3e}, {:.3e}], med = {:.3e}'.format(
                    y_property_name, y_props_d.min(), y_props_d.max(), np.median(y_props_d),
                )
            )

            if x_property_limits:
                indices = ut.array.get_indices(x_props_d, x_property_limits)
                x_props_d = x_props_d[indices]
                y_props_d = y_props_d[indices]

            if 'statistic' in plot_kind and len(x_props_d) > 0:
                y_stat_d = Bin.get_statistics_of_array(x_props_d, y_props_d)

            if y_property_limits:
                indices = ut.array.get_indices(y_props_d, y_property_limits)
                x_props_d = x_props_d[indices]
                y_props_d = y_props_d[indices]

            if len(y_props_d) == 0:
                Say.say('! no groups within limits')

            x_props_g.append(x_props_d)
            y_props_g.append(y_props_d)
            y_stats_g.append(y_stat_d)

        x_property_values.append(x_props_g)
        y_property_values.append(y_props_g)
        y_property_statistics.append(y_stats_g)

    x_property_values = np.array(x_property_values, dtype=object)
    y_property_values = np.array(y_property_values, dtype=object)

    if len(grps) > 1:
        colors = ut.plot.get_colors(len(grps))
    else:
        colors = ut.plot.get_colors(max(host_distance_bin_number, 2))

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)  # , figure_sizes=[12.8, 9.6])

    axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot,
        x_property_log_scale,
        x_property_limits,
        x_property_values,
        y_property_log_scale,
        y_property_limits,
        y_property_values,
    )

    axis_x_label = ut.plot.Label.get_label(x_property_name)
    subplot.set_xlabel(axis_x_label)
    axis_y_label = ut.plot.Label.get_label(y_property_name)
    subplot.set_ylabel(axis_y_label)  # , fontsize=30)

    label = None

    if 'point' in plot_kind:
        # plot as individual points

        for grp_i, grp in enumerate(grps):
            if len(grps) > 1:
                label = grp.info['simulation.name']
                color = colors[grp_i]

            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if (
                    len(grps) == 1
                    and host_distance_limits is not None
                    and len(host_distance_limits) > 1
                ):
                    label = ut.plot.Label.get_label('host.distance.total', host_distance_limits)
                    color = colors[dist_i]

                subplot.plot(
                    x_property_values[grp_i, dist_i],
                    y_property_values[grp_i, dist_i],
                    marker='.',
                    color=color,
                    markersize=3,
                    alpha=0.8,
                    label=label,
                )

    elif 'histogram' in plot_kind:
        # plot histogram
        for grp_i, grp in enumerate(grps):
            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if x_property_log_scale:
                    x_property_values[grp_i, dist_i] = ut.math.get_log(
                        x_property_values[grp_i, dist_i]
                    )

                if y_property_log_scale:
                    y_property_values[grp_i, dist_i] = ut.math.get_log(
                        y_property_values[grp_i, dist_i]
                    )

                if host_distance_limits is not None and len(host_distance_limits) > 0:
                    label = ut.plot.Label.get_label('host.distance.total', host_distance_limits)

            valuess, _xs, _ys = np.histogram2d(
                x_property_values[grp_i, dist_i],
                y_property_values[grp_i, dist_i],
                histogram_bin_number,
                range=[axis_x_limits, axis_y_limits],
            )

            subplot.imshow(
                valuess.transpose(),
                # norm=LogNorm(),
                cmap=plt.cm.YlOrBr,  # pylint: disable=no-member
                aspect='auto',
                # interpolation='nearest',
                interpolation='none',
                extent=(axis_x_limits[0], axis_x_limits[1], axis_y_limits[0], axis_y_limits[1]),
                vmin=np.min(valuess),
                vmax=np.max(valuess),
                label=label,
            )

        # plt.colorbar()

    if 'statistic' in plot_kind:
        # plot statistics
        statistic_name = 'median'

        for grp_i, grp in enumerate(grps):
            if len(grps) > 1:
                label = grp.info['simulation.name']
                color = colors[grp_i]

            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if (
                    len(grps) == 1
                    and host_distance_limits is not None
                    and len(host_distance_limits) > 1
                ):
                    label = ut.plot.Label.get_label('host.distance.total', host_distance_limits)
                    color = colors[dist_i]

                masks = y_property_statistics[grp_i][dist_i][statistic_name] > 0

                subplot.fill_between(
                    y_property_statistics[grp_i][dist_i]['bin.mid'][masks],
                    y_property_statistics[grp_i][dist_i]['percent.16'][masks],
                    y_property_statistics[grp_i][dist_i]['percent.84'][masks],
                    color=color,
                    edgecolor=None,
                    alpha=0.4,
                )

                subplot.plot(
                    y_property_statistics[grp_i][dist_i]['bin.mid'][masks],
                    y_property_statistics[grp_i][dist_i][statistic_name][masks],
                    color=color,
                    linestyle='-',
                    alpha=0.5,
                    label=label,
                )

    if label is not None:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = y_property_name + '_v_' + x_property_name
        if grps is not None and len(grps) > 0:
            plot_file_name += ut.plot.get_time_label('redshift', grps[0].snapshot)
        else:
            plot_file_name = plot_file_name.replace('.part', '')
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_property_v_distance(
    grps=None,
    mass_name='mass',
    mass_limitss=[[]],
    distance_limits=[0, 300],
    distance_bin_width=1,
    distance_log_scale=False,
    property_name='host.velocity.tan',
    property_statistic='median',
    property_limits=None,
    property_log_scale=False,
    grp_indicess=None,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot property v distance, in bins of mass_name.

    Parameters
    ----------
    grps : dict or list
        catalog[s] of groups at snapshot
    mass_name : str
        mass kind to plot
    mass_limitss : list or list of lists
        min and max limits of group mass
    distance_limits : list
        min and max distance from host [kpc physical]
    distance_bin_width : float
        width of distance bin
    distance_log_scale : bool
        whether to use logarithmic scaling for distance bins
    property_name : str
        name of property to plot
    statistic : str
        statistic of property to plot
    property_limits : list
        min and max limits to impose on y-axis
    property_log_scale : bool
        whether to use logarithmic scaling for y axis
    grp_indicess : array or list of arrays
        prior indices of groups to include
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        index of figure for matplotlib
    '''
    dimension_number = 3

    if isinstance(grps, dict):
        grps = [grps]

    if mass_limitss is not None:
        mass_limitss = np.array(mass_limitss)
        if np.ndim(mass_limitss) == 1:
            mass_limitss = np.array([mass_limitss])
        mass_number = mass_limitss.shape[0]

    if not isinstance(grp_indicess, list):
        grp_indicess = [grp_indicess for _ in grps]

    DistanceBin = ut.binning.DistanceBinClass(
        distance_limits,
        distance_bin_width,
        include_max=True,
        log_scale=distance_log_scale,
        dimension_number=dimension_number,
    )

    grp_stat = {}

    # get statistics
    if grps is not None and len(grps) > 0:
        for grp_i, grp in enumerate(grps):
            if grp_indicess[grp_i] is None or len(grp_indicess[grp_i]) == 0:
                grp_indices = ut.array.get_arange(grp.prop(mass_name))
            else:
                grp_indices = grp_indicess[grp_i]

            grp_stat_h = {}

            for _m_i, mass_limits in enumerate(mass_limitss):
                gis_m = ut.array.get_indices(grp.prop(mass_name), mass_limits, grp_indices)
                grp_stat_m = DistanceBin.get_statistics_profile(
                    grp.prop('host.distance.total', gis_m), grp.prop(property_name, gis_m)
                )
                ut.array.append_dictionary(grp_stat_h, grp_stat_m)
            ut.array.append_dictionary(grp_stat, grp_stat_h)
        # ut.array.arrayize_dictionary(grp_stat)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot, distance_log_scale, distance_limits, None, property_log_scale, property_limits
    )

    subplot.set_xlabel('distance $\\left[ {\\rm kpc} \\right]$')
    subplot.set_ylabel(property_name)
    # subplot.set_ylabel('$V_{tan} / ( \sqrt{2} V_{rad} )$')

    colors = ut.plot.get_colors(len(grps))
    line_styles = ut.plot.get_line_styles(mass_number)

    # plot
    if grps is not None and len(grps) > 0:
        for grp_i, grp in enumerate(grps):
            for m_i, _mass_limits in enumerate(mass_limitss):
                linewidth = 3.0
                alpha = 0.9
                color = colors[grp_i]

                label = grp.info['simulation.name']

                subplot.plot(
                    grp_stat['distance.mid'][grp_i][m_i],
                    grp_stat[property_statistic][grp_i][m_i],
                    color=color,
                    linestyle=line_styles[m_i],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )

    if len(grps) > 1:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    if plot_file_name is True or plot_file_name == '':
        redshift_label = ''
        if len(grps) > 0:
            redshift_label = ut.plot.get_time_label('redshift', grps[0].snapshot)
        plot_file_name = f'{property_name}.{property_statistic}_v_distance{redshift_label}'
    ut.plot.parse_output(plot_file_name, plot_directory)
