#!/usr/bin/env python3

'''
Post-process elemental abundances using age-tracer passive scalars in FIRE-3 simulations.

In addition to elemental abundances tracked natively (see gizmo_star.py), FIRE-3 simulations can
compute elemental enrichment via a stored array of weights in bins of age, as stars at a given age
deposit weights (into their corresponding age bin) into neighboring gas elements.

This post-processing requires one to compute the weightings for each age bin for each element.
This consists of a table which contains the total amount of mass of each element produced during
each time bin. You can construct this here using a default rate + yield model, or you can define
your own custom rate and/or yield model, via an object that accepts an element name and time as
parameters and returns the instantaneous mass loss rate of that element at that time.

@author:
    Andrew Emerick <aemerick11@gmail.com>
    Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    time [Gyr]
    elemental abundance [mass fraction]
'''

import os
import copy
import numpy as np
from scipy import integrate

import utilities as ut
from . import gizmo_star


# --------------------------------------------------------------------------------------------------
# age tracers
# --------------------------------------------------------------------------------------------------
class ElementAgetracerClass(dict, ut.io.SayClass):
    '''
    Dictionary class to store information about age-tracer bins for post-processing elemental
    abundances.
    '''

    def __init__(self):
        self.info = {}
        self.info['has.element.agetracer'] = 0  # assume off initially

        self._yield_table = None
        self._postprocess_elements = None

        self._initial_abundances = {}  # dictionary to store elemental abundances at the IC

        # left edges of stellar age bins plus right edge of last bin [Myr]
        self.age_bins = None

    def set_initial_abundances(self, initial_abundances):
        '''
        For use with post-processing of elemental abundances with element age tracer model.
        Sets the initial abundances to add to the returned age tracer values.

        Parameters
        ----------
        initial_abundances : dict
            Dictionary with keys matching element names and values corresponding
            to the initial mass fractions for that element. Excluded elements
            will be given an initial abundance of zero by default.
        '''

        if self._postprocess_elements is None:
            print('yield table (set_yield_table) must be set first')
            raise RuntimeError

        # set initial values in both element name and symbol for versatility
        for e in self._postprocess_elements:
            self._initial_abundances[e] = 0.0
            if e != 'metals' and 'rprocess' not in e:
                self._initial_abundances[ut.constant.element_map_name_symbol[e]] = 0.0

        # set actual values in both name and symbol
        for e in initial_abundances:
            self._initial_abundances[e] = initial_abundances[e]
            if e != 'metals' and 'rprocess' not in e:
                self._initial_abundances[
                    ut.constant.element_map_name_symbol[e]
                ] = initial_abundances[e]

    def set_yield_table(self, yield_table, elements):
        '''
        For post-processing of elemental abundances with age-tracer model.
        Set the yield table to use in generating the fields along with some error checking.

        Parameters
        -----------
        yield_table : ndarray, float
            A 2-d numpy array containing the yield table scalings to map the age tracer field
            values to individual element yields in each bin. Each element should contain the
            mass produced of a given element in each time bin per solar mass of star formation
            (see gizmo_agetracers.py for more information). First axis should be the number of
            age bins and second the number of elements.

        elements :  str or list
            name[s] of element[s] to get from the yield table
        '''

        if np.shape(yield_table)[0] != self.info['age.bin.number']:
            raise ValueError(
                'yield table provided has incorrect dimensions.'
                + ' first dimension should be {} not {}'.format(
                    self.info['age.bin.number'], np.shape(yield_table)[0],
                )
            )

        if np.shape(yield_table)[1] != len(elements):
            raise ValueError(
                'yield table provided has incorrect dimensions.'
                + ' second dimension ({}) should match provided elements length ({})'.format(
                    np.shape(yield_table)[1], len(elements)
                )
            )

        self._yield_table = copy.deepcopy(yield_table)
        self._postprocess_elements = copy.deepcopy(elements)

    @property
    def yield_table(self):
        '''
        Return a copy of the internal yield table for age tracer post-processing, such that this
        table is read-only. You can change it using 'set_yield_table'.
        '''
        return self._yield_table.copy()

    def read_agetracer_times(self, directory='.', file_name='age_bins.txt'):
        '''
        Within input directory, search for and read stellar age tracer field bin times
        (if custom sized age tracer fields used). Store as an array.

        Parameters
        -----------
        directory : str
            directory where snapshot times/scale-factor file is
        '''
        directory = ut.io.get_path(directory)

        self.read_age_bin_info(directory=directory)

        self.generate_age_bins(directory, file_name)

    def read_age_bin_info(self, directory='.'):
        '''
        Check for GIZMO_config.h in the simulation directory, or the gizmo/ sub-directory, or
        for gizmo.out in the simulation directory. Need one of these to read the compile-time
        flags that determine the number of age-tracer time bins used in the simulation.

        Parameters
        -----------
        directory : str
            top-level simulation directory
        '''
        directory = ut.io.get_path(directory)

        possible_file_names = ['GIZMO_config.h', 'gizmo_config.h', 'gizmo.out', 'gizmo.out.txt']
        possible_path_names = [directory, ut.io.get_path(directory + 'gizmo')]

        for file_name in possible_file_names:
            for path_name in possible_path_names:
                if os.path.exists(path_name + file_name):
                    path_file_name = path_name + file_name
                    break
        else:
            print(f'cannot read gizmo_config.h or gizmo.out* in {directory} or {directory}/gizmo')
            print('cannot assign element age-tracer information')
            self.info['has.element.agetracer'] = 0
            self.info['has.element.agetracer.custom'] = 0
            return

        if 'GIZMO_config.h' in path_file_name:
            delimiter = ' '
        else:
            delimiter = '='

        self.info['has.element.agetracer'] = 0
        self.info['has.element.agetracer.custom'] = 0

        self.info['use.log.age.bins'] = True  # assume by default
        count = 0

        self.info['element.index.min'] = -1
        for line in open(path_file_name, 'r'):

            # ----------------------  WARNING: ---------------------------------
            #          If the number of age tracers is ever controlled outside
            #          of this flag (e.g. some other flag turns on a default
            #          value), this WILL need to be changed to account for that.
            #

            if 'GALSF_FB_FIRE_AGE_TRACERS' + delimiter in line:
                self.info['age.bin.number'] = int(line.strip('\n').split(delimiter)[-1])
                self.info['has.element.agetracer'] = 1

            if 'GALSF_FB_FIRE_AGE_TRACERS_CUSTOM' in line:
                self.info['has.element.agetracer.custom'] = 1
                self.info['use.log.age.bins'] = False

            # ----------------------  WARNING: ---------------------------------
            # if number of default elements and r process elements
            # changes from 11 (10 + metallicity) and 4 respecitvely, then
            # this cannot be hard-coded like this any longer. Will need some trickier
            # if statements too in this case to ensure backwards compatability
            #
            # hard-coded values:  11 = number of metals tracked
            #                      4 = number of rprocess fields
            if 'FIRE_PHYSICS_DEFAULTS' in line:
                if self.info['element.index.min'] == -1:
                    self.info['element.index.min'] = (
                        11 + 4
                    )  # first age tracer is field Metallicity_XX
                else:
                    self.info['element.index.min'] = self.info['element.index.min'] + 11

            # if this is specified, then more than 4 are being tracked.
            # check this.
            if 'GALSF_FB_FIRE_RPROCESS' in line:
                numr = int(line.split(delimiter)[-1])
                if self.info['element.index.min'] == -1:
                    self.info['element.index.min'] = 0 + numr
                else:
                    self.info['element.index.min'] = self.info['element.index.min'] - 4 + numr

            if count > 400:  # arbitrary limit
                break
            count = count + 1

        self.info['element.index.max'] = self.info['element.index.min']
        if self.info['has.element.agetracer']:
            self.info['element.index.max'] += self.info['age.bin.number']

    def generate_age_bins(self, directory='.', file_name='age_bins.txt'):
        '''
        If element age-tracers are contained in the output file (as determined by
        determine_tracer_info), generate or read the age-tracer bins used by the simulation.
        This requires either the presence of params.txt-usedvalues,
        parameters-usedvalues, or gizmo.out in the top-level directory or
        the output subdirectory if used log-spaced bins. Otherwise, this requires
        the presence of the 'age-bins.txt' input file to read in custom-spaced bins.

        Parameters
        ----------
        directory : str
            top-level simulation directory

        file_name : str
            only used if custom space age tracers are used. Name of bin file, assumed to be a
            single column of bin left edges plus the right edge of last bin (number of lines
            should be number of tracers + 1)
        '''

        if self.info['has.element.agetracer'] == 0:
            self.age_bins = None

            return

        if self.info['use.log.age.bins']:
            # ----------------------  WARNING: ---------------------------------
            # this wil need to be changed if the Gizmo parameters are changed
            # from AgeTracerBinStart and AgeTracerBinEnd
            #
            # need to grab this from parameter file or gizmo.out
            path_file_name = ut.io.get_path(directory) + 'params.txt-usedvalues'
            if not os.path.exists(path_file_name):
                path_file_name = ut.io.get_path(directory) + 'gizmo.out'

                if not os.path.exists(path_file_name):
                    path_file_name = ut.io.get_path(directory + 'output') + 'parameters-usedvalues'
                    if not os.path.exists(path_file_name):

                        raise OSError(
                            f'cannot find "params.txt-usedvalues" or "gizmo.out" in {directory}'
                        )

            self.say(
                '* reading element age-tracer bins from :  {}\n'.format(path_file_name.strip('./')),
            )

            for line in open(path_file_name):
                if 'AgeTracerBinStart' in line:
                    self.info['age.min'] = float(line.split(' ')[-1])
                if 'AgeTracerBinEnd' in line:
                    self.info['age.max'] = float(line.split(' ')[-1])

            age_min = np.log10(self.info['age.min'])
            age_max = np.log10(self.info['age.max'])
            self.age_bins = np.logspace(age_min, age_max, self.info['age.bin.number'] + 1)
            self.age_bins[0] = 0  # ensure minimum bin age is 0

        else:
            self.read_age_bins(directory, file_name)

        if len(self.age_bins) > self.info['age.bin.number'] + 1:
            raise RuntimeError('number of age bins implies that there should be more age tracers')
        elif len(self.age_bins) < self.info['age.bin.number'] + 1:
            raise RuntimeError('number of age bins implies that there should be fewer age tracers')

    def read_age_bins(
        self, directory='.', file_name='age_bins.txt',
    ):
        '''
        Parameters
        ----------
        directory : str
            top-level simulation directory

        file_name : str
            Only used if custom space age tracers are used. Name of bin file, assumed to be a
            single column of bin left edges plus the right edge of last bin (number of lines
            should be number of tracers + 1). Default : 'age_bins.txt'
        '''

        if self.info['has.element.agetracer'] == 0:
            self.age_bins = None
            return

        try:
            path_file_name = ut.io.get_path(directory) + file_name
            self.age_bins = np.genfromtxt(path_file_name)
        except OSError:
            raise OSError(f'cannot find file of age tracer bins: {path_file_name}')


ElementAgetracer = ElementAgetracerClass()


# Functions and classes for generating post-processing yield tables


def construct_yield_table(yield_object, agebins, elements=None, integration_points=None):
    '''
    Construct a table of weights to post-process the age tracer fields
    with individual elemental abundances. Given a function that accepts
    elements and time as arguments, build this table by summing up the total
    yield for each element during each time bin. This integrates over the
    `yield` function in the passed `yield_object` (see below) for each time bin.

    Parameters
    -----------
    yield_object : obj
        An object with a required method 'yield' that is a function of 2 arguments:
        the first is time [Gyr] and the second is an element name (for example, 'oxygen','carbon').
        This function must return the instantaneous specific mass loss rate for that
        element [Msun / Gyr per Msun of star formation].

        This object also must have an attribute 'elements', which lists
        all elements generated by this yield model. If making your own object to generate a yields
        table, for convenience, one can refer to the 'element_*' dictionaries
        in the `utilities` package in `basic/constant.py`. If you will compute total metallicity,
        'metals' MUST be one of these elements.

        If `yield_object` contains the attribute 'integration_points', will pass these to the
        integrator (scipy.integrate.quad). Otherwise you can provide these as a separate argument.

    elements : list
        List of elements to generate for the table, if only generating a subset.
        If None, will use all possible elements in yield_object.elements.

    integration_points : (sequence of floats, ints)
        Points to pass to scipy.integrate.quad to be careful around.
        You also can pass these if 'integration_points' is an attribute of the yields_object.
        This argument overrides the yield_object attribute if provided.

    Returns
    -------
    yield_table : np.ndarray
        2-D array (N_tracer x N_elements) containing the weights for each element in each age-tracer
        time bin. Each value represents the mass  of each element produced during each time bin
        [Msun per Msun of star formation].
    '''

    # assume to generate this for all elements
    if elements is None:
        elements = yield_object.elements
    else:
        for e in elements:
            assert e in yield_object.elements

    yield_table = np.zeros((np.size(agebins) - 1, np.size(elements)))

    # grab points to be carful around for integration (if available)
    points = None
    if integration_points is not None:
        points = integration_points
    else:
        if hasattr(yield_object, 'integration_points'):
            if len(yield_object.integration_points) > 0:
                points = yield_object.integration_points

    # generate yield weights for each age bin
    for i in np.arange(np.size(agebins) - 1):
        if i == 0:  # ensure min time starts at 0
            t_min = 0.0
        else:
            t_min = agebins[i]

        t_max = agebins[i + 1]

        for j, e in enumerate(elements):
            yield_table[i][j] = integrate.quad(
                yield_object.yields, t_min, t_max, args=(e,), points=points
            )[0]

    return yield_table


class YieldsObject:
    '''
    .
    '''

    def __init__(self, name=''):
        self.name = name

        self.elements = []

        return

    def yields(self, t, element):
        pass


# --------------------------------------------------------------------------------------------------
# FIRE-2 Yield Class object for generating yield tables for post-processing via age-tracers
# --------------------------------------------------------------------------------------------------
class FIRE2YieldClass(YieldsObject):
    '''
    Object to use with the construct_yield_table method.
    This object provides the yields in the FIRE-2 model.
    This model uses some metallicity-dependent yields, determined by 2 paramters.
    '''

    def __init__(self, name='fire2', metallicity=1.0, scale_metallicity=True):
        '''
        Initialize object and pre-load some things for convenience.

        Parameters
        -----------
        name : str
            Optional name for this table
        metallicity : float
            metallicity (wrt Solar) for  progenitor metallicity dependent yields
        scale_metallicity : bool
            apply progenitor metallicity dependence (in approximate fashion)
        '''

        super().__init__(name)

        # not required. Specific parameters for this model
        self.model_parameters = {'metallicity': metallicity, 'scale.metallicity': scale_metallicity}

        # not required, but useful
        self.elements = [
            'metals',
            'helium',
            'carbon',
            'nitrogen',
            'oxygen',
            'neon',
            'magnesium',
            'silicon',
            'sulphur',
            'calcium',
            'iron',
        ]

        # not required
        # to use for metallicity dependent corrections on the yields
        # this just assumes that all yields come from stars with metallicities
        # and individual abundances scaled to the solar abundance pattern
        # this isn't accurate in practice but gives better agreement between
        # post-processed yields and native simulated yields in FIRE-2
        self._star_massfraction = {}
        for e in self.elements:
            self._star_massfraction[e] = metallicity * gizmo_star.sun_massfraction[e]

        # pre-load yields since they are constants in time.
        # in general, this probably cannot be done if they are time-varying
        # and would have to make separete function calls or something in
        # the yields method
        self.compute_yields()

        # ages [Gyr] to be careful around during integration
        self.integration_points = np.sort([3.401, 10.37, 37.53, 1, 50, 100, 1000, 14000]) / 1000

    def compute_yields(self, metallicity=None):
        '''
        This function is not necessary in a YieldsObject but exists here
        to generate the yields from different channels from the underlying
        model in gizmo_analysis in order keep the `yields` function
        cleaner and to avoid having to re-compute this every time the
        yields function is called.
        '''

        if metallicity is not None:
            self.model_parameters['metallicity'] = metallicity

        if self.model_parameters['metallicity'] is None:
            self.model_parameters['metallicity'] = 1.0  # default

        # Yields here is a dictionary with element names as kwargs
        # and yields (in Msun) as values
        self.snia_yield = gizmo_star.get_nucleosynthetic_yields(
            'supernova.ia',
            star_metallicity=self.model_parameters['metallicity'],
            star_massfraction=self._star_massfraction,
            normalize=False,
        )

        self.sncc_yield = gizmo_star.get_nucleosynthetic_yields(
            'supernova.cc',
            star_metallicity=self.model_parameters['metallicity'],
            star_massfraction=self._star_massfraction,
            normalize=False,
        )

        # wind yields do not have quantized rates. These are mass fraction
        self.wind_yield = gizmo_star.get_nucleosynthetic_yields(
            'wind',
            star_metallicity=self.model_parameters['metallicity'],
            star_massfraction=self._star_massfraction,
            normalize=False,
        )

    def yields(self, age, element_name):
        '''
        Return the total yields for all stellar processes.
        construct_yield_table requires this method.

        Parameters
        ----------
        age : float or array
            stellar age [Gyr] to compute instantaneous yield
        element_name : str
            name of element, must be in self.elements

        Returns
        -------
        y : float or np.ndarray
            total yields at a given time for desired element, in units of Msun / Gyr per Msun of
            star formation
        '''

        assert element_name in self.elements

        # get supernova Ia rate at input time [units of 1/Myr per Msun of stars formed]
        snia_rate = gizmo_star.SupernovaIa.get_rate(age * 1000, 'mannucci')

        # get core-collapse supernova rate at input time [units of 1/Myr per Msun of stars formed]
        sncc_rate = gizmo_star.SupernovaCC.get_rate(age * 1000)

        # get stellar wind rate at input time [units of M_sun / Myr per Msun of stars formed]
        # this is the only rate in FIRE-2 that depends on metallicity
        wind_rate = gizmo_star.StellarWind.get_rate(
            age * 1000, metallicity=self.model_parameters['metallicity']
        )

        y = (
            self.wind_yield[element_name] * wind_rate
            + self.snia_yield[element_name] * snia_rate
            + self.sncc_yield[element_name] * sncc_rate
        ) * 1000  # [M_sun / Gyr]

        return y


# ------------------------------------------------------------------------------
# NuGrid yield class object for generating yield tables for age-tracer
# postprocessing using the Sygma module
# ------------------------------------------------------------------------------

try:
    import sygma

    NuPyCEE_loaded = True
except ImportError:
    NuPyCEE_loaded = False


class NuGridYieldClass(YieldsObject):
    '''
    Object designed for use with the construct_yield_table method. This
    object proves the yields from the NuGrid collaboration using the
    Sygma interface. The NuPyCEE code must be instssssssalled (and its dependencies)
    in the python path for this object to work.

    https://nugrid.github.io/NuPyCEE/index.html
    '''

    def __init__(self, name='NuGrid', **kwargs):

        if not NuPyCEE_loaded:
            print(
                'Cannot load the NuPyCEE module Sygma. '
                + 'This is necessary to use the NuGrid yields.'
                + 'Make sure this is installed in your python path. '
                + 'For more info: https://nugrid.github.io/NuPyCEE/index.html'
            )

            raise RuntimeError

        super().__init__(name)

        # if elements is None
        # if isinstance(elements,str):
        #    elements = [elements]
        # self.elements = elements

        # if not 'metals' in elements:
        #    self.elements = ['metals'] + self.elements

        self.model_parameters = {}
        # set some defaults:
        self.model_parameters = {
            'dt': 1.0e5,  # first dt
            'special_timesteps': 1000,  # number of timesteps (using logspacing alg)
            'tend': 1.4e10,  # end time in years
            'mgal': 1.0,
        }  # galaxy / SSP mass in solar masses

        for k in kwargs:
            self.model_parameters[k] = kwargs[k]

        # pre-compute sygma model
        self.compute_yields()

        self.elements = self._sygma_model.history.elements
        self.elements = ['metals'] + self.elements

    def compute_yields(self, **kwargs):
        '''
        Runs through the Sygma model given model parameters.
        This function is not strictly necessary in a YieldsObject,
        but exists here to pre-compute information needed for
        the `yields` function so it is not re-computed each time
        it is called.
        '''

        for k in kwargs:
            self.model_parameters[k] = kwargs[k]

        if 'iniZ' not in self.model_parameters:
            self.model_parameters['iniZ'] = 0.02
            print('Using default iniZ %8.5f' % (self.model_parameters['iniZ']))

        if self.model_parameters['mgal'] != 1.0:
            print('Galaxy mass is not 1 solar mass. Are you sure?')
            print('This will throw off scalings')

        self._sygma_model = sygma.sygma(**self.model_parameters)

        # get yields. This is the 'ISM' mass fraction of all elements. But since
        # mgal above is 1.0, this is the solar masses of each element in the ISM
        # as a function of time. The diff of this is dM. Need to skip first
        # which is the initial values
        skip_index = 1
        skip_h_he = 2

        self._model_yields = np.array(self._sygma_model.history.ism_elem_yield)[skip_index:, :]
        # construct total metals (2: skips H and He)
        self._total_metals = np.sum(self._model_yields[:, skip_h_he:], axis=1)
        self._model_time = np.array(self._sygma_model.history.age[1:]) / 1.0e9  # in yr -> Gyr

        # in Msun / Gyr
        # print(np.shape(dt), np.shape(self._model_yields), np.shape(self._model_total_metal_rate))
        dt = np.diff(self._model_time)
        self._model_yield_rate = (np.diff(self._model_yields, axis=0).transpose() / dt).transpose()
        self._model_total_metal_rate = (np.diff(self._total_metals) / dt).transpose()

    def yields(self, t, element):
        '''

        Returns the total yields for all yield channels in NuGrid.
        This method is REQUIRED by construct_yield_table.

        Parameters
        -----------
        t : float or np.ndarray
            Time (in Gyr) to compute instantaneous yield rate
        element : str : Must be in self.elements
            Element name

        Returns
        -----------
        y : float or np.ndarray
            Total yields at a given time for desired element in units of
            Msun / Gyr per Msun of star formation.
        '''

        assert element in self.elements

        x = 0.5 * (self._model_time[1:] + self._model_time[:-1])

        if element == 'metals':
            y = self._model_total_metal_rate
        else:
            element_index = self._sygma_model.history.elements.index(element)
            y = self._model_yield_rate[:, element_index]

        return np.interp(t, x, y)
