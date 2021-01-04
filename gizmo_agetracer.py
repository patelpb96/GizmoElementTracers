#!/usr/bin/env python3

'''
Assign elemental abundances to star and gas particles in post-processing using age-tracer passive
scalars in Gizmo simulations.

If age-tracer was enabled when running a Gizmo simulation, each gas particle stored an array of
weights in bins of stellar age, assigned as the simulation ran, when star particles of a given age
deposited weights (into their corresponding age bin) into their neighboring gas particles.
Each star particle inherits the age-tracer weight array from its progenitor gas particle.

Assigning elemental abundances then requires you to compute the weightings for each stellar age bin
for each element. This consists of a table that contains the total amount of mass of each element
that a stellar population produced in each age bin. You can construct this using a default
rate + yield model below, or you can define your own custom rate and yield model that accepts
stellar age[s] and an element name as inputs and returns the instantaneous mass-loss rate of that
element at that stellar age.

@author:
    Andrew Emerick <aemerick11@gmail.com>
    Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    time [Gyr]
    elemental abundance [(linear) mass fraction]
'''

import os
import numpy as np
from scipy import integrate

import utilities as ut


# --------------------------------------------------------------------------------------------------
# master class for using age tracers to assign elemental abundances
# --------------------------------------------------------------------------------------------------
class ElementAgetracerClass(ut.io.SayClass):
    '''
    Class for storing and using the age-tracer weights from a Gizmo simulation to assign
    elemental abundances to star and gas particles in post-processing.
    '''

    def __init__(self):
        '''
        .
        '''
        self.info = {
            'age.bin.number': 0,
            'has.custom.age.bins': False,
        }

        # stellar age bins [Gyr]
        # should have N_age-bins + 1 values: left edges plus right edge of final bin
        self.age_bins = None
        self.yield_dict = {}
        self._abundances_initial = {}  # dictionary to store elemental abundances at the IC

    def assign_element_yield_dict(self, element_yield_dict):
        '''
        Assign the yield table and list of element names to self.

        Parameters
        -----------
        element_yield_dict : dict of 1-D arrays
            nucleosynthetic yield fractional mass [M_sun per M_sun of stars formed] of each element
            produced *within* each age bin, to map the age-tracer weights into individual element
            yields in each age bin
        '''
        for element_name in element_yield_dict:
            assert np.shape(element_yield_dict[element_name])[1] == self.info['age.bin.number']

        for element_name in element_yield_dict:
            self.yield_dict[element_name] = 1.0 * element_yield_dict[element_name]

    def assign_element_abundances_intitial(self, initial_abundances):
        '''
        Set the initial conditions for the elemental abundances, to add to the age-tracer enrichment
        values.

        Parameters
        ----------
        initial_abundances : float or dict
            Dictionary with keys matching element names and values corresponding
            to the initial mass fractions for that element. Excluded elements
            will be given an initial abundance of zero by default.
        '''
        if self.element_names is None:
            print('first must set element_names via assign_element_yields()')
            raise RuntimeError

        # set initial values in both element name and symbol for versatility
        for e in self.element_names:
            self._initial_abundances[e] = 0.0
            if e != 'metals' and 'rprocess' not in e:
                self._initial_abundances[ut.constant.element_map_name_symbol[e]] = 0.0

        # set actual values in both name and symbol
        for element_name in initial_abundances:
            self._initial_abundances[element_name] = initial_abundances[element_name]
            if element_name != 'metals' and 'rprocess' not in element_name:
                self._initial_abundances[
                    ut.constant.element_map_name_symbol[element_name]
                ] = initial_abundances[element_name]

    def read_assign_age_bins(self, directory='.', file_name='age_bins.txt'):
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

        self.assign_age_bins(directory, file_name)

    def read_age_bin_info(self, directory='.'):
        '''
        Check for GIZMO_config.h in the simulation directory, or the gizmo/ sub-directory, or
        for gizmo.out in the simulation directory. Need one of these to read the compile-time
        flags that determine the number of age-tracer time bins used in the simulation.

        Parameters
        -----------
        directory : str
            base directoy of a simulation
        '''
        directory = ut.io.get_path(directory)

        possible_file_names = ['GIZMO_config.h', 'gizmo_config.h', 'gizmo.out', 'gizmo.out.txt']
        possible_path_names = [directory, ut.io.get_path(directory + 'gizmo')]

        path_file_name = None
        for file_name in possible_file_names:
            for path_name in possible_path_names:
                if os.path.exists(path_name + file_name):
                    path_file_name = path_name + file_name
                    break

        if path_file_name is None:
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

    def assign_age_bins(self, directory='.', file_name='age_bins.txt'):
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
        Read file that contains custom age bins for age tracers.
        Relevant if defined GALSF_FB_FIRE_AGE_TRACERS_CUSTOM in Config.sh.
        file_name set via AgeTracerListFilename in gizmo_parameters.txt.

        Parameters
        ----------
        directory : str
            top-level simulation directory
        file_name : str
            name of file that contains custom age bins for age tracers
            assume this to be a single column of bin left edges plus the right edge of final bin
            (number of lines should be number of tracers + 1)
        '''
        try:
            path_file_name = ut.io.get_path(directory) + file_name
            self.age_bins = np.genfromtxt(path_file_name)
        except OSError as exc:
            raise OSError(f'cannot find file of age-tracer age bins: {path_file_name}') from exc


ElementAgetracer = ElementAgetracerClass()


# --------------------------------------------------------------------------------------------------
# FIRE-2 and FIRE-3 model for nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class FIREYieldClass:
    '''
    Provide the stellar nucleosynthetic yields in the FIRE-2 or FIRE-3 model.

    In FIRE-2, these nucleosynthetic yields and mass-loss rates depend on progenitor metallicity:
        stellar wind: overall mass-loss rate and oxygen yield
        core-collapse supernova: nitrogen yield
    '''

    def __init__(self, model='fire2', progenitor_metallicity=1.0):
        '''
        Parameters
        -----------
        model : str
            name for this yield model
        progenitor_metallicity : float
            metallicity [mass fraction wrt Solar], for yields that depend on progenitor metallicity
        '''
        from . import gizmo_star

        self.model = model.lower()
        assert self.model in ['fire2', 'fire3']

        self.NucleosyntheticYield = gizmo_star.NucleosyntheticYieldClass(model)
        self.SupernovaCC = gizmo_star.SupernovaCCClass(model)
        self.SupernovaIa = gizmo_star.SupernovaIaClass(model)
        self.StellarWind = gizmo_star.StellarWindClass(model)

        # names of elements tracked in this model
        self.element_names = [
            element_name.lower() for element_name in self.NucleosyntheticYield.sun_massfraction
        ]
        """
        self.element_names = [
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
        """

        # critical/transition/discontinuous ages [Gyr] in this model to be careful around when
        # integrating across age to get cumulative yields
        # self.ages_critical = None
        self.ages_critical = np.sort([3.401, 10.37, 37.53, 1, 50, 100, 1000, 14000]) / 1000  # [Gyr]

        # store this (default) progenitor metallicity and the mass fraction of each element
        # use the latter to compute metallicity-dependent corrections to the yields
        # scale all abundances to Solar abundance ratios - this is not accurate in detail,
        # but it provides better agreement between post-processed yields and native yields in FIRE-2
        self.progenitor_metallicity = progenitor_metallicity
        self.progenitor_massfraction_dict = {}
        for element_name in self.element_names:
            # scale to Solar abundance ratios
            self.progenitor_massfraction_dict[element_name] = (
                progenitor_metallicity * self.NucleosyntheticYield.sun_massfraction[element_name]
            )

        # store all yields, because in FIRE-2 they are independent of both stellar age and
        # ejecta/mass-loss rates
        self.NucleosyntheticYield.assign_yields(
            progenitor_massfraction_dict=self.progenitor_massfraction_dict
        )

    def get_element_yield_dict(self, age_bins, element_names=None):
        '''
        Construct and return a dictionary of nucleosynthetic yields,
        with element names as keys, and a 1-D array of yields in each age bins as values.
        Construct by integrating the total yield for each element within each input age bin.
        Use to assign elemental abundances via the age-tracer weights in a Gizmo simulation.

        Parameters
        -----------
        age_bins : array
            stellar age bins used in Gizmo for age-tracer module [Gyr]
            should have N_age-bins + 1 values: left edges plus right edge of final bin
        element_names : list
            names of elements to generate, if only generating a subset
            if None, assign all elements in this model

        Returns
        -------
        element_yield_dict : dict of 1-D arrays
            fractional mass of each element [M_sun per M_sun of stars formed] produced *within* each
            age bin
        '''
        if element_names is None:
            element_names = self.element_names  # generate yields for all elements in this model
        else:
            element_names_safe = []
            for element_name in element_names:
                assert element_name in self.element_names
                element_names_safe = element_name.lower()
            element_names = element_names_safe

        element_yield_dict = {}
        for element_name in element_names:
            element_yield_dict[element_name] = np.zeros(np.size(age_bins) - 1)

        # ages to be careful around during integration
        if not hasattr(self, 'ages_critical'):
            self.ages_critical = None

        # compile yields within each age bin by integrating over the underlying rates
        for ai in np.arange(np.size(age_bins) - 1):
            if ai == 0:
                age_min = 0  # ensure min age starts at 0
            else:
                age_min = age_bins[ai]

            age_max = age_bins[ai + 1]

            for element_name in element_names:
                element_yield_dict[element_name][ai] = integrate.quad(
                    self._get_yield_rate,
                    age_min,
                    age_max,
                    args=(element_name,),
                    points=self.ages_critical,
                )[0]

        return element_yield_dict

    def _get_yield_rate(self, age, element_name, progenitor_metallicity=None):
        '''
        Return the specific rate[s] [M_sun / Gyr per M_sun of star formation] of nucleosynthetic
        yield[s] at input stellar age[s] [Gyr] for input element_name, from all stellar processes.
        get_yields() uses this method to integrate over ages within each stellar age bin.

        Parameters
        ----------
        ages : float or array
            stellar age[s] [Gyr] at which to compute the nucleosynthetic yield rate[s]
        element_name : str
            name of element, must be in self.element_names
        progenitor_metallicity : float
            metallicity [linear mass fraction, wrt Solar], for stellar wind rate

        Returns
        -------
        element_yield_rate : float or array
            specific rate[s] of nucleosynthetic yield[s] at input age[s] for input element_name
            [M_sun / Gyr per M_sun of star formation]
        '''

        assert element_name in self.element_names

        if progenitor_metallicity is None:
            progenitor_metallicity = self.progenitor_metallicity

        age_Myr = age * 1000  # convert to [Myr]

        # stellar wind rate[s] at input age[s] [M_sun / Myr per M_sun of stars formed]
        # this is the only rate in FIRE-2 that depends on metallicity
        wind_rate = self.StellarWind.get_rate(age_Myr, metallicity=progenitor_metallicity)

        # core-collapse supernova rate[s] at input age[s] [Myr^-1 per M_sun of stars formed]
        sncc_rate = self.SupernovaCC.get_rate(age_Myr)

        # supernova Ia rate[s] at input age[s] [Myr^-1 per M_sun of stars formed]
        snia_rate = self.SupernovaIa.get_rate(age_Myr)

        element_yield_rate = (
            self.NucleosyntheticYield.wind_yield[element_name] * wind_rate
            + self.NucleosyntheticYield.snia_yield[element_name] * snia_rate
            + self.NucleosyntheticYield.sncc_yield[element_name] * sncc_rate
        )  # [M_sun / Myr]

        element_yield_rate *= 1000  # [M_sun / Gyr]

        return element_yield_rate


# --------------------------------------------------------------------------------------------------
# NuGrid/Sygma model for nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class NuGridYieldClass:
    '''
    Object designed for use with the construct_yield_table method. This
    object proves the yields from the NuGrid collaboration using the
    Sygma interface. The NuPyCEE code must be instssssssalled (and its dependencies)
    in the python path for this object to work.

    https://nugrid.github.io/NuPyCEE/index.html
    '''

    def __init__(self, name='NuGrid', **kwargs):
        '''
        .
        '''
        try:
            import sygma  # pyright: reportMissingImports=false

            self.sygma = sygma
        except ImportError as exc:
            raise ImportError(
                '! Cannot load the NuPyCEE module Sygma.'
                + ' This is necessary to use the NuGrid yields.'
                + ' Make sure this is installed in your python path.'
                + ' For more info: https://nugrid.github.io/NuPyCEE/index.html'
            ) from exc

        self.name = name

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
        This function is not strictly necessary in a YieldsObject, but exists here to pre-compute
        information needed for the `yields` function so it is not re-computed each time called.
        '''

        for k in kwargs:
            self.model_parameters[k] = kwargs[k]

        if 'iniZ' not in self.model_parameters:
            self.model_parameters['iniZ'] = 0.02
            print('Using default iniZ %8.5f' % (self.model_parameters['iniZ']))

        if self.model_parameters['mgal'] != 1.0:
            print('Galaxy mass is not 1 solar mass. Are you sure?')
            print('This will throw off scalings')

        self._sygma_model = self.sygma.sygma(**self.model_parameters)

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

    def get_yields(self, t, element):
        '''

        Returns the total yields for all yield channels in NuGrid.
        This method is REQUIRED by construct_yield_table.

        Parameters
        -----------
        t : float or array
            time [Gyr] to compute instantaneous yield rate
        element : str
            element name, must be in self dictionary

        Returns
        -----------
        y : float or array
            total yields at a given time for desired element in units of Msun / Gyr per Msun of
            star formation
        '''

        assert element in self.elements

        x = 0.5 * (self._model_time[1:] + self._model_time[:-1])

        if element == 'metals':
            y = self._model_total_metal_rate
        else:
            element_index = self._sygma_model.history.elements.index(element)
            y = self._model_yield_rate[:, element_index]

        return np.interp(t, x, y)
