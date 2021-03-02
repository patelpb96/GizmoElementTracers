#!/usr/bin/env python3

'''
Assign elemental abundances to star and gas particles in post-processing using age-tracer
weights stored in Gizmo simulation snapshots.

If the age-tracer model was enabled when running a Gizmo simulation (by defining
GALSF_FB_FIRE_AGE_TRACERS, to set the number of age bins, in Gizmo's Config.sh),
each gas particle stored an array of weights in bins of stellar age, assigned to the gas particle
from any neighboring star particles as the simulation ran, to record the mass (fraction) of
ejecta/winds that would have been deposited into the gas particle had there been a metal enrichment
event at that timestep. Thus, a star particle of a given age deposited weights into its
corresponding stellar age bin into its neighboring gas particles.
A star particle then inherits the array of age-tracer mass weights from its progenitor gas particle.

Assigning elemental abundances to particles at z = 0 (or any snapshot) then requires you to
set/compute the total nucleosynthetic yields from all stellar processes within/across each
age-tracer stellar age bin for each element.
Pass this to ElementAgeTracerClass as a dictionary of 1-D arrays, which contain the total mass
(fraction) of each element that a stellar population produces within each age bin.
You can construct this dictionary using a default rate + yield model below,
or you can define your own custom rate and yield model that accepts stellar age[s] and an element
name as inputs and returns the instantaneous mass-loss rate of that element at that stellar age.

For reference, these are the relevant age-tracer settings in Gizmo configuation and parameter files:
    in Config.sh
        GALSF_FB_FIRE_AGE_TRACERS - master switch that turns on age-tracers and sets the number
            of age bins, which by default are equally spaced in log age
        GALSF_FB_FIRE_AGE_TRACERS_CUSTOM - instead, read a custom list of arbitrary age bins
    in gizmo_parameters.txt
        AgeTracerBinStart - minimum age of age bins (if not custom list) [Myr]
        AgeTracerBinEnd - maximum age of age bins (if not custom list) [Myr]
        AgeTracerListFilename - name of text file that contains custom age bins
        AgeTracerActiveTimestepFraction - targeted number of deposition events per age bin,
            if <= 0, deposit age-tracer weights at each timestep

@author:
    Andrew Wetzel <arwetzel@gmail.com>
    Andrew Emerick <aemerick11@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    time [Myr] (different than most other modules in this package, which default to Gyr)
    elemental abundance [(linear) mass fraction]
'''

import numpy as np
from scipy import integrate

from utilities import constant


# --------------------------------------------------------------------------------------------------
# class for using age-tracer weights to assign elemental abundances
# --------------------------------------------------------------------------------------------------
class ElementAgeTracerClass(dict):
    '''
    Class for storing and using the age-tracer array of weights from a Gizmo simulation to assign
    elemental abundances to star and gas particles in post-processing.
    '''

    def __init__(self, header_dict=None, element_index_start=11):
        '''
        Initialize self dictionary to store all age-tracer information.

        Parameters
        ----------
        header_dict : dict
            dictionary that contains header information from a Gizmo snapshot file, as tabulated in
            gizmo_io.py, to assign all age-tracer age bin information. [optional]
            if you do not input header_dict, you need to assign age bins via assign_age_bins()
        element_index_start : int
            index of first age-tracer field in Gizmo's particle element mass fraction array.
            if input header_dict, it will over-ride any value here.
        '''
        # min and max ages [Myr] to impose on age bins
        # impose this *after* defining the bins, so it does not affect bin spacing
        # it only affect the integration of the yield rates into time-averaged age bins
        self._age_min_impose = 0
        self._age_max_impose = 138000

        # whether using custom stellar age bins
        # default (non-custom) is equally spaced in log age
        self['has.custom.age.bin'] = False
        # targeted number of (sampled) injection events per age bin when the simulation ran
        self['event.number.per.age.bin'] = None
        # number of stellar age bins
        self['age.bin.number'] = None
        # array of ages of bin edges [Myr]
        # should have N_age-bins + 1 values: left edges plus right edge of final bin
        self['age.bins'] = None
        # starting index of age-tracer fields within Gizmo's particle massfraction array
        self['element.index.start'] = None
        # dictionary of nucleosynthetic yields for each element within each age bin
        self['yields'] = {}
        # float or dictionary to store initial conditions of elemental abundances
        self['massfractions.initial'] = None

        if header_dict is not None:
            self.assign_age_bins(header_dict)
            # assign from header
            self['element.index.start'] = (
                header_dict['element.number'] - header_dict['agetracer.number']
            )
        elif element_index_start is not None:
            self['element.index.start'] = element_index_start

    def assign_age_bins(
        self, header_dict=None, age_bins=None, age_bin_number=None, age_min=None, age_max=None
    ):
        '''
        Assign to self the age bins used by the age-tracer module in a Gizmo simulation.
        You can do this 3 ways:
            (1) input a dictionary that contains the header information from a Gizmo simulation,
                as tabulated in gizmo_io.py, which contains the age-tracer age bin information
            (2) input an array of (custom) age bins
            (3) input the number of age bins and the min and max age,
                to use to generate age bins, assuming equal spacing in log age

        Parameters
        ----------
        header_dict : dict
            dictionary that contains header information from a Gizmo snapshot file
            use this to assign age-tracer age bin information
        age_bins : array
            age bins [Myr], with N_age-bins + 1 values: left edges plus right edge of final bin
        age_bin_number : int
            number of age bins
        age_min : float
            minimum age (left edge of first bin), though over-ride this to be age_min_impose
        age_max : float
            maximum age (right edge of final bin), though over-ride this to be age_max_impose
        '''
        if header_dict is not None:
            if 'agetracer.number' not in header_dict:
                print('! input header dict, but it has no age-tracer information')
                print('  assuming age-tracers were not enabled in this Gizmo simulation')
                return
            elif header_dict['agetracer.number'] < 1:
                print(
                    '! header dict indicates only {} age-tracer bins, which is non-sensical'.format(
                        header_dict['agetracer.number']
                    )
                )
                return

            self['age.bin.number'] = header_dict['agetracer.number']

            if 'agetracer.events.per.bin' in header_dict:
                self['events.per.age.bin'] = header_dict['agetracer.events.per.bin']

            if 'agetracer.min' in header_dict and 'agetracer.max' in header_dict:
                assert header_dict['agetracer.min'] > 0 and header_dict['agetracer.max'] > 0
                self['age.bins'] = np.logspace(
                    np.log10(header_dict['agetracer.min']),
                    np.log10(header_dict['agetracer.max']),
                    header_dict['agetracer.number'] + 1,
                )
            elif 'agetracer.bins' in header_dict:
                assert len(header_dict['agetracer.bins']) > 1
                assert header_dict['agetracer.number'] == len(header_dict['agetracer.bins']) - 1
                self['age.bins'] = header_dict['agetracer.bins']
                self['has.custom.age.bin'] = True
            else:
                print('! input header dict, but cannot make sense of age-tracer information')
                return

        elif age_bins is not None:
            # assume input custom age bins
            assert len(age_bins) > 1
            self['age.bins'] = age_bins
            self['age.bin.number'] = len(age_bins) - 1
            self['has.custom.age.bin'] = True

        elif age_bin_number is not None:
            # assume uniformly log spaced age bins
            assert age_bin_number > 0 and age_min > 0 and age_max > 0
            self['age.bin.number'] = age_bin_number
            self['age.bins'] = np.logspace(np.log10(age_min), np.log10(age_max), age_bin_number + 1)

        else:
            raise ValueError(
                'not sure how to parse inputs to assign_age_bins():'
                + f' header_dict = {header_dict}, age_bins = {age_bins},'
                + f' age_bin_number = {age_bin_number}, age_min = {age_min}, age_max = {age_max}'
            )

        # ensure minimum and maximum age of
        self['age.bins'][0] = self._age_min_impose
        if self['age.bins'][-1] > self._age_max_impose:
            self['age.bins'][-1] = self._age_max_impose

    def _read_age_bins(self, directory='.', file_name='agetracer_bins.txt'):
        '''
        Read file that contains (custom) age bins for age-tracer model.
        Relevant if defined GALSF_FB_FIRE_AGE_TRACERS_CUSTOM in Gizmo's Config.sh.
        Gizmo sets file_name via AgeTracerListFilename in gizmo_parameters.txt.

        Gizmo now stores this information in its header, but retain this method for debugging.

        Parameters
        ----------
        directory : str
            base directory of simulation
        file_name : str
            name of file that contains custom age bins for the age-tracer model
            this should be a single column of bin left edges plus the right edge of the final bin,
            so the number of lines should be number of age bins + 1
        '''
        if directory[-1] != '/':
            directory += '/'
        path_file_name = directory + file_name

        try:
            self['age.bins'] = np.genfromtxt(path_file_name)
            self['age.bin.number'] = len(self['age.bins']) - 1
        except OSError as exc:
            raise OSError(
                f'cannot read file of custom age-tracer age bins:  {path_file_name}'
            ) from exc

    def assign_element_yields(self, element_yield_dict=None):
        '''
        Assign to self a dictionary of stellar nucleosynthetic yields within stellar age bins.

        Parameters
        -----------
        element_yield_dict : dict of 1-D arrays
            nucleosynthetic yield fractional mass [M_sun per M_sun of stars formed] of each element
            produced within/across each age bin, to map the age-tracer mass weights in each age bin
            into actual element yields
        '''
        for element_name in element_yield_dict:
            assert len(element_yield_dict[element_name]) == self['age.bin.number']
            self['yields'][element_name] = np.array(element_yield_dict[element_name])

    def assign_element_massfraction_initial(
        self, massfraction_initial_dict=None, metallicity=None, helium_massfraction=0.24
    ):
        '''
        Set the initial conditions for the elemental abundances (mass fractions),
        to add to the age-tracer nucleosynthetic yields.

        Parameters
        ----------
        massfraction_initial_dict : dict
            Keys are element names and values are (linear) absolute mass fractions for each element,
            to use as initial conditions (at arbitrarily early cosmic times) for each particle.
            Default to 0 for any element not in this input massfraction_initial_dict.
        metallicity : float
            (linear) metallity relative to Solar.
            If defined, assume that input massfraction_initial_dict mass fractions are relative to
            Solar, and scale them by this value.
        helium_massfraction : float
            If defined, use this for the initial mass fraction of helium, over-writing any value in
            input massfraction_initial_dict.
        '''
        # sanity checks
        assert self['yields'] is not None and len(self['yields']) > 0
        for element_name in massfraction_initial_dict:
            assert element_name in self['yields']
        if metallicity is not None:
            # ensure is linear mass fraction relative to Solar
            assert np.isscalar(metallicity) and metallicity >= 0

        if isinstance(massfraction_initial_dict, dict) and len(massfraction_initial_dict) > 0:
            # initialize to 0 for all elements in model, then over-write with elements in input dict
            self['massfractions.initial'] = {}
            for element_name in self['yields']:
                self['massfractions.initial'][element_name] = 0
            for element_name in massfraction_initial_dict:
                self['massfractions.initial'][element_name] = massfraction_initial_dict[
                    element_name
                ]
                if metallicity is not None:
                    self['massfractions.initial'][element_name] *= metallicity

            if helium_massfraction is not None and 'helium' in self['massfractions.initial']:
                self['massfractions.initial']['helium'] = helium_massfraction

        else:
            print(
                '! not sure how to parse input massfraction_initial_dict'
                + f' = {massfraction_initial_dict}'
            )

    def get_element_massfractions(self, element_name, agetracer_mass_weights, _metallicities=None):
        '''
        Get the elemental abundances (mass fractions) for input element_name[s],
        using the the input 2-D array of age-tracer weights.

        Before you call this method, you must:
            set up the age bins via:  assign_age_bins()
            assign the nucleosynthetic yield within each age bin for each element via:
                assign_element_yields()
            (optinally) assign the initial abundance (mass fraction) for each element via:
                assign_element_massfraction_initial()

        Parameters
        ----------
        element_name : str
            name of element to get mass fraction of for each particle
        agetracer_mass_weights : 2-D array (N_particle x N_age-bins)
            age-tracer mass weights for particles - should be values from
                part[species_name]['massfraction'][:, self['element.index.start']:],
                where species_name = 'star' or 'gas'

        Returns
        -------
        element_mass_fractions : 1-D array
            mass fraction of element_name for each particle
        '''
        # sanity check, if input element symbol or other alais, convert to default name
        if element_name not in self['yields']:
            if element_name in constant.element_name_from_symbol:
                element_name = constant.element_name_from_symbol[element_name]
            else:
                raise KeyError(
                    f'cannot compute mass fraction for element_name = {element_name}\n'
                    + 'element age-tracer dictionary has these elements available:  {}'.format(
                        self['yields'].keys()
                    )
                )

        # weight the yield within each age bin by the age-tracer mass weights
        # and sum all age bins to get the total enrichment
        element_mass_fractions = np.sum(
            agetracer_mass_weights * self['yields'][element_name], axis=1
        )

        # add initial abundances (if applicable)
        if self['massfractions.initial'] is not None:
            if isinstance(self['massfractions.initial'], dict):
                # if using a different initial abundance for each element
                assert element_name in self['massfractions.initial']
                element_mass_fractions += self['massfractions.initial'][element_name]
            elif self['massfractions.initial'] > 0:
                # if using the same initial abundance for all elements
                element_mass_fractions += self['massfractions.initial']

        return element_mass_fractions


ElementAgeTracer = ElementAgeTracerClass()


class ElementAgeTracerZClass(ElementAgeTracerClass):
    '''
    Store and assign yields in bins of progenitor metallicity.
    '''

    def assign_element_yields(self, element_yield_dicts=None, progenitor_metal_massfractions=None):
        '''
        Assign to self a dictionary of stellar nucleosynthetic yields within stellar age bins.

        Parameters
        -----------
        element_yield_dicts : list of dicts of 1-D arrays
            nucleosynthetic yield fractional mass [M_sun per M_sun of stars formed] of each element
            produced within/across each age bin, to map the age-tracer mass weights in each age bin
            into actual element yields
        '''
        element_yield_dict = element_yield_dicts[0]
        element_name = tuple(element_yield_dict.keys())[0]
        for element_name in element_yield_dict:
            self['yields'][element_name] = np.zeros(
                (progenitor_metal_massfractions.size, element_yield_dict[element_name].size),
                element_yield_dict[element_name].dtype,
            )

        for zi, _progenitor_metal_massfractions in enumerate(progenitor_metal_massfractions):
            element_yield_dict = element_yield_dicts[zi]
            for element_name in element_yield_dict:
                if element_name == 'metals':
                    continue
                assert len(element_yield_dict[element_name]) == self['age.bin.number']
                self['yields'][element_name][zi] = np.array(element_yield_dict[element_name])
                # assign element symbol name as dictionary key as well, for convenience later
                element_symbol = constant.element_symbol_from_name[element_name]
                self['yields'][element_symbol][zi] = np.array(element_yield_dict[element_name])

        self['progenitor.metal.massfractions'] = progenitor_metal_massfractions

    def get_element_massfractions(
        self, element_name, agetracer_mass_weights, metal_massfractions=None
    ):
        '''
        Get the elemental abundances (mass fractions) for input element_name[s],
        using the the input 2-D array of age-tracer weights.

        Before you call this method, you must:
            set up the age bins via:  assign_age_bins()
            assign the nucleosynthetic yield within each age bin for each element via:
                assign_element_yields()
            (optinally) assign the initial abundance (mass fraction) for each element via:
                assign_element_massfraction_initial()

        Parameters
        ----------
        element_name : str
            name of element to get mass fraction of for each particle
        agetracer_mass_weights : 2-D array (N_particle x N_age-bins)
            age-tracer mass weights for particles - should be values from
                part[species_name]['massfraction'][:, self['element.index.start']:],
                where species_name = 'star' or 'gas'
        metallicities : array

        Returns
        -------
        element_mass_fractions : 1-D array
            mass fraction of element_name for each particle
        '''
        # sanity check
        if element_name not in self['yields']:
            raise KeyError(
                f'cannot compute mass fraction for element_name = {element_name}\n'
                + 'element age-tracer dictionary has these elements available:  {}'.format(
                    self['yields'].keys()
                )
            )

        if metal_massfractions is not None:
            # prog_metal_massfractions = 10 ** 0.45 * metal_massfractions
            prog_metal_massfractions = metal_massfractions
            zis = np.digitize(prog_metal_massfractions, self['progenitor.metal.massfractions'])
            zis = np.clip(zis, 0, self['progenitor.metal.massfractions'].size - 1)

        # weight the yield within each age bin by the age-tracer mass weights
        # and sum all age bins to get the total enrichment
        element_mass_fractions = np.zeros(
            agetracer_mass_weights.shape[0], agetracer_mass_weights.dtype
        )
        for zi, _progenitor_metal_massfraction in enumerate(self['progenitor.metal.massfractions']):
            pis = np.where(zis == zi)[0]
            if pis.size > 0:
                element_mass_fractions[pis] = np.sum(
                    agetracer_mass_weights[pis] * self['yields'][element_name][zi], axis=1,
                )

        # add initial abundances (if applicable)
        if self['massfraction.initial'] is not None:
            if isinstance(self['massfraction.initial'], dict):
                # if using a different initial abundance for each element
                assert element_name in self['massfraction.initial']
                element_mass_fractions += self['massfraction.initial'][element_name]
            elif self['massfraction.initial'] > 0:
                # if using the same initial abundance for all elements
                element_mass_fractions += self['massfraction.initial']

        return element_mass_fractions


# --------------------------------------------------------------------------------------------------
# FIRE-2 and FIRE-3 model for stellar nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class FIREYieldClass:
    '''
    Provide the stellar nucleosynthetic yields in the FIRE-2 or FIRE-3 model.

    In FIRE-2, the following yields and mass-loss rates depend on progenitor metallicity:
        stellar winds: overall mass-loss rate and oxygen yield
        core-collapse supernovae: nitrogen yield
    '''

    def __init__(self, model='fire2', progenitor_metallicity=1.0):
        '''
        Parameters
        -----------
        model : str
            name for this rate + yield model
        progenitor_metallicity : float
            metallicity [(linear) mass fraction relative to Solar]
            for yields that depend on progenitor metallicity
        '''
        from . import gizmo_star

        self.model = model.lower()
        assert self.model in ['fire2', 'fire2.1', 'fire3']

        self.NucleosyntheticYield = gizmo_star.NucleosyntheticYieldClass(model)
        self.SupernovaCC = gizmo_star.SupernovaCCClass(model)
        self.SupernovaIa = gizmo_star.SupernovaIaClass(model)
        self.StellarWind = gizmo_star.StellarWindClass(model)
        self.sun_massfraction = self.NucleosyntheticYield.sun_massfraction

        # names of elements tracked in this model
        self.element_names = [element_name.lower() for element_name in self.sun_massfraction]
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

        # critical/transition/discontinuous ages [Myr] in this model to be careful around when
        # integrating across age to get cumulative yields
        # self.ages_critical = None
        self.ages_critical = np.sort([3.401, 10.37, 37.53, 1, 50, 100, 1000, 14000])

        # store this (default) progenitor metallicity, including the mass fraction for each element
        # use the latter to compute metallicity-dependent corrections to the yields
        # scale all progenitor abundances to Solar ratios - this is not accurate in detail,
        # but it provides better agreement between post-processed yields and native yields in FIRE-2
        self.progenitor_metallicity = progenitor_metallicity
        self.progenitor_massfraction_dict = {}
        for element_name in self.element_names:
            # scale to Solar abundance ratios
            self.progenitor_massfraction_dict[element_name] = (
                progenitor_metallicity * self.sun_massfraction[element_name]
            )

        # store all yields, because in FIRE-2 they are independent of both stellar age and
        # ejecta/mass-loss rates
        self.NucleosyntheticYield.assign_yields(
            # match FIRE-2
            progenitor_massfraction_dict=self.progenitor_massfraction_dict,
            # test: do not model correction of yields from pre-existing surface abundances
            # progenitor_metallicity=self.progenitor_metallicity,
            # progenitor_massfraction_dict=None,
        )

    def get_element_yields(self, age_bins, element_names=None):
        '''
        Construct and return a dictionary of stellar nucleosynthetic yields:
            Each key is an element name.
            Each value is a 1-D array of yields within each input age bin,
            constructed by integrating the total yield for each element across each age bin.

        Supply this element_yield_dict to ElementAgeTracerClass.assign_element_yields(),
        to assign elemental abundances to particles via age-tracer weights in a Gizmo simulation.

        Parameters
        -----------
        age_bins : array
            stellar age bins used in Gizmo for the age-tracer model [Myr]
            should have N_age-bins + 1 values: left edges plus right edge of final bin
        element_names : list
            names of elements to generate, if only generating a subset
            if input None, assign all elements in this model

        Returns
        -------
        element_yield_dict : dict of 1-D arrays
            fractional mass of each element [M_sun per M_sun of stars formed] produced within
            each age bin
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
                age_min = 0  # ensure min age starts at 0 Myr
            else:
                age_min = age_bins[ai]

            age_max = age_bins[ai + 1]

            for element_name in element_names:
                # get integrated total yield within the age bin
                element_yield_dict[element_name][ai] = integrate.quad(
                    self._get_element_yield_rate,
                    age_min,
                    age_max,
                    args=(element_name,),
                    points=self.ages_critical,
                )[0]

        return element_yield_dict

    def _get_element_yield_rate(self, age, element_name, progenitor_metallicity=None):
        '''
        Return the specific rate[s] [M_sun / Myr per M_sun of star formation] of nucleosynthetic
        yield[s] at input stellar age[s] [Myr] for input element_name, from all stellar processes.
        get_element_yields() uses this method to integrate across age within each age bin.

        Parameters
        ----------
        ages : float or array
            stellar age[s] at which to compute the nucleosynthetic yield rate[s] [Myr]
        element_name : str
            name of element, must be in self.element_names
        progenitor_metallicity : float
            metallicity [(linear) mass fraction, relative to Solar], for the stellar wind rate

        Returns
        -------
        element_yield_rate : float or array
            specific rate[s] of nucleosynthetic yield[s] at input age[s] for input element_name
            [M_sun / Myr per M_sun of star formation]
        '''
        assert element_name in self.element_names

        if progenitor_metallicity is None:
            progenitor_metallicity = self.progenitor_metallicity

        # stellar wind rate[s] at input age[s] [M_sun / Myr per M_sun of stars formed]
        # this is the only rate in FIRE-2 that depends on metallicity
        wind_rate = self.StellarWind.get_rate(age, metallicity=progenitor_metallicity)

        # core-collapse supernova rate[s] at input age[s] [Myr^-1 per M_sun of stars formed]
        sncc_rate = self.SupernovaCC.get_rate(age)

        # supernova Ia rate[s] at input age[s] [Myr^-1 per M_sun of stars formed]
        snia_rate = self.SupernovaIa.get_rate(age)

        element_yield_rate = (
            self.NucleosyntheticYield.wind_yield[element_name] * wind_rate
            + self.NucleosyntheticYield.snia_yield[element_name] * snia_rate
            + self.NucleosyntheticYield.sncc_yield[element_name] * sncc_rate
        )  # [M_sun / Myr]

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

        # [M_sun / Gyr]
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
