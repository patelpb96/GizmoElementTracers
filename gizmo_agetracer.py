'''
Assign elemental abundances to star and gas particles in post-processing using age-tracer
mass weights stored in Gizmo simulation snapshots.

If age-tracers were enabled when running a Gizmo simulation
(by defining GALSF_FB_FIRE_AGE_TRACERS in Config.sh),
each gas particle stores an array of mass weights in bins of stellar age,
assigned to the gas particle from any neighboring star particles as the simulation ran,
to record the fraction of mass from ejecta/winds that would have been deposited into the
gas particle had there been an enrichment event at that timestep.
In other words, a star particle of a given age deposits a mass weight into a
corresponding stellar age bin in all of its neighboring gas particles.

The each age-tracer weight is dimensionless.
It represents the mass fraction (relative to the gas particle mass) of winds/ejecta deposited into
it (as a gas particle) during each age-tracer age bin.
An age-tracer weight of 1 in an age bin therefore means that the particle received winds/ejecta
from its own mass of stars for a duration that spans the entire age bin.
In other words, it received the equivalent of all of the winds/ejecta across that time bin from
a star particles equal to its own mass.
Thus, to convert an age-tracer weight to an abundance, just multiply it by the mass of an element
produced per mass of star formation, integrated across a given age bin.

A star particle then inherits the array of age-tracer weights from its progenitor gas particle.

Assigning elemental abundances to particles at any snapshot then requires you to compile the
total nucleosynthetic yield mass fractions (relative to the mass of stars at formation)
from all stellar processes within/across each age-tracer stellar age bin for each element.
Store this as a dictionary of 1-D arrays, where each array contain the total mass fraction
(relative to the mass of stars at formation) that a stellar population produces within/across each
age bin, and where each dictionary key corresponds to an element.
You can construct this dictionary using a default rate + yield model below, or you can define your
own custom rate and yield model that accepts stellar age[s] and an element name as inputs and
returns the instantaneous fractional mass-loss rate of that element at that stellar age.
Then supply this dictionary to ElementAgeTracerClass.assign_element_yield_massfractions().

For reference, these are the relevant age-tracer settings in Gizmo configuation and parameter files
    in Config.sh
        GALSF_FB_FIRE_AGE_TRACERS - master switch that turns on age-tracers and sets the number
            of age bins, which by default are equally spaced in log age
        GALSF_FB_FIRE_AGE_TRACERS_CUSTOM - instead, read a custom list of arbitrary age bins
    in gizmo_parameters.txt
        AgeTracerBinStart - minimum age of age bins (if not custom list) [Myr]
        AgeTracerBinEnd - maximum age of age bins (if not custom list) [Myr]
        AgeTracerListFilename - name of text file that contains custom age bins
        AgeTracerActiveTimestepFraction (stored as AgeTracerRateNormalization within Gizmo)
        - targeted number of deposition events per age-tracer age bin,
        if <= 0, deposit age-tracer weights at each particle timestep

----------
@author:
    Andrew Wetzel <arwetzel@gmail.com>
    Andrew Emerick <aemerick11@gmail.com>

----------
Units

Unless otherwise noted, all quantities are in (combinations of)
    mass [M_sun]
    time [Myr] (note: most other modules in this GizmoAnalysis package default to Gyr)
    elemental abundance [linear mass fraction]
'''

import numpy as np
from scipy import integrate

from utilities import constant


# --------------------------------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------------------------------
def parse_element_names(element_names_possible, element_names=None, scalarize=False):
    '''
    .
    '''
    if element_names is None:
        element_names_safe = element_names_possible  # use all elements in this model
    else:
        if np.isscalar(element_names):
            element_names = [element_names]

        element_names_safe = []
        for element_name in element_names:
            if element_name not in element_names_possible:
                if element_name in constant.element_name_from_symbol:
                    element_name = constant.element_name_from_symbol[element_name]
                else:
                    raise KeyError(
                        f'cannot parse input element_name = {element_name}\n'
                        + f'only these elements are valid inputs:  {element_names_possible}'
                    )
            element_names_safe.append(element_name.lower())

    if scalarize and len(element_names_safe) == 1:
        element_names_safe = element_names_safe[0]

    return element_names_safe


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
        Initialize self's dictionary to store all age-tracer information.

        Parameters
        ----------
        header_dict : dict
            dictionary that contains header information from a Gizmo snapshot file, as tabulated in
            gizmo_io.py, to assign all age-tracer age bin information.
            If you do not input header_dict, you need to assign age bins via assign_age_bins().
        element_index_start : int
            index of first age-tracer field in Gizmo's particle element mass fraction array.
            If you input header_dict, it will over-ride any value here.
        '''
        # min and max ages [Myr] to impose on age bins
        # impose this *after* defining the bins, so it does not affect bin spacing
        # this only affect the integration of the yield rates into age-bin yields
        self._age_min_impose = 0
        self._age_max_impose = 13700

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
        self['yield.massfractions'] = {}
        # dictionary to store initial conditions of elemental abundances
        self['initial.massfraction'] = {}

        if header_dict is not None:
            self.assign_age_bins(header_dict)
            # assign from header
            self['element.index.start'] = (
                header_dict['element.number'] - header_dict['agetracer.age.bin.number']
            )
        elif element_index_start is not None:
            self['element.index.start'] = element_index_start

    def assign_age_bins(
        self, header_dict=None, age_bins=None, age_bin_number=None, age_min=None, age_max=None
    ):
        '''
        Assign to self the age bins used by the age-tracer module in a Gizmo simulation.
        You can do this 3 ways:
            (1) input a dictionary of the header information from a Gizmo simulation snapshot,
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
            minimum age (left edge of first bin) [Myr] (over-ride this to be age_min_impose)
        age_max : float
            maximum age (right edge of final bin) [Myr] (over-ride this to be age_max_impose)
        '''
        if header_dict is not None:
            # use header dictionary to get age bins
            if 'agetracer.age.bin.number' not in header_dict:
                print('! input header dict, but it has no age-tracer information')
                print('  assuming age-tracers were not enabled in this Gizmo simulation')
                return
            elif header_dict['agetracer.age.bin.number'] < 1:
                print(
                    '! header dict indicates only {} age-tracer bins, which is non-sensical'.format(
                        header_dict['agetracer.age.bin.number']
                    )
                )
                return

            self['age.bin.number'] = header_dict['agetracer.age.bin.number']

            if 'agetracer.event.number.per.age.bin' in header_dict:
                self['event.number.per.age.bin'] = header_dict['agetracer.event.number.per.age.bin']

            if 'agetracer.age.min' in header_dict and 'agetracer.age.max' in header_dict:
                assert header_dict['agetracer.age.min'] > 0 and header_dict['agetracer.age.max'] > 0
                self['age.bins'] = np.logspace(
                    np.log10(header_dict['agetracer.age.min']),
                    np.log10(header_dict['agetracer.age.max']),
                    header_dict['agetracer.age.bin.number'] + 1,
                )
            elif 'agetracer.age.bins' in header_dict:
                assert len(header_dict['agetracer.age.bins']) > 1
                assert (
                    header_dict['agetracer.age.bin.number']
                    == len(header_dict['agetracer.age.bins']) - 1
                )
                self['age.bins'] = header_dict['agetracer.age.bins']
                self['has.custom.age.bin'] = True
            else:
                print('! input header dict, but cannot make sense of age-tracer information')
                return

        elif age_bins is not None:
            # input custom age bins
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

        # ensure sane minimum and maximum of age bins [Myr]
        self['age.bins'][0] = self._age_min_impose
        if self['age.bins'][-1] > self._age_max_impose:
            self['age.bins'][-1] = self._age_max_impose

    def assign_element_yield_massfractions(
        self, element_yield_dict, _progenitor_metal_massfractions=None
    ):
        '''
        Assign to self a dictionary of stellar nucleosynthetic yield mass fractions within
        stellar age bins.
        Use to map the age-tracer mass weights in each age bin into elemental abundances
        (mass fractions).

        Parameters
        -----------
        element_yield_dict : dict of 1-D arrays
            nucleosynthetic yield fractional mass (relative to mass of stars at formation)
            of each element produced within each age bin
        _progenitor_metal_massfractions : array
            placeholder for future development
        '''
        for element_name in element_yield_dict:
            assert len(element_yield_dict[element_name]) == self['age.bin.number']
            self['yield.massfractions'][element_name] = np.array(element_yield_dict[element_name])

    def assign_element_initial_massfraction(
        self, massfraction_initial_dict=None, metallicity=None, helium_massfraction=0.24
    ):
        '''
        Set the initial conditions for the elemental abundances (mass fractions),
        to add to the age-tracer nucleosynthetic yields.
        You need to run assign_element_yield_massfractions() before this.
        If you do not run this, will assume that all particles started with initial mass fraction
        for all elements (beyond H) of 0.

        Parameters
        ----------
        massfraction_initial_dict : dict
            Keys are element names and values are the linear mass fraction for each element,
            to use as initial conditions (at early cosmic times) for each particle.
            Default to 0 for any element that is not in input massfraction_initial_dict.
        metallicity : float
            Linear total metallity relative to Solar.
            If defined, assume that input massfraction_initial_dict are Solar mass fractions,
            and multiply them by this value.
        helium_massfraction : float
            If defined, use this for the initial mass fraction of helium, over-writing any value in
            input massfraction_initial_dict.
        '''
        # sanity checks
        # need to set yield dictionary first, to know which elements tracking
        assert self['yield.massfractions'] is not None and len(self['yield.massfractions']) > 0
        if isinstance(massfraction_initial_dict, dict) and len(massfraction_initial_dict) > 0:
            for element_name in massfraction_initial_dict:
                assert element_name in self['yield.massfractions']
        if metallicity is not None:
            # ensure is linear mass fraction relative to Solar
            assert np.isscalar(metallicity) and metallicity >= 0

        # initialize to 0 for all elements in model, then over-write with input dict below
        for element_name in self['yield.massfractions']:
            self['initial.massfraction'][element_name] = 0

        if isinstance(massfraction_initial_dict, dict) and len(massfraction_initial_dict) > 0:
            for element_name in massfraction_initial_dict:
                self['initial.massfraction'][element_name] = massfraction_initial_dict[element_name]
                if metallicity is not None:
                    self['initial.massfraction'][element_name] *= metallicity
        else:
            print(
                '! not able to parse input massfraction_initial_dict'
                + ', setting initial mass fraction to 0 for all elements'
            )

        if helium_massfraction is not None and 'helium' in self['yield.massfractions']:
            self['initial.massfraction']['helium'] = helium_massfraction

    def get_element_massfractions(
        self, element_name, agetracer_mass_weights, _metal_massfractions=None
    ):
        '''
        Get the elemental abundances (mass fractions) for input element_name[s],
        using the the input 2-D array of age-tracer mass weights.

        Before you call this method, you must
            (1) set up the age bins via:
                assign_age_bins()
            (2) assign the nucleosynthetic yield mass fraction within each age bin for each element:
                assign_element_yield_massfractions()
            (3) (optional) assign the initial abundance (mass fraction) for each element via:
                assign_element_initial_massfraction()

        Parameters
        ----------
        element_name : str
            name of element to get mass fraction of for each particle
        agetracer_mass_weights : 2-D array (N_particle x N_age-bins)
            age-tracer mass weights for particles - should be values from
                part[species_name]['massfraction'][:, self['element.index.start']:],
                where species_name = 'star' or 'gas'
        _metal_massfractions : 1-D array
            placeholder for future development

        Returns
        -------
        element_mass_fractions : 1-D array
            mass fraction of element_name for each particle
        '''
        # check that input valid element name
        assert element_name
        element_name = parse_element_names(
            self['yield.massfractions'].keys(), element_name, scalarize=True
        )

        if np.ndim(agetracer_mass_weights) == 2:
            axis = 1  # input age-tracer weights for multiple particles
        elif np.ndim(agetracer_mass_weights) == 1:
            axis = 0  # input age-tracer weights for single particle

        # weight the yield within each age bin by the age-tracer mass weights
        # and sum across all age bins to get the total abundance
        element_mass_fractions = np.sum(
            agetracer_mass_weights * self['yield.massfractions'][element_name], axis=axis
        )

        # add initial abundances (if applicable)
        if len(self['initial.massfraction']):
            assert element_name in self['initial.massfraction']
            element_mass_fractions += self['initial.massfraction'][element_name]

        return element_mass_fractions


ElementAgeTracer = ElementAgeTracerClass()


class ElementAgeTracerZClass(ElementAgeTracerClass):
    '''
    Store and assign yields in bins of progenitor metallicity.
    '''

    def assign_element_yield_massfractions(
        self, element_yield_dict=None, progenitor_metal_massfractions=None
    ):
        '''
        Assign to self a dictionary of stellar nucleosynthetic yields within stellar age bins.

        Parameters
        -----------
        element_yield_dict : list of dicts of 1-D arrays
            nucleosynthetic yield fractional mass [M_sun per M_sun of stars formed] of each element
            produced within/across each age bin, to map the age-tracer mass weights in each age bin
            into actual element yields
        progenitor_metal_massfractions : array
        '''
        element_yield_dicts = element_yield_dict
        element_yield_dict = element_yield_dicts[0]
        element_name = tuple(element_yield_dict.keys())[0]
        for element_name in element_yield_dict:
            self['yield.massfractions'][element_name] = np.zeros(
                (progenitor_metal_massfractions.size, element_yield_dict[element_name].size),
                element_yield_dict[element_name].dtype,
            )

        for zi, _progenitor_metal_massfractions in enumerate(progenitor_metal_massfractions):
            element_yield_dict = element_yield_dicts[zi]
            for element_name in element_yield_dict:
                assert len(element_yield_dict[element_name]) == self['age.bin.number']
                self['yield.massfractions'][element_name][zi] = np.array(
                    element_yield_dict[element_name]
                )

        self['progenitor.metal.massfractions'] = progenitor_metal_massfractions

    def get_element_massfractions(
        self, element_name, agetracer_mass_weights, metal_massfractions=None
    ):
        '''
        Get the elemental abundances (mass fractions) for input element_name[s],
        using the the input 2-D array of age-tracer weights.

        Before you call this method, you must:
            (1) set up the age bins via:
                assign_age_bins()
            (2) assign the nucleosynthetic yield mass fraction within each age bin for each element:
                assign_element_yield_massfractions()
            (3) (optional) assign the initial abundance (mass fraction) for each element via:
                assign_element_initial_massfraction()

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
        # check that input valid element name
        assert element_name
        element_name = parse_element_names(
            self['yield.massfractions'].keys(), element_name, scalarize=True
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
                    agetracer_mass_weights[pis] * self['yield.massfractions'][element_name][zi],
                    axis=1,
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
    Provide stellar nucleosynthetic yields in the FIRE-2 or FIRE-3 model.

    In FIRE-2, the following yields and mass-loss rates depend on progenitor metallicity
        stellar winds: overall mass-loss rate and oxygen yield
        core-collapse supernovae: nitrogen yield
    '''

    def __init__(self, model='fire2', progenitor_metallicity=1.0):
        '''
        Parameters
        -----------
        model : str
            name for this rate + yield model: 'fire2', 'fire3'
        progenitor_metallicity : float
            metallicity [linear mass fraction relative to Solar]
            for nucleosynthetic rates and yields that depend on progenitor metallicity
        '''
        from . import gizmo_star

        self.model = model.lower()
        assert self.model in ['fire2', 'fire2.1', 'fire2.2', 'fire3']

        # this class computes the yields
        self.NucleosyntheticYield = gizmo_star.NucleosyntheticYieldClass(model)

        # these classes compute the rates
        self.SupernovaCC = gizmo_star.SupernovaCCClass(model)
        self.SupernovaIa = gizmo_star.SupernovaIaClass(model)
        self.StellarWind = gizmo_star.StellarWindClass(model)

        # store the Solar abundances
        self.sun_massfraction = self.NucleosyntheticYield.sun_massfraction

        # store list of names of elements tracked in this model
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
            'sulfur',
            'calcium',
            'iron',
        ]
        """

        # transition/discontinuous ages [Myr] in this model to be careful near integrating
        self.ages_transition = gizmo_star.get_ages_transition(model)
        # self.ages_transition = None

        # store this (default) progenitor metallicity, including the mass fraction for each element
        # scale all progenitor abundances to Solar ratios - this is not accurate in detail,
        # but it is a reasonable approximation
        self.progenitor_metallicity = progenitor_metallicity
        self.progenitor_massfraction_dict = {}
        for element_name in self.element_names:
            # scale to Solar abundance ratios
            self.progenitor_massfraction_dict[element_name] = (
                progenitor_metallicity * self.sun_massfraction[element_name]
            )

        # store all yields as mass fraction relative to mass of stars at formation
        # store because in FIRE-2 yields are independent of stellar age and ejecta/mass-loss rates
        self.NucleosyntheticYield.assign_element_yields(
            # match FIRE-2
            progenitor_massfraction_dict=self.progenitor_massfraction_dict,
            # test: do not model correction of yields from pre-existing surface abundances
            # progenitor_metallicity=self.progenitor_metallicity,
            # progenitor_massfraction_dict=None,
        )

    def get_element_yields(self, age_bins, element_names=None):
        '''
        Construct and return a dictionary of stellar nucleosynthetic yields.
        * Each key is an element name.
        * Each key value is a 1-D array of yields within each input age bin,
          constructed by integrating the total yield for each element across each age bin.

        Supply the returned element_yield_dict to
            ElementAgeTracerClass.assign_element_yield_massfractions()
        to assign elemental abundances to particles via the age-tracer mass weights in a Gizmo
        simulation.

        Parameters
        -----------
        age_bins : array
            stellar age bins used in Gizmo for the age-tracer model [Myr]
            Should have N_age-bins + 1 values: left edges plus right edge of final bin.
        element_names : list
            names of elements to generate, if only generating a subset
            If input None, assign all elements in this model.

        Returns
        -------
        element_yield_dict : dict of 1-D arrays
            fractional mass (relative to mass of stars at formation) of each element produced
            within each age bin
        '''
        # if input element_names is None, generate yields for all elements in this model
        element_names = parse_element_names(self.element_names, element_names, scalarize=False)

        # initialize main dictionary
        element_yield_dict = {}
        for element_name in element_names:
            element_yield_dict[element_name] = np.zeros(np.size(age_bins) - 1)

        # ages to be careful around during integration
        if not hasattr(self, 'ages_transition'):
            self.ages_transition = None

        # compile yields within/across each age bin by integrating over the assumed rates
        for ai in np.arange(np.size(age_bins) - 1):
            age_min = age_bins[ai]
            if ai == 0:
                age_min = 0  # ensure min age starts at 0
            age_max = age_bins[ai + 1]

            for element_name in element_names:
                # get the integrated yield mass within/across the age bin
                element_yield_dict[element_name][ai] = integrate.quad(
                    self._get_element_yield_rate,
                    age_min,
                    age_max,
                    (element_name,),
                    points=self.ages_transition,
                )[0]

        return element_yield_dict

    def _get_element_yield_rate(self, age, element_name, progenitor_metallicity=None):
        '''
        Return the specific rate of nucleosynthetic yield (yield mass relative to mass of stars at
        formation) [Myr ^ -1] at input stellar age [Myr] for input element_name,
        from all stellar processes.
        get_element_yields() uses this to integrate across age within each age bin.

        Parameters
        ----------
        age : float
            stellar age [Myr] at which to compute the nucleosynthetic yield rate
        element_name : str
            name of element, must be in self.element_names
        progenitor_metallicity : float
            metallicity [linear mass fraction relative to Solar]
            In FIRE-3 and FIRE-3, this determines the mass-loss rate of stellar winds.

        Returns
        -------
        element_yield_rate : float
            specific rate (yield mass relative to mass of stars at formation) [Myr ^ -1] at input
            age for input element_name
        '''
        if progenitor_metallicity is None:
            progenitor_metallicity = self.progenitor_metallicity

        # rates below are fractional mass rate relative to mass of stars at formation [Myr ^ -1]
        # wind is the only mass loss rate in FIRE-2 and FIRE-3 that depends on metallicity
        wind_rate = self.StellarWind.get_mass_loss_rate(age, metallicity=progenitor_metallicity)
        sncc_rate = self.SupernovaCC.get_mass_loss_rate(age)
        snia_rate = self.SupernovaIa.get_mass_loss_rate(age)

        element_yield_rate = (
            self.NucleosyntheticYield['wind'][element_name] * wind_rate
            + self.NucleosyntheticYield['supernova.cc'][element_name] * sncc_rate
            + self.NucleosyntheticYield['supernova.ia'][element_name] * snia_rate
        )

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

    def get_element_yields(self, t, element):
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
