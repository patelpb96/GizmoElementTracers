'''
Assign elemental abundances to star particles and gas cells in post-processing using age-tracer
mass weights stored in Gizmo simulation snapshots.

If a Gizmo simulation included age-tracers (by defining GALSF_FB_FIRE_AGE_TRACERS in Config.sh),
each gas cell stores an array of mass weights in bins of stellar age,
contributed to the gas cell from any neighboring star particles (within the kernel radius).
These weights record the fraction of mass from ejecta/winds that would have been deposited into the
gas cell had there been a stellar enrichment event at that timestep.
In other words, a star particle of a given age deposits a mass weight into a corresponding stellar
age bin in all of its neighboring gas cells.

The age-tracer weights are dimensionless, in units of mass fraction.
Specifically, each injected weight is: the total mass of the star particle at that timestep,
multiplied by the geometric fraction of the wind/ejecta mass that would have been deposited into
the gas cell, divided by the mass of the gas cell at that timestep.
An age-tracer weight of 1.0 in an age bin means that the gas cell received winds/ejecta
from 1 or several star particle[s] equal to the gas cell's own mass for a duration that spans
the age bin. Thus, to convert an age-tracer mass weight to an element abundance (mass fraction),
multiply the age-tracer weight by the nucleosynthetic yield mass of an element produced/injected
per IMF-averaged mass of stars, integrated across an age bin.

A star particle then inherits the array of age-tracer weights from its progenitor gas cell.

Assigning elemental abundances to particles at any snapshot then requires you to compile the
total nucleosynthetic yield mass fractions (relative to the IMF-averaged mass of stars)
from all stellar processes within/across each age-tracer stellar age bin for each element.
Store this as a dictionary of 1-D arrays, where each array contain the total mass fraction
(relative to the IMF-averaged mass of stars) that a stellar population produces within/across each
age bin, and where each dictionary key corresponds to an element.
You can construct this dictionary using a default rate + yield model below, or you can define your
own custom rate and yield model that accepts stellar age[s] and an element name as inputs and
returns the instantaneous fractional mass-loss rate of that element at that stellar age.
Then supply this dictionary to ElementAgeTracerClass.assign_element_yield_massfractions().

For reference, these are the relevant age-tracer settings in Gizmo configuation and parameter files
    in Config.sh
        GALSF_FB_FIRE_AGE_TRACERS
            Master switch that turns on age-tracers and sets the number
            of age bins, which by default are equally spaced in log age.
        GALSF_FB_FIRE_AGE_TRACERS_CUSTOM
            Instead, read a custom list of arbitrary age bins
        GALSF_FB_FIRE_AGE_TRACERS_DISABLE_SURFACE_YIELDS
            Disable surface return of age tracers, that is, inject into a gas cell only the
            mass corresponding to *new* nucleosynthesis in that star particle, so ignore any
            pre-existing metals in the star particle.
            By definition, this does not conserve age-tracer weights as a simulation progresses.
    in gizmo_parameters.txt
        AgeTracerBinStart
            Minimum age of age bins (if not custom list) [Myr]
        AgeTracerBinEnd
            Maximum age of age bins (if not custom list) [Myr]
        AgeTracerListFilename
            Name of text file that contains custom age bins
        AgeTracerActiveTimestepFraction
            (stored as AgeTracerRateNormalization within Gizmo)
            Targeted number of deposition events per age-tracer stellar age bin.
            If <= 0, deposit age-tracer weights at each particle timestep.

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

TDD_DEFAULT = -1.1 # default Maoz time delay distribution
NIA_DEFAULT = 2.6e-7 # default Maoz Ia prefactor
NUM_IA_FIDUCIAL = 0.002257708396415009 # fiducial integrated massloss rate for SNe Ia
NUM_CC_FIDUCIAL = 0.11152830837937755 # fiducial integrated massloss rate for CCSN

# --------------------------------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------------------------------
def parse_element_names(element_names_possible, element_names=None, scalarize=False):
    '''
    Utility function to parse the input element names.

    Parameters
    ----------
    element_names_possible : list of str
        possible names of elements in a given model
    element_names : str or list
        name[s] of element[s]
    scalarize : bool
        whether to return single string (instead of list) if input single string

    Returns
    -------
    element_names_safe : list or str
        name[s] of element[s]
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
            'fire2.1' is 'fire2' but removes prgenitor metallicity dependence to all yields
            'fire2.2' is 'fire2.1' but also removes progenitor metallicity dependence to
            stellar wind mass-loss rates
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
        self.SupernovaWD = gizmo_star.SupernovaWDClass(model)
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

        # store all yields as mass fraction relative to the IMF-averaged mass of stars at that time
        # can store because in FIRE-2 yields are independent of stellar age and mass-loss rate
        self.NucleosyntheticYield.assign_element_yields(
            # match FIRE-2
            progenitor_massfraction_dict=self.progenitor_massfraction_dict,
            # test: do not model correction of yields from pre-existing surface abundances
            # progenitor_metallicity=self.progenitor_metallicity,
            # progenitor_massfraction_dict=None,
        )

    def get_element_yields(self, age_bins, element_names=None, testing = False):
        '''
        Construct and return a dictionary of stellar nucleosynthetic yields.
        * Each key is an element name
        * Each key value is a 1-D array of yields within each input age bin,
          constructed by integrating the total yield for each element across each age bin.

        Supply the returned element_yield_dict to
            ElementAgeTracerClass.assign_element_yield_massfractions()
        to assign elemental abundances to star particles or gas cells via the age-tracer mass
        weights ina Gizmo simulation.

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
            fractional mass (relative to IMF-averaged mass of stars at that time) of each element
            produced within each age bin
        '''
        # if input element_names is None, generate yields for all elements in this model
        element_names = parse_element_names(self.element_names, element_names, scalarize=False)
        test = testing

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
                    (element_name,test),
                    points=self.ages_transition
                )[0]

        #print(self.ages_transition)
        return element_yield_dict

    def _get_element_yield_rate(self, age, element_name, test = False, progenitor_metallicity=None):
        '''
        Return the specific rate of nucleosynthetic yield (yield mass relative to IMF-averaged mass
        of stars at that time) [Myr ^ -1] at input stellar age [Myr] for input element_name,
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
            In FIRE-2 and FIRE-3, this determines the mass-loss rate of stellar winds.

        Returns
        -------
        element_yield_rate : float
            specific rate (yield mass relative to IMF-averaged mass of stars at that time)
            [Myr ^ -1] at input age for input element_name
        '''
        if progenitor_metallicity is None:
            progenitor_metallicity = self.progenitor_metallicity

        # rates below are fractional mass rate relative to mass of stars at formation [Myr ^ -1]
        # wind is the only mass loss rate in FIRE-2 and FIRE-3 that depends on metallicity
        wind_rate = self.StellarWind.get_mass_loss_rate(age, metallicity=progenitor_metallicity)
        sncc_rate = self.SupernovaCC.get_mass_loss_rate(age)
        snwd_rate = self.SupernovaWD.get_mass_loss_rate(age)

        if test:
            if test == 'winds':
                return self.NucleosyntheticYield['wind'][element_name] * wind_rate
            
            if test == 'cc':
                return self.NucleosyntheticYield['supernova.cc'][element_name] * sncc_rate

            if test == 'ia':
                return self.NucleosyntheticYield['supernova.wd'][element_name] * snwd_rate
            
            if test == 'no_winds':
                eyr = (self.NucleosyntheticYield['supernova.cc'][element_name] * sncc_rate + self.NucleosyntheticYield['supernova.wd'][element_name] * snwd_rate)
                return eyr
            
            if test == 'no_cc':
                eyr = (self.NucleosyntheticYield['wind'][element_name] * wind_rate + self.NucleosyntheticYield['supernova.wd'][element_name] * snwd_rate)
                return eyr

            if test == 'no_wd':
                eyr = (self.NucleosyntheticYield['wind'][element_name] * wind_rate + self.NucleosyntheticYield['supernova.cc'][element_name] * sncc_rate)
                return eyr

        element_yield_rate = (
            self.NucleosyntheticYield['wind'][element_name] * wind_rate
            + self.NucleosyntheticYield['supernova.cc'][element_name] * sncc_rate
            + self.NucleosyntheticYield['supernova.wd'][element_name] * snwd_rate
        )

        return element_yield_rate

class FIREYieldClass2:
    '''
    Modified FIREYieldClass to work with Preet's implementation of the FIRE-2 model. 
    '''

    def __init__(self, model='fire2.1', 
                 progenitor_metallicity=1.0, 
                 
                 # WDSN/SNe Ia parameters
                 ia_type = 'maoz', 
                 trans_time_ia = [37.53], 
                 tdd_ia = TDD_DEFAULT, 
                 normalization_ia = NIA_DEFAULT,

                 # CCSN parameters
                 normalization_ccsn = [0, 5.408e-4, 2.516e-4, 0],
                 trans_time_ccsn = [3.4, 10.37, 37.53],

                 # Winds parameters
                 normalization_winds = False,
                 trans_time_winds = [1.0, 3.5, 100]
                 ):
        '''
        Parameters
        -----------
        model : str
            name for this rate + yield model: 'fire2', 'fire3'
            'fire2.1' is 'fire2' but removes prgenitor metallicity dependence to all yields
            'fire2.2' is 'fire2.1' but also removes progenitor metallicity dependence to
            stellar wind mass-loss rates
        progenitor_metallicity : float
            metallicity [linear mass fraction relative to Solar]
            for nucleosynthetic rates and yields that depend on progenitor metallicity
        ia_type : string {'maoz', 'mannucci'}
            choice of feedback model for type Ia's/WDSN
        trans_time_ia : list[float()]
            transition time(s) for ia model
        tdd_ia : float
            default value for maoz time-delay distribution exponent
        normalization_ia : float
            default value for maoz model normalization constant
        '''
        from . import gizmo_star
        from . import gizmo_model

        self.model = model.lower()
        self.gizmo_model = gizmo_model

        # For use with Agetracer grid 
        self.ia_model = ia_type
        self.ia_transition_time = trans_time_ia
        self.ia_normalization = normalization_ia
        self.ia_tdd = tdd_ia
        self.cc_transition_time = trans_time_ccsn
        self.cc_normalization = normalization_ccsn
        self.winds_transition_time = trans_time_winds
        self.winds_normalization = normalization_winds

        #print("Self.ia_model = " + str(ia_type))
        
        assert self.model in ['fire2', 'fire2.1', 'fire2.2', 'fire3']

        # this class computes the yields
        #self.NucleosyntheticYield = gizmo_star.NucleosyntheticYieldClass(model)
        self.NucleosyntheticYield = gizmo_model.nucleosyntheticYieldDict
        #print("Type NucleosyntheticYield: " + str(type(self.NucleosyntheticYield)))
        #print(self.NucleosyntheticYield)

        # these classes compute the rates
        #self.SupernovaCC = gizmo_model.feedback(source = 'cc')
        #self.SupernovaIa = gizmo_model.feedback(source = 'ia', ia_model = ia_type)
        #self.StellarWind = gizmo_model.feedback(source = 'wind')

        #self.rate_cc, self.age_cc, self.trans_cc = self.SupernovaCC.get_rate_cc()
        #self.rate_ia, self.age_ia, self.trans_ia = self.SupernovaIa.get_rate_wd()
        #self.rate_wind, self.age_wind, self.trans_wind = self.StellarWind.get_rate_wind()
        

        # store the Solar abundances
        self.sun_massfraction = gizmo_model.get_sun_massfraction()

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
        #self.ages_transition = np.array([1, 3.4, 3.5, 10.37, 37.53, 50, 100, 1000])
        self.ages_transition = np.unique(np.concatenate([self.winds_transition_time, self.cc_transition_time, self.winds_transition_time, np.array([50,1000])]))

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

        # store all yields as mass fraction relative to the IMF-averaged mass of stars at that time
        # can store because in FIRE-2 yields are independent of stellar age and mass-loss rate
            
    def get_element_yields(self, 
                           age_bins, 
                           element_names=None, 
                           continuous = False, 
                           old_int = False, 
                           fast_int = False, 
                           int_error = 1.5e-8, 
                           lim = 10, 
                           testing = False):
        '''
        Construct and return a dictionary of stellar nucleosynthetic yields.
        * Each key is an element name
        * Each key value is a 1-D array of yields within each input age bin,
          constructed by integrating the total yield for each element across each age bin.

        Supply the returned element_yield_dict to
            ElementAgeTracerClass.assign_element_yield_massfractions()
        to assign elemental abundances to star or gas particles via the age-tracer mass weights in
        a Gizmo simulation.

        Parameters
        -----------
        age_bins : array
            stellar age bins used in Gizmo for the age-tracer model [Myr]
            Should have N_age-bins + 1 values: left edges plus right edge of final bin.
        element_names : list
            names of elements to generate, if only generating a subset
            If input None, assign all elements in this model.
        continuous : bool
            determines type of integration
        old_int : bool
            also used for type of integration (deprecated)
        fast_int : bool
            when yields are fixed (i.e. no progenitor metallicity dependence), this speeds up the integrator.
        int_error : float
            scipy integration has some threshold for error when performing integration, and this value corresponds to the best I could find
        testing : bool {False}, string {'cc', 'wd', 'wind'}
            False when not isolating feedback channels, otherwise string 

        Returns
        -------
        element_yield_dict : dict of 1-D arrays
            fractional mass (relative to IMF-averaged mass of stars at that time) of each element
            produced within each age bin
        '''

        # if input element_names is None, generate yields for all elements in this model
        element_names = parse_element_names(self.element_names, element_names, scalarize=False)
        test = testing
        #print("element names: " + str(element_names))
        #print("element names type: " + str(type(element_names)))

        #print("continuous: " + str(continuous))
        #print("fast_integ: " + str(fast_int)) # refers to the simplifcation of the yield dictionary. 

        # initialize main dictionary
        element_yield_dict = {}
        for element_name in element_names:
            element_yield_dict[element_name] = np.zeros(np.size(age_bins) - 1)

        # ages to be careful around during integration
        if not hasattr(self, 'ages_transition'):
            print("FATAL ERROR !! ")
            self.ages_transition = None

        for ai in np.arange(np.size(age_bins) - 1):
            age_min = age_bins[ai]
            #print("For ai: " + str(ai) + "| For age_min: " + str(age_min) )
            if ai == 0:
                age_min = 0  # ensure min age starts arbitrarily close to 0
            age_max = age_bins[ai + 1]

            if continuous == True:

                for element_name in element_names:
                    # get the integrated yield mass within/across the age bin
                    element_yield_dict[element_name][ai] = integrate.quad(
                        self._feedback_handler,
                        age_min,
                        age_max,
                        args = (element_name, test),
                        points = self.ages_transition,
                        epsabs = int_error,
                        limit = lim
                    )[0]

                    #c = self._feedback_handler(age_min, element_name, test)
                    #log = str(c) + ",FY2," +str(ai) + "," + str(element_name)
                    #print(log)
                    
            if continuous == False:
                # if we want to use pre-tabulated values to integrate over instead of calculating the function at each relevant step
                # speedtests show this is always the slower option for the ElementTracers, but saving your data beforehand instead of 
                # calculating it each time CAN be useful in certain instances.

                for element_name in element_names:
                    # get the integrated yield mass within/across the age bin
                    #print(age_min, age_max)

                    r_ia, a_ia, t_ia = self.gizmo_model.feedback(source = 'ia', 
                                                                 elem_name = element_name, 
                                                                 ia_model=self.ia_model, 
                                                                 t_ia = self.ia_transition_time, 
                                                                 n_ia = self.ia_normalization).get_rate_wd()
                    
                    mask = np.logical_and(age_min <= a_ia, a_ia <= age_max)
                    int_ia = integrate.trapz(r_ia[mask]/len(r_ia[mask]), x = [age_min, age_max])#, a_ia[mask])

                    r_cc, a_cc, t_cc = self.gizmo_model.feedback(source = 'cc', elem_name = element_name).get_rate_cc()
                    mask = np.logical_and(age_min <= a_cc, a_cc <= age_max)
                    int_cc = integrate.trapz(r_cc[mask]/len(r_cc[mask]), x = [age_min, age_max])#, a_cc[mask])

                    r_w, a_w, t_w = self.gizmo_model.feedback(source = 'wind', elem_name = element_name).get_rate_wind()
                    mask = np.logical_and(age_min <= a_w, a_w <= age_max)
                    int_w = integrate.trapz(r_w[mask]/len(r_w[mask]), x = [age_min, age_max])#, a_w[mask])

                    element_yield_dict[element_name][ai] = int_ia + int_w + int_cc

            if continuous and fast_int:
                #If we aren't concerned about inter-metal dependencies/progenitor metallicity dependence, we can pre-compute the integrals and multiply by yields for faster computation
                int_w = integrate.quad(
                    self._feedback_handler,
                    age_min,
                    age_max,
                    args = (False,'winds'),
                    points = self.ages_transition,
                    epsabs = int_error,
                    limit = lim
                )[0]
                int_cc = integrate.quad(
                    self._feedback_handler,
                    age_min,
                    age_max,
                    args = (False,'cc'),
                    points = self.ages_transition,
                    epsabs = int_error,
                    limit = lim
                )[0]
                int_ia = integrate.quad(
                    self._feedback_handler,
                    age_min,
                    age_max,
                    args = (False,'ia'),
                    points = self.ages_transition,
                    epsabs = int_error,
                    limit = lim
                )[0]

                # normalization constant which preserves the total number of Ia events.
                A_wd = NUM_IA_FIDUCIAL/int_ia[-1]
                A_cc = NUM_CC_FIDUCIAL/int_cc[-1]

                for element_name in element_names:
                    # get the integrated yield mass within/across the age bin
                    element_yield_dict[element_name][ai] = A_wd*self.gizmo_model.element_yield_ia[element_name]*int_ia + self.gizmo_model.element_yield_wind[element_name]*int_w + self.gizmo_model.element_yield_cc[element_name]*int_cc

        return element_yield_dict

    def _feedback_handler(self, some_time, element_of_choice, test_process = False):
        #print("eoc: " + str(element_of_choice))
    
        element_name = element_of_choice

        if test_process == False:
            r_ia = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'ia', ia_model=self.ia_model, t_ia = self.ia_transition_time, n_ia = self.ia_normalization, t_dd = self.ia_tdd).get_rate_wd()[0]
            r_cc = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'cc', n_cc = self.cc_normalization, t_cc = self.cc_transition_time).get_rate_cc()[0]
            r_w = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'wind', n_w=self.winds_normalization, t_w=self.winds_transition_time).get_rate_wind()[0]

            return (r_ia + r_cc + r_w)
        
        if test_process == 'default_wd':
            r_ia = self.gizmo_model.feedback(time_span = [some_time], 
                                             elem_name=element_name, 
                                             source = 'ia', ia_model=self.ia_model, 
                                             t_ia = self.ia_transition_time, 
                                             n_ia = NIA_DEFAULT, 
                                             t_dd = TDD_DEFAULT).get_rate_wd()[0]

        if test_process == 'winds' or test_process == 'wind':
            r_w, a_w, t_w = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name,source = 'wind', n_w=self.winds_normalization, t_w=self.winds_transition_time).get_rate_wind()
            return r_w
            #print("TESTING WINDS")

        if test_process == 'cc' or test_process == 'ccsn':
            r_cc, a_cc, t_cc = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name,source = 'cc', n_cc = self.cc_normalization, t_cc = self.cc_transition_time).get_rate_cc()
            return r_cc
            #print("TESTING CCSN")

        if test_process == 'ia' or test_process == 'wd' or test_process == 'wdsn':
            r_ia = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'ia', ia_model=self.ia_model, t_ia = self.ia_transition_time, n_ia = self.ia_normalization, t_dd = self.ia_tdd).get_rate_wd()[0]
            return r_ia
            #print("TESTING IA")

        '''
        #Switches for speed, only works on Python 3.10+. Not functional on stampede2, functional on peloton.
        if test_process:
            # Lots of reasons to use the test parameter, not sure if I should rename to 'specify' or something like that. 
            match test_process:
                case 'default_wd': # calculate with fiducial Maoz rate
                    r_ia = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'ia', ia_model=self.ia_model, t_ia = self.ia_transition_time, n_ia = NIA_DEFAULT, t_dd = TDD_DEFAULT).get_rate_wd()[0]

                case 'wind':
                    r_w, a_w, t_w = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name,source = 'wind').get_rate_wind()
                    return r_w
                    #print("TESTING WINDS")

                case 'ccsn':
                    r_cc, a_cc, t_cc = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name,source = 'cc').get_rate_cc()
                    return r_cc
                    #print("TESTING CCSN")

                case 'wd':
                    r_ia = self.gizmo_model.feedback(time_span = [some_time], elem_name=element_name, source = 'ia', ia_model=self.ia_model, t_ia = self.ia_transition_time, n_ia = self.ia_normalization, t_dd = self.ia_tdd).get_rate_wd()[0]
                    return r_ia
                    #print("TESTING IA")
        '''

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
            import sygma

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


# --------------------------------------------------------------------------------------------------
# class to use age-tracer weights to assign elemental abundances
# --------------------------------------------------------------------------------------------------
class ElementAgeTracerClass(dict):
    '''
    Class to store and use the age-tracer array of mass weights from a Gizmo simulation to assign
    elemental abundances to star particles and gas cells in post-processing.
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
        # this only affects the limits in integrating the yield rates across each stellar age bin
        self._age_min_impose = 0
        self._age_max_impose = 13700

        # whether using custom stellar age bins - default (non-custom) is equally spaced in log age
        self['has.custom.age.bin'] = False
        # targeted number of (sampled) injection events per stellar age bin when the simulation ran
        self['event.number.per.age.bin'] = None
        # number of stellar age bins
        self['age.bin.number'] = None
        # array of ages of bin edges [Myr]
        # should have N_age-bins + 1 values: left edges plus right edge of final bin
        self['age.bins'] = None
        # starting index of age-tracer fields within Gizmo's particle massfraction array
        self['element.index.start'] = None
        # dictionary of nucleosynthetic yield mass fractions for each element within each age bin
        self['yield.massfractions'] = {}
        # dictionary to store initial conditions of elemental abundances at simulation start
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
            use this to assign age-tracer stellar age bin information
        age_bins : array
            age bins [Myr], with N_age-bins + 1 values: left edges plus right edge of final bin
        age_bin_number : int
            number of age bins
        age_min : float
            minimum age (left edge of first bin) [Myr] (over-ride this to be self._age_min_impose)
        age_max : float
            maximum age (right edge of final bin) [Myr] (over-ride this to be self._age_max_impose)
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
        self, element_yield_dict, _progenitor_metal_massfractions=None, flush = False
    ):
        '''
        Assign to self a dictionary of stellar nucleosynthetic yield mass fractions within
        stellar age bins.
        Use to map (multiply) the age-tracer mass weights in each age bin into elemental abundances
        (mass fractions).

        Parameters
        -----------
        element_yield_dict : dict of 1-D arrays
            nucleosynthetic yield fractional mass (relative to IMF-averaged mass of stars at that
            time) of each element produced within each age bin
        _progenitor_metal_massfractions : array
            placeholder for future development
        '''

        if flush == True:
            for element_name in element_yield_dict:
                try:
                    del self['yield.massfractions'][element_name]
                except:
                    print("Dictionary check passed")
                    break

        for element_name in element_yield_dict:
            assert len(element_yield_dict[element_name]) == self['age.bin.number']
            self['yield.massfractions'][element_name] = np.array(element_yield_dict[element_name])


    def assign_element_initial_massfraction(
        self, massfraction_initial_dict=None, metallicity=None, helium_massfraction=0.24
    ):
        '''
        Set the initial conditions for the elemental abundances (mass fractions),
        to add to the age-tracer nucleosynthetic yields.
        You need to run assign_element_yield_massfractions() before calling this.
        If you do not call this, will assume that all particles started with initial mass fraction
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
        Get the actual elemental abundances (mass fractions) for input element_name[s],
        using the the input 2-D array of age-tracer mass weights.

        Before you call this method, you must
            (1) set up the age bins via
                assign_age_bins()
            (2) assign the nucleosynthetic yield mass fraction within each age bin for each element
                assign_element_yield_massfractions()
            (3) (optional) assign the initial abundance (mass fraction) for each element via
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

        # agetracer_mass_weights : come from the simulation
        # self['yield.massfractions'][element_name] : calculated from feedback models 

        element_mass_fractions = np.sum(
            agetracer_mass_weights * self['yield.massfractions'][element_name], axis=axis
        )

        # add initial abundances (if applicable)
        if len(self['initial.massfraction']):
            assert element_name in self['initial.massfraction']
            element_mass_fractions += self['initial.massfraction'][element_name]

        return element_mass_fractions


class ElementAgeTracerZClass(ElementAgeTracerClass):
    '''
    EXPERIMENTAL: Store and assign yields in bins of progenitor metallicity.
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
# master function to set up age-tracers in particle catalog
# --------------------------------------------------------------------------------------------------
def initialize_agetracers(
    parts,
    species_names=['star', 'gas'],
    progenitor_metallicity=0.6,
    metallicity_initial=1e-5,
    yield_model=None,
    YieldClass=FIREYieldClass,
):
    '''
    Master function to set up age-tracer information in particle catalog[s].

    Parameters
    ----------
    parts : dict or list
        catalogs[s] of particles at a snapshot
    species_names : str or list
        name[s] of particle species to assign age-tracers to: 'star', 'gas'
    progenitor_metallicity : float
        metallicity to assume for all progenitor stars in calculating nucleosynthetic yields
    metallicity_initial : float
        initial metallicity to assign to 'unenriched' gas cells at the start of the simulation
        [linear mass fraction relative to Solar]
    yield_model : str
        model to use for yields: 'fire2', 'fire2.1', 'fire2.2'
    YieldClass : class
        class to calculate nucleosynthetic yields in age bins
    '''
    # ensure lists
    if isinstance(parts, dict):
        parts = [parts]
    if isinstance(species_names, str):
        species_names = [species_names]

    for part in parts:
        if yield_model is not None and len(yield_model) > 0:
            yield_model_use = yield_model
        elif part.info['fire.model'] is not None and len(part.info['fire.model']) > 0:
            yield_model_use = part.info['fire.model']
        else:
            raise ValueError('no input yield_model and none stored in particle catalog')

        print(f'using {yield_model_use} yield model for age-tracers')

        # generate nucleosynthetic yield model
        Yield = YieldClass(yield_model_use, progenitor_metallicity=progenitor_metallicity)
        spec_name = species_names[0]
        # get dictionary of nucleosynthetic yield mass fractions in stellar age bins
        yield_dict = Yield.get_element_yields(part[spec_name].ElementAgeTracer['age.bins'])

        for spec_name in species_names:
            # initialize age-tracer stellar age bins
            part[spec_name].ElementAgeTracer = ElementAgeTracerClass(part.info)
            # transfer yields to ElementAgeTracer
            part[spec_name].ElementAgeTracer.assign_element_yield_massfractions(yield_dict)
            # set initial conditions for elemental mass fractions at start of simulation
            part[spec_name].ElementAgeTracer.assign_element_initial_massfraction(
                Yield.sun_massfraction, metallicity_initial
            )
