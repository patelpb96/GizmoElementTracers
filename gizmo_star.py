'''
Contains the models for stellar evolution as implemented in Gizmo for the FIRE-2 and FIRE-3 models,
specifically, nucleosynthetic yields and mass-loss rates for
    (1) core-collapse supernovae
    (2) Ia supernovae
    (3) stellar winds

The following nucleosynthetic yields and mass-loss rates depend on progenitor metallicity
    FIRE-2
        stellar wind: overall mass-loss rate, oxygen yield
        core-collapse supernova: nitrogen yield
    FIRE-3
        stellar wind: overall mass-loss rate, yields for He, C, N, O

----------
@author: Andrew Wetzel <arwetzel@gmail.com>

----------
Units

Unless otherwise noted, all quantities are in (combinations of)
    mass [M_sun]
    time [Myr] (note: most other modules in this GizmoAnalysis package default to Gyr)
    elemental abundance [linear mass fraction]
'''

import os
import collections
import pickle
import numpy as np
from scipy import integrate

import utilities as ut

# default model for stellar evolution rates and yields to assume throughout
FIRE_MODEL_DEFAULT = 'fire2'


# --------------------------------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------------------------------
def get_sun_massfraction(model=FIRE_MODEL_DEFAULT):
    '''
    Get dictionary of Solar abundances (mass fractions) for the elements that Gizmo tracks.
    (These mass fractions may differ by up to a percent from the values in utilities.constant,
    given choices of mean atomic mass.)

    Parameters
    ----------
    model : str
        stellar evolution model: 'fire2', 'fire3'
    '''

    model = model.lower()
    assert model in ['fire2', 'fire2.1', 'fire2.2', 'fire3']

    sun_massfraction = collections.OrderedDict()

    if 'fire2' in model:
        # FIRE-2 uses Anders & Grevesse 1989 for Solar
        sun_massfraction['metals'] = 0.02  # total of all metals (everything not H, He)
        sun_massfraction['helium'] = 0.28
        sun_massfraction['carbon'] = 3.26e-3
        sun_massfraction['nitrogen'] = 1.32e-3
        sun_massfraction['oxygen'] = 8.65e-3
        sun_massfraction['neon'] = 2.22e-3
        sun_massfraction['magnesium'] = 9.31e-4
        sun_massfraction['silicon'] = 1.08e-3
        sun_massfraction['sulfur'] = 6.44e-4
        sun_massfraction['calcium'] = 1.01e-4
        sun_massfraction['iron'] = 1.73e-3

    elif model == 'fire3':
        # FIRE-3 uses Asplund et al 2009 proto-solar for Solar
        sun_massfraction['metals'] = 0.0142  # total of all metals (everything not H, He)
        sun_massfraction['helium'] = 0.2703
        sun_massfraction['carbon'] = 2.53e-3
        sun_massfraction['nitrogen'] = 7.41e-4
        sun_massfraction['oxygen'] = 6.13e-3
        sun_massfraction['neon'] = 1.34e-3
        sun_massfraction['magnesium'] = 7.57e-4
        sun_massfraction['silicon'] = 7.12e-4
        sun_massfraction['sulfur'] = 3.31e-4
        sun_massfraction['calcium'] = 6.87e-5
        sun_massfraction['iron'] = 1.38e-3

    return sun_massfraction


def get_ages_transition(model=FIRE_MODEL_DEFAULT):
    '''
    Get array of ages [Myr] that mark transitions in stellar evolution for a given model.
    Use to supply to numerical integrators.

    Parameters
    ----------
    model : str
        stellar evolution model: 'fire2', 'fire3'
    '''
    model = model.lower()
    assert model in ['fire2', 'fire2.1', 'fire2.2', 'fire3']

    if 'fire2' in model:
        ages_transition = np.sort([1.0, 3.4, 3.5, 10.37, 37.53, 50, 100, 1000])  # [Myr]
    elif 'fire3' in model:
        ages_transition = np.sort([1.7, 3.7, 4.0, 6.5, 8, 18, 20, 30, 44, 1000])  # [Myr]

    return ages_transition


# --------------------------------------------------------------------------------------------------
# nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class NucleosyntheticYieldClass(dict):
    '''
    Nucleosynthetic yields in the FIRE-2 or FIRE-3 models.

    Yields that depend on Progenitor metallicity:
        FIRE-2
            stellar winds: oxygen
            core-collpase supernova: nitgrogen
        FIRE-3
            stellar winds: He, C, N, O

    Model variants for FIRE-2
         'fire2.1' = remove dependence on progenitor metallicity for all yields
            (via setting all progenitors to Solar)
         'fire2.2' = above + turn off dependence on progenitor metallicity for stellar wind rate
            (via setting all progenitors to Solar)
    '''

    def __init__(self, model=FIRE_MODEL_DEFAULT):
        '''
        Store Solar elemental abundances, as linear mass fractions.

        FIRE-2 uses Solar values from Anders & Grevesse 1989.
        FIRE-3 uses proto-Solar values from Asplund et al 2009.

        Parameters
        ----------
        model : str
            stellar evolution model for yields: 'fire2', 'fire3'
        '''
        self._event_kinds = ['wind', 'supernova.cc', 'supernova.ia']
        self._models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire3']
        self.model = None
        self.sun_massfraction = None
        self._parse_model(model)

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            stellar evolution model for yields: 'fire2', 'fire3'
        '''
        reset_parameters = False

        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                reset_parameters = True
            self.model = model

        assert self.model in self._models_available

        if reset_parameters:
            self.sun_massfraction = get_sun_massfraction(self.model)  # reset Solar abundances

    def get_element_yields(
        self,
        event_kind='supernova.cc',
        progenitor_metallicity=1.0,
        progenitor_massfraction_dict={},
        age=None,
        model=None,
        return_mass=False,
    ):
        '''
        Get dictionary of stellar nucleosynthetic yields for a single event_kind event
        in the FIRE-2 or FIRE-3 model.
        Return each yield as a mass fraction [mass of element / total ejecta mass] if return_mass
        is False (default), else return mass of element [Msun] if return_mass is True.
        Stellar wind yields are always intrinsically mass fractions [wrt wind mass].

        For stellar winds, Gizmo adds the existing surface abundances in the progenitor
        for elements not included in its yield.

        Parameters
        ----------
        event_kind : str
            stellar event: 'wind', 'supernova.cc' or 'supernova.ii', 'supernova.ia'
        progenitor_metallicity : float
            total metallicity of progenitor [linear mass fraction wrt sun_mass_fraction['metals']]
        progenitor_massfraction_dict : dict or bool [optional]
            optional: dictionary that contains the mass fraction of each element in the progenitor
            if blank, then assume Solar abundance ratios and use progenitor_metallicity to normalize
            For FIRE-2, use to compute higher-order corrections to surface abundances.
            For FIRE-3, use to compute stellar winds yields.
        age : float
            stellar age [Myr]
        model : str
            stellar evolution model for yields: 'fire2', 'fire2.1', 'fire2.2', 'fire3'
        return_mass : bool
            whether to return total mass of each element [Msun],
            instead of mass fraction wrt total ejecta/wind mass (default)

        Returns
        -------
        element_yield : ordered dict
            stellar nucleosynthetic yield for each element,
            in mass [M_sun] or mass fraction (wrt total ejecta mass)
        '''
        element_yield = collections.OrderedDict()
        for element_name in self.sun_massfraction:
            element_yield[element_name] = 0.0

        event_kind = event_kind.lower()
        assert event_kind in ['wind', 'supernova.cc', 'supernova.ii', 'supernova.ia']

        self._parse_model(model)

        # determine progenitor abundance[s]
        if isinstance(progenitor_massfraction_dict, dict) and len(progenitor_massfraction_dict) > 0:
            # input mass fraction for each element
            for element_name in element_yield:
                assert element_name in progenitor_massfraction_dict
        else:
            assert progenitor_metallicity >= 0
            # assume Solar abundance ratios and use progenitor_metallicity to normalize
            progenitor_massfraction_dict = {}
            for element_name in self.sun_massfraction:
                progenitor_massfraction_dict[element_name] = (
                    progenitor_metallicity * self.sun_massfraction[element_name]
                )

        if event_kind == 'wind':
            ejecta_mass = 1  # stellar wind yields are intrinsically mass fractions

            if 'fire2' in self.model:
                # FIRE-2: stellar_evolution.c line 583
                # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
                # assume the total wind abundances for He, C, N, and O as below
                # for all other elements, simply return progenitor surface abundance
                # below are mass fractions
                element_yield['helium'] = 0.36
                element_yield['carbon'] = 0.016
                element_yield['nitrogen'] = 0.0041
                element_yield['oxygen'] = 0.0118

                if self.model == 'fire2':
                    # oxygen yield increases linearly with progenitor metallicity at Z/Z_sun < 1.65
                    if progenitor_massfraction_dict['metals'] < 0.033:
                        element_yield['oxygen'] *= (
                            progenitor_massfraction_dict['metals'] / self.sun_massfraction['metals']
                        )
                    else:
                        element_yield['oxygen'] *= 1.65
                elif self.model in ['fire2.1', 'fire2.2']:
                    pass  # no dependence on progenitor metallicity

            elif self.model == 'fire3':
                # FIRE-3: stellar_evolution.c line 563
                # use surface abundance for all elements except He, C, N, O, S-process
                # C, N, O conserved to high accuracy in sum for secondary production

                # define initial fractions of H, He, C, N, O
                f_H_0 = (
                    1
                    - progenitor_massfraction_dict['metals']
                    - progenitor_massfraction_dict['helium']
                )
                f_He_0 = progenitor_massfraction_dict['helium']
                f_C_0 = progenitor_massfraction_dict['carbon']
                f_N_0 = progenitor_massfraction_dict['nitrogen']
                f_O_0 = progenitor_massfraction_dict['oxygen']
                f_CNO_0 = f_C_0 + f_N_0 + f_O_0 + 1e-10
                # CNO abundance scaled to Solar
                Z_CNO_0 = f_CNO_0 / (
                    self.sun_massfraction['carbon']
                    + self.sun_massfraction['nitrogen']
                    + self.sun_massfraction['oxygen']
                )
                # He production scales off of the fraction of H in IC
                # y represents the yield of He produced by burning H, scales off availability
                t1 = 2.8  # [Myr]
                t2 = 10
                t3 = 2300
                t4 = 3000
                y1 = 0.4 * min((Z_CNO_0 + 1e-3) ** 0.6, 2)
                y2 = 0.08
                y3 = 0.07
                y4 = 0.042
                if age < t1:
                    y = y1 * (age / t1) ** 3
                elif age < t2:
                    y = y1 * (age / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif age < t3:
                    y = y2 * (age / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
                elif age < t4:
                    y = y3 * (age / t3) ** (np.log(y4 / y3) / np.log(t4 / t3))
                else:
                    y = y4

                element_yield['helium'] = f_He_0 + y * f_H_0

                # secondary N production in CNO cycle: scales off of initial fraction of CNO:
                # y here represents fraction of CO mass converted to additional N
                t1 = 1
                t2 = 2.8
                t3 = 50
                t4 = 1900
                t5 = 14000
                y1 = 0.2 * max(1e-4, min(Z_CNO_0 ** 2, 0.9))
                y2 = 0.68 * min((Z_CNO_0 + 1e-3) ** 0.1, 0.9)
                y3 = 0.4
                y4 = 0.23
                y5 = 0.065
                if age < t1:
                    y = y1 * (age / t1) ** 3.5
                elif age < t2:
                    y = y1 * (age / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif age < t3:
                    y = y2 * (age / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
                elif age < t4:
                    y = y3 * (age / t3) ** (np.log(y4 / y3) / np.log(t4 / t3))
                elif age < t5:
                    y = y4 * (age / t4) ** (np.log(y5 / y4) / np.log(t5 / t4))
                else:
                    y = y5
                y = max(0, min(1, y))
                frac_loss_from_C = 0.5
                f_loss_CO = y * (f_C_0 + f_O_0)
                f_loss_C = min(frac_loss_from_C * f_loss_CO, 0.99 * f_C_0)
                f_loss_O = f_loss_CO - f_loss_C
                # convert mass from CO to N, conserving total CNO mass
                element_yield['nitrogen'] = f_N_0 + f_loss_CO
                element_yield['carbon'] = f_C_0 - f_loss_C
                element_yield['oxygen'] = f_O_0 - f_loss_O

                # primary C production: scales off initial H+He, generally small compared to loss
                # fraction above in SB99, large in some other models, small for early OB winds
                t1 = 5  # [Myr]
                t2 = 40
                t3 = 10000
                y1 = 1.0e-6
                y2 = 0.001
                y3 = 0.005
                if age < t1:
                    y = y1 * (age / t1) ** 3
                elif age < t2:
                    y = y1 * (age / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif age < t3:
                    y = y2 * (age / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
                else:
                    y = y3
                # simply multiple initial He by this factor to get final production
                y_H_to_C = (
                    1 - progenitor_massfraction_dict['metals'] - element_yield['helium']
                ) * y
                y_He_to_C = f_He_0 * y
                element_yield['helium'] -= y_He_to_C
                # transfer this mass fraction from H+He to C
                # gives stable results if 0 < f_He_0_to_C < 1
                element_yield['carbon'] += y_H_to_C + y_He_to_C

            # sum total metal mass (not including H or He)
            for k in element_yield:
                if k != 'helium':
                    element_yield['metals'] += element_yield[k]

        elif event_kind in ['supernova.cc' or 'supernova.ii']:
            if 'fire2' in self.model:
                # FIRE-2: stellar_evolution.c line 501 (or so)
                # yields from Nomoto et al 2006, IMF averaged
                # y = [He: 3.69e-1, C: 1.27e-2, N: 4.56e-3, O: 1.11e-1, Ne: 3.81e-2, Mg: 9.40e-3,
                # Si: 8.89e-3, S: 3.78e-3, Ca: 4.36e-4, Fe: 7.06e-3]
                ejecta_mass = 10.5  # [M_sun]
                # below are mass fractions
                element_yield['metals'] = 0.19
                element_yield['helium'] = 0.369
                element_yield['carbon'] = 0.0127
                element_yield['nitrogen'] = 0.00456
                element_yield['oxygen'] = 0.111
                # element_yield['neon'] = 0.0286  # original FIRE-2
                element_yield['neon'] = 0.0381  # later FIRE-2
                element_yield['magnesium'] = 0.00940
                element_yield['silicon'] = 0.00889
                element_yield['sulfur'] = 0.00378
                element_yield['calcium'] = 0.000436  # Nomoto et al 2013 suggest 0.05 - 0.1 M_sun
                element_yield['iron'] = 0.00706

                if model == 'fire2':
                    yield_nitrogen_orig = np.float(element_yield['nitrogen'])

                    # nitrogen yield increases linearly with progenitor metallicity @ Z/Z_sun < 1.65
                    if progenitor_massfraction_dict['metals'] < 0.033:
                        element_yield['nitrogen'] *= (
                            progenitor_massfraction_dict['metals'] / self.sun_massfraction['metals']
                        )
                    else:
                        element_yield['nitrogen'] *= 1.65
                    # correct total metal mass for nitrogen
                    element_yield['metals'] += element_yield['nitrogen'] - yield_nitrogen_orig
                elif model in ['fire2.1', 'fire2.2']:
                    pass  # no dependence on progenitor metallicity

            elif self.model == 'fire3':
                # FIRE-3: stellar_evolution.c line 471 (or so)
                # ejecta_mass = 8.72  # IMF-averaged value [M_sun], but FIRE-3 does not use it

                # numbers for interpolation of ejecta masses
                # [must be careful here that this integrates to the correct -total- ejecta mass]
                # these break times: tmin = 3.7 Myr corresponds to the first explosions
                # (Eddington-limited lifetime of the most massive stars), tbrk = 6.5 Myr to the end
                # of this early phase, stars with ZAMS mass ~30+ Msun here. curve flattens both from
                # IMF but also b/c mass-loss less efficient. tmax = 44 Myr to the last explosion
                # determined by lifetime of stars at 8 Msun
                cc_age_min = 3.7
                cc_age_brk = 6.5
                cc_age_max = 44
                cc_mass_max = 35
                cc_mass_brk = 10
                cc_mass_min = 6
                # power-law interpolation of ejecta mass
                if age <= cc_age_brk:
                    ejecta_mass = cc_mass_max * (age / cc_age_min) ** (
                        np.log(cc_mass_brk / cc_mass_max) / np.log(cc_age_brk / cc_age_min)
                    )
                else:
                    ejecta_mass = cc_mass_brk * (age / cc_age_brk) ** (
                        np.log(cc_mass_min / cc_mass_brk) / np.log(cc_age_max / cc_age_brk)
                    )
                cc_ages = np.array([3.7, 8, 18, 30, 44])  # [Myr]
                cc_yields_v_age = {
                    # He [IMF-mean y = 3.67e-1]
                    # have to remove normal solar correction and take care with winds
                    'helium': [4.61e-01, 3.30e-01, 3.58e-01, 3.65e-01, 3.59e-01],
                    # C [IMF-mean y = 3.08e-2]
                    # care needed in fitting out winds: wind = 6.5e-3, ejecta_only = 1.0e-3
                    'carbon': [2.37e-01, 8.57e-03, 1.69e-02, 9.33e-03, 4.47e-03],
                    # N [IMF-mean y = 4.47e-3] - care needed with winds, but not as essential
                    'nitrogen': [1.07e-02, 3.48e-03, 3.44e-03, 3.72e-03, 3.50e-03],
                    # O [IMF-mean y = 7.26e-2] - reasonable, generally IMF-integrated
                    # alpha-element total mass-yields lower than FIRE-2 by ~0.65
                    'oxygen': [9.53e-02, 1.02e-01, 9.85e-02, 1.73e-02, 8.20e-03],
                    # Ne [IMF-mean y = 1.58e-2] - roughly a hybrid of fit direct to ejecta and
                    # fit to all mass as above, truncating at highest masses
                    'neon': [2.60e-02, 2.20e-02, 1.93e-02, 2.70e-03, 2.75e-03],
                    # Mg [IMF-mean y = 9.48e-3]
                    # fit directly on ejecta and ignore mass-fraction rescaling because that is not
                    # reliable at early times: this gives a reasonable vnumber.
                    # important to note that early supernovae strongly dominate Mg
                    'magnesium': [2.89e-02, 1.25e-02, 5.77e-03, 1.03e-03, 1.03e-03],
                    # Si [IMF-mean y = 4.53e-3]
                    # lots comes from 1a's, so low here is not an issue
                    'silicon': [4.12e-04, 7.69e-03, 8.73e-03, 2.23e-03, 1.18e-03],
                    # S [IMF-mean y=3.01e-3] - more from Ia's
                    'sulfur': [3.63e-04, 5.61e-03, 5.49e-03, 1.26e-03, 5.75e-04],
                    # Ca [IMF-mean y = 2.77e-4] - Ia
                    'calcium': [4.28e-05, 3.21e-04, 6.00e-04, 1.84e-04, 9.64e-05],
                    # Fe [IMF-mean y = 4.11e-3] - Ia
                    'iron': [5.46e-04, 2.18e-03, 1.08e-02, 4.57e-03, 1.83e-03],
                }

                # use the fit parameters above for the piecewise power-law components to define the
                # yields at each time
                # int i_t=-1
                # for(k=0;k<i_tvec;k++)
                #     if(t_myr>tvec[k]) {i_t=k;}
                # for(k=0;k<10;k++) {
                #     int i_y = k + 1;
                #     if(i_t<0) {yields[i_y]=fvec[k][0];}
                #     else if(i_t>=i_tvec-1) {yields[i_y]=fvec[k][i_tvec-1];}
                #     else {yields[i_y] = fvec[k][i_t] * pow(t_myr/tvec[i_t] ,
                #         log(fvec[k][i_t+1]/fvec[k][i_t]) / log(tvec[i_t+1]/tvec[i_t]));}}

                ti = np.digitize(age, cc_ages, right=True) - 1

                for element_name in cc_yields_v_age:
                    sncc_yield_v_age = cc_yields_v_age[element_name]
                    if ti < 0:
                        element_yield[element_name] = sncc_yield_v_age[0]
                    elif ti >= cc_ages.size - 1:
                        element_yield[element_name] = sncc_yield_v_age[-1]
                    else:
                        element_yield[element_name] = sncc_yield_v_age[ti] * (
                            age / cc_ages[ti]
                        ) ** (
                            np.log(sncc_yield_v_age[ti + 1] / sncc_yield_v_age[ti])
                            / np.log(cc_ages[ti + 1] / cc_ages[ti])
                        )

                # sum heavy element yields to get the total metal yield, multiplying by a small
                # correction term to account for trace species not explicitly followed above
                # [mean for CC]
                element_yield['metals'] = 0
                for element_name in element_yield:
                    if element_name not in ['metals', 'helium']:
                        # assume some trace species proportional to each species,
                        # not correct in detail, but a tiny correction, so negligible
                        element_yield['metals'] += 1.0144 * element_yield[element_name]

        elif event_kind == 'supernova.ia':
            ejecta_mass = 1.4  # [M_sun]

            if 'fire2' in self.model:
                # FIRE-2: stellar_evolution.c line 498 (or so)
                # yields from Iwamoto et al 1999, W7 model, IMF averaged
                # below are mass fractions
                element_yield['metals'] = 1
                element_yield['helium'] = 0
                element_yield['carbon'] = 0.035
                element_yield['nitrogen'] = 8.57e-7
                element_yield['oxygen'] = 0.102
                element_yield['neon'] = 0.00321
                element_yield['magnesium'] = 0.00614
                element_yield['silicon'] = 0.111
                element_yield['sulfur'] = 0.0621
                element_yield['calcium'] = 0.00857
                element_yield['iron'] = 0.531

            elif self.model == 'fire3':
                # FIRE-3: stellar_evolution.c line 460 (or so)
                # total metal mass (species below, + residuals primarily in Ar, Cr, Mn, Ni)
                element_yield['metals'] = 1
                # adopted yield: mean of W7 and WDD2 in Mori et al 2018
                # other models included below for reference in comments
                # arguably better obs calibration versus LN/NL papers
                element_yield['helium'] = 0
                element_yield['carbon'] = 1.76e-2
                element_yield['nitrogen'] = 2.10e-06
                element_yield['oxygen'] = 7.36e-2
                element_yield['neon'] = 2.02e-3
                element_yield['magnesium'] = 6.21e-3
                element_yield['silicon'] = 1.46e-1
                element_yield['sulfur'] = 7.62e-2
                element_yield['calcium'] = 1.29e-2
                element_yield['iron'] = 5.58e-1
                # updated W7 in Nomoto + Leung 18 review - not significantly different from updated
                # W7 below, bit more of an outlier and review tables seem a bit unreliable (typos)
                # yields[2]=3.71e-2; yields[3]=7.79e-10; yields[4]=1.32e-1; yields[5]=3.11e-3;
                # yields[6]=3.07e-3; yields[7]=1.19e-1; yields[8]=5.76e-2; yields[9]=8.21e-3;
                # yields[10]=5.73e-1
                # mean of new yields for W7 + WDD2 in Leung + Nomoto et al 2018
                # yields[2]=1.54e-2; yields[3]=1.24e-08; yields[4]=8.93e-2; yields[5]=2.41e-3;
                # yields[6]=3.86e-3; yields[7]=1.34e-1; yields[8]=7.39e-2; yields[9]=1.19e-2;
                # yields[10]=5.54e-1
                # W7 [Mori+18] [3.42428571e-02, 4.16428571e-06, 9.68571429e-02, 2.67928571e-03,
                # 7.32857143e-03, 1.25296429e-01, 5.65937143e-02, 8.09285714e-03, 5.68700000e-01]
                # -- absolute yield in solar // WDD2 [Mori+18] [9.70714286e-04, 2.36285714e-08,
                # 5.04357143e-02, 1.35621429e-03, 5.10112857e-03, 1.65785714e-01, 9.57078571e-02,
                # 1.76928571e-02, 5.47890000e-01] -- absolute yield in solar
                # updated W7 in Leung + Nomoto et al 2018 - seems bit low in Ca/Fe,
                # less plausible if those dominated by Ia's
                # yields[2]=1.31e-2; yields[3]=7.59e-10; yields[4]=9.29e-2; yields[5]=1.79e-3;
                # yields[6]=2.82e-3; yields[7]=1.06e-1; yields[8]=5.30e-2; yields[9]=6.27e-3;
                # yields[10]=5.77e-1
                # Seitenzahl et al 2013, model N100 [favored]
                # high Si, seems bit less plausible vs other models here
                # yields[2]=2.17e-3; yields[3]=2.29e-06; yields[4]=7.21e-2; yields[5]=2.55e-3;
                # yields[6]=1.10e-2; yields[7]=2.05e-1; yields[8]=8.22e-2; yields[9]=1.05e-2;
                # yields[10]=5.29e-1
                # new benchmark model in Leung + Nomoto et al 2018 [closer to WDD2 in lighter
                # elements, to W7 in heavier elements] - arguably better theory motivation versus
                # Mori et al combination
                # yields[2]=1.21e-3; yields[3]=1.40e-10; yields[4]=4.06e-2; yields[5]=1.29e-4;
                # yields[6]=7.86e-4; yields[7]=1.68e-1; yields[8]=8.79e-2; yields[9]=1.28e-2;
                # yields[10]=6.14e-1

        if (
            (self.model == 'fire2' and 'supernova' in event_kind)
            and isinstance(progenitor_massfraction_dict, dict)
            and len(progenitor_massfraction_dict) > 0
        ):
            # FIRE-2: stellar_evolution.c line 509
            # enforce that yields obey pre-existing surface abundances
            # allow for larger abundances in the progenitor star - usually irrelevant
            # original FIRE-2 version applied this to all mass-loss channels (including winds)
            # later FIRE-2 version applies this only to supernovae

            # get pure (non-metal) mass fraction of star
            pure_mass_fraction = 1 - progenitor_massfraction_dict['metals']

            for element_name in element_yield:
                if element_yield[element_name] > 0:
                    # apply (new) yield only to pure (non-metal) mass of star
                    element_yield[element_name] *= pure_mass_fraction
                    # correction relative to solar abundance
                    element_yield[element_name] += (
                        progenitor_massfraction_dict[element_name]
                        - self.sun_massfraction[element_name]  # I do not understand this term
                    )
                    element_yield[element_name] = np.clip(element_yield[element_name], 0, 1)

        if return_mass:
            # convert to yield masses [M_sun]
            for element_name in element_yield:
                element_yield[element_name] *= ejecta_mass

        return element_yield

    def assign_element_yields(
        self, progenitor_metallicity=None, progenitor_massfraction_dict=None, age=None
    ):
        '''
        Store nucleosynthetic yields from all stellar channels, for a fixed progenitor metallicity,
        as dictionaries with element name as kwargs and yield [mass fraction wrt total
        ejecta/wind mass] as values.
        Useful to avoid having to re-call get_element_yields() many times.

        Parameters
        -----------
        progenitor_metallicity : float
            total metallicity of progenitor [linear mass fraction wrt sun_mass_fraction['metals']]
        progenitor_massfraction_dict : dict or bool [optional]
            optional: dictionary that contains the mass fraction of each element in the progenitor
            if blank, then assume Solar abundance ratios and use progenitor_metallicity to normalize
            For FIRE-2, use to compute higher-order corrections to surface abundances.
            For FIRE-3, use to compute stellar winds yields.
        age : float
            stellar age [Myr]
        '''
        # store yields as mass fraction wrt total ejecta/wind mass
        for event_kind in self._event_kinds:
            self[event_kind] = self.get_element_yields(
                event_kind, progenitor_metallicity, progenitor_massfraction_dict, age=age,
            )


NucleosyntheticYield = NucleosyntheticYieldClass()


# --------------------------------------------------------------------------------------------------
# stellar mass loss
# --------------------------------------------------------------------------------------------------
class StellarWindClass:
    '''
    Compute mass-loss rates and cumulative mass-loss fractions (with respect to mass at formation)
    for stellar winds in the FIRE-2 or FIRE-3 model.

    Model variants for FIRE-2
         'fire2.1' = remove dependence on progenitor metallicity for all yields
            (via setting all progenitors to Solar)
         'fire2.2' = above + turn off dependence on progenitor metallicity for stellar wind rate
            (via setting all progenitors to Solar)
    '''

    def __init__(self, model=FIRE_MODEL_DEFAULT):
        '''
        Parameters
        ----------
        model : str
             model for wind rate: 'fire2', 'fire2.1', 'fire2.2', 'fire3'
        '''
        # stellar wind mass loss is intrinsically mass fraction wrt formation mass
        self.ejecta_mass = 1.0

        self._models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire3']
        self.sun_massfraction = None
        self.ages_transition = None

        self._parse_model(model)

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for stellar wind rate: 'fire2', 'fire3'
        '''
        reset_parameters = False

        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                reset_parameters = True
            self.model = model

        assert self.model in self._models_available

        if reset_parameters:
            # reset solar abundances
            self.sun_massfraction = get_sun_massfraction(self.model)

            # set transition ages [Myr]
            if 'fire2' in self.model:
                self.ages_transition = np.array([1.0, 3.5, 100])  # [Myr]
            elif 'fire3' in self.model:
                self.ages_transition = np.array([1.7, 4, 20, 1000])  # [Myr]

    def get_mass_loss_rate(
        self, ages, metallicity=1, metal_mass_fraction=None, model=None, element_name=None
    ):
        '''
        Get rate of fractional mass loss wrt mass at formation [Myr ^ -1] from stellar
        winds in FIRE-2 or FIRE-3.
        Input either metallicity (linear, wrt Solar) or (raw) metal_mass_fraction.

        Includes all non-supernova mass-loss channels, dominated by O, B, and AGB stars.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            metallicity [(linear) wrt Sun] of progenitor, for scaling the wind rates
            input either this or metal_mass_fraction (below)
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02]
            For FIRE-3, this should be Iron abundance [scaled to Solar := 1.38e-3]
        metal_mass_fraction : float
            optional: mass fraction of given metal in progenitor stars
            input either this or metallicity (above)
            For FIRE-2, this should be *total* metals (everything not H, He)
            For FIRE-3, this should be iron
        model : str
            model for wind rate: 'fire2', 'fire2.1', 'fire2.2', 'fire3'
        element_name : str
            name of element to get fractional mass loss of
            if None or '', get total fractional mass loss wrt mass at formation

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss wrt mass at formation [Myr ^ -1]
        '''
        # min and max imposed in FIRE-2 and FIRE-3 for stellar wind rates for stability
        metallicity_min = 0.01
        metallicity_max = 3
        age_min = 0  # [Myr]
        age_max = 14001

        self._parse_model(model)

        if metal_mass_fraction is not None:
            if 'fire2' in self.model:
                metallicity = metal_mass_fraction / self.sun_massfraction['metals']
            elif 'fire3' in self.model:
                metallicity = metal_mass_fraction / self.sun_massfraction['iron']

        metallicity = np.clip(metallicity, metallicity_min, metallicity_max)

        if self.model == 'fire2.2':
            # force wind rates to be independent of progenitor metallicity,
            # by setting all progenitors to Solar abundance
            metallicity = 1

        if 'fire2' in self.model:
            # FIRE-2: stellar_evolution.c line 350
            if np.isscalar(ages):
                assert ages >= age_min and ages < age_max
                # FIRE-2: stellar_evolution.c line 352
                if ages <= 1:
                    # rates = 4.76317  # rate [Gyr^-1], used (accidentally?) in original FIRE-2
                    rates = 4.76317 * metallicity  # # rate [Gyr^-1]
                elif ages <= 3.5:
                    rates = 4.76317 * metallicity * ages ** (1.838 * (0.79 + np.log10(metallicity)))
                elif ages <= 100:
                    rates = 29.4 * (ages / 3.5) ** -3.25 + 0.0041987
                else:
                    rates = 0.41987 * (ages / 1e3) ** -1.1 / (12.9 - np.log(ages / 1e3))
            else:
                assert np.min(ages) >= age_min and np.max(ages) < age_max
                ages = np.asarray(ages)
                rates = np.zeros(ages.size)

                masks = np.where(ages <= 1)[0]
                # rates[masks] = 4.76317  # rate [Gyr^-1], used (accidentally?) in original FIRE-2
                rates[masks] = 4.76317 * metallicity  # rate [Gyr^-1]

                masks = np.where((ages > 1) * (ages <= 3.5))[0]
                rates[masks] = (
                    4.76317 * metallicity * ages[masks] ** (1.838 * (0.79 + np.log10(metallicity)))
                )

                masks = np.where((ages > 3.5) * (ages <= 100))[0]
                rates[masks] = 29.4 * (ages[masks] / 3.5) ** -3.25 + 0.0041987

                masks = np.where(ages > 100)[0]
                rates[masks] = (
                    0.41987 * (ages[masks] / 1e3) ** -1.1 / (12.9 - np.log(ages[masks] / 1e3))
                )

        elif 'fire3' in self.model:
            # FIRE-3: stellar_evolution.c line 355
            # separates the more robust line-driven winds [massive-star-dominated] component,
            # and -very- uncertain AGB. extremely good fits to updated STARBURST99 result for a
            # 3-part Kroupa IMF (0.3,1.3,2.3 slope, 0.01-0.08-0.5-100 Msun, 8-120 SNe/BH cutoff,
            # wind model evolution, Geneva v40 [rotating, Geneva 2013 updated tracks, at all
            # metallicities available, ~0.1-1 solar], sampling times 1e4-2e10 yr at high res
            # massive stars: piecewise continuous, linking constant early and rapid late decay
            f1 = 3 * metallicity ** 0.87  # rates [Gyr^-1]
            f2 = 20 * metallicity ** 0.45
            f3 = 0.6 * metallicity
            t1 = 1.7  # transition ages [Myr]
            t2 = 4
            t3 = 20
            # AGB: note that essentially no models [any of the SB99 geneva or padova tracks,
            # or NuGrid, or recent other MESA models] predict a significant dependence on
            # metallicity (that shifts slightly when the 'bump' occurs, but not the overall loss
            # rate), so this term is effectively metallicity-independent
            f_agb = 0.01
            f_agb2 = 0.01
            t_agb = 1000  # [Myr]

            if np.isscalar(ages):
                assert ages >= age_min and ages < age_max

                if ages <= t1:
                    rates = f1
                elif ages <= t2:
                    rates = f1 * (ages / t1) ** (np.log(f2 / f1) / np.log(t2 / t1))
                elif ages <= t3:
                    rates = f2 * (ages / t2) ** (np.log(f3 / f2) / np.log(t3 / t2))
                else:
                    rates = f3 * (ages / t3) ** -3.1

                # add AGB
                rates += f_agb / ((1 + (ages / t_agb) ** 1.1) * (1 + f_agb2 / (ages / t_agb)))

            else:
                assert np.min(ages) >= age_min and np.max(ages) < age_max
                ages = np.asarray(ages)
                rates = np.zeros(ages.size)

                masks = np.where(ages <= t1)[0]
                rates[masks] = f1

                masks = np.where((ages > t1) * (ages <= t2))[0]
                rates[masks] = f1 * (ages[masks] / t1) ** (np.log(f2 / f1) / np.log(t2 / t1))

                masks = np.where((ages > t2) * (ages <= t3))[0]
                rates[masks] = f2 * (ages[masks] / t2) ** (np.log(f3 / f2) / np.log(t3 / t2))

                masks = np.where(ages > t3)[0]
                rates[masks] = f3 * (ages[masks] / t3) ** -3.1

                # add AGB
                rates += f_agb / ((1 + (ages / t_agb) ** 1.1) * (1 + f_agb2 / (ages / t_agb)))

        rates *= 1e-3  # convert fractional mass loss rate to [Myr ^ -1]
        # rates *= 1.4 * 0.291175  # old: expected return fraction from stellar winds alone (~17%)

        if element_name:
            NucleosyntheticYield = NucleosyntheticYieldClass(self.model)

            if ages is not None and not np.isscalar(ages) and 'fire3' in self.model:
                for ai, age in enumerate(ages):
                    element_yield = NucleosyntheticYield.get_element_yields(
                        'wind', metallicity, age=age
                    )
                    rates[ai] *= element_yield[element_name]
            else:
                element_yield = NucleosyntheticYield.get_element_yields(
                    'wind', metallicity, age=ages,
                )
                rates *= element_yield[element_name]

        return rates

    def get_mass_loss(
        self,
        age_min=0,
        age_maxs=99,
        metallicity=1,
        metal_mass_fraction=None,
        model=None,
        element_name='',
    ):
        '''
        Get cumulative fractional mass loss wrt mass at formation via stellar winds
        within input age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        metallicity : float
            metallicity [(linear) wrt Sun] of progenitor, for scaling the wind rates
            input either this or metal_mass_fraction (below)
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02]
            For FIRE-3, this should be Iron abundance [scaled to Solar := 1.38e-3]
        metal_mass_fraction : float
            optional: mass fraction of given metal in progenitor stars
            input either this or metallicity (above)
            For FIRE-2, this should be *total* metals (everything not H, He)
            For FIRE-3, this should be iron
        model : str
            model for wind rate: 'fire2', 'fire2.1', 'fire2.2', 'fire3'
        element_name : str
            name of element to get fractional mass loss of
            if None or '', get total fractional mass loss wrt mass at formation

        Returns
        -------
        mass_loss_fractions : float or array
            fractional mass loss[es] wrt formation mass
        '''
        self._parse_model(model)

        if np.isscalar(age_maxs):
            age_maxs = [age_maxs]

        mass_loss_fractions = np.zeros(len(age_maxs))
        for age_i, age in enumerate(age_maxs):
            mass_loss_fractions[age_i] = integrate.quad(
                self.get_mass_loss_rate,
                age_min,
                age,
                (metallicity, metal_mass_fraction, None, element_name),
                points=self.ages_transition,
            )[0]

            # this method may be more stable for piece-wise (discontinuous) function
            # age_bin_width = 0.001  # [Myr]
            # ages = np.arange(age_min, age + age_bin_width, age_bin_width)
            # mass_loss_fractions[age_i] = self.get_rate(
            #    ages, metallicity, metal_mass_fraction).sum() * age_bin_width

        if len(mass_loss_fractions) == 1:
            mass_loss_fractions = [mass_loss_fractions]

        return mass_loss_fractions


StellarWind = StellarWindClass()


class SupernovaCCClass:
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for core-collapse supernovae
    in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(
        self, model=FIRE_MODEL_DEFAULT, cc_age_min=None, cc_age_break=None, cc_age_max=None
    ):
        '''
        Parameters
        ----------
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]
        '''
        self._models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire3']
        self.model = None
        self.cc_age_min = None
        self.cc_age_break = None
        self.cc_age_max = None
        self.sun_massfraction = None
        self.ages_transition = None

        self._parse_model(model, cc_age_min, cc_age_break, cc_age_max)

    def _parse_model(self, model, cc_age_min=None, cc_age_break=None, cc_age_max=None):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]
        '''
        reset_parameters = False

        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                reset_parameters = True
            self.model = model

        assert self.model in self._models_available

        if reset_parameters:
            # reset solar abundances
            self.sun_massfraction = get_sun_massfraction(self.model)

            if 'fire2' in self.model:
                self.ejecta_mass = 10.5  # ejecta mass per event, IMF-averaged [M_sun]
            elif 'fire3' in self.model:
                # IMF-averaged mass per event [M_sun], but FIRE-3 does not use this directly,
                # because it samples different ejecta masses for different mass supernovae
                self.ejecta_mass = 8.72
                self.cc_mass_max = 35
                self.cc_mass_brk = 10
                self.cc_mass_min = 6

            # reset transition ages
            if cc_age_min is None:
                if 'fire2' in self.model:
                    cc_age_min = 3.4  # [Myr]
                elif 'fire3' in self.model:
                    cc_age_min = 3.7  # [Myr]
            assert cc_age_min >= 0
            self.cc_age_min = cc_age_min

            if cc_age_break is None:
                if 'fire2' in self.model:
                    cc_age_break = 10.37  # [Myr]
                elif 'fire3' in self.model:
                    cc_age_break = 6.5  # [Myr]
            assert cc_age_break >= 0
            self.cc_age_break = cc_age_break

            if cc_age_max is None:
                if 'fire2' in self.model:
                    cc_age_max = 37.53  # [Myr]
                elif 'fire3' in self.model:
                    cc_age_max = 44  # [Myr]
            assert cc_age_max >= 0
            self.cc_age_max = cc_age_max

            self.ages_transition = np.sort([self.cc_age_min, self.cc_age_break, self.cc_age_max])

    def get_rate(self, ages, model=None, cc_age_min=None, cc_age_break=None, cc_age_max=None):
        '''
        Get specific rate [Myr ^ -1 per M_sun of stars at formation] of core-collapse supernovae.

        FIRE-2
            Rates are from Starburst99 energetics: get rate from overall energetics assuming each
            core-collapse supernova is 10^51 erg.
            Core-collapse supernovae occur from 3.4 to 37.53 Myr after formation:
                3.4 to 10.37 Myr:    rate / M_sun = 5.408e-10 yr ^ -1
                10.37 to 37.53 Myr:  rate / M_sun = 2.516e-10 yr ^ -1

        Parameters
        ----------
        ages : float or array
            age[s] of stellar population [Myr]
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]

        Returns
        -------
        rates : float or array
            specific rate[s] [Myr ^ -1 per M_sun of stars at formation]
        '''

        def _get_rate_fire3(age, kind):
            rate1 = 3.9e-4  # [Myr ^ -1]
            rate2 = 5.1e-4  # [Myr ^ -1]
            rate3 = 1.8e-4  # [Myr ^ -1]
            if kind == 'early':
                return rate1 * (age / self.cc_age_min) ** (
                    np.log(rate2 / rate1) / np.log(self.cc_age_break / self.cc_age_min)
                )
            elif kind == 'late':
                return rate2 * (age / self.cc_age_break) ** (
                    np.log(rate3 / rate2) / np.log(self.cc_age_max / self.cc_age_break)
                )

        fire2_rate_early = 5.408e-4  # [Myr ^ -1]
        fire2_rate_late = 2.516e-4  # [Myr ^ -1]

        age_min = 0
        age_max = 14001

        self._parse_model(model, cc_age_min, cc_age_break, cc_age_max)

        if np.isscalar(ages):
            assert ages >= age_min and ages < age_max
            if ages < self.cc_age_min or ages > self.cc_age_max:
                rates = 0
            elif ages <= self.cc_age_break:
                if 'fire2' in self.model:
                    rates = fire2_rate_early
                elif self.model == 'fire3':
                    rates = _get_rate_fire3(ages, 'early')
            elif ages > self.cc_age_break:
                if 'fire2' in self.model:
                    rates = fire2_rate_late
                elif self.model == 'fire3':
                    rates = _get_rate_fire3(ages, 'late')
        else:
            assert np.min(ages) >= age_min and np.max(ages) < age_max
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where((ages >= self.cc_age_min) * (ages <= self.cc_age_break))[0]
            if 'fire2' in self.model:
                rates[masks] = fire2_rate_early
            elif self.model == 'fire3':
                rates[masks] = _get_rate_fire3(ages[masks], 'early')

            masks = np.where((ages > self.cc_age_break) * (ages <= self.cc_age_max))[0]
            if 'fire2' in self.model:
                rates[masks] = fire2_rate_late
            elif self.model == 'fire3':
                rates[masks] = _get_rate_fire3(ages[masks], 'late')

        return rates

    def get_number(
        self,
        age_min=0,
        age_maxs=99,
        model=None,
        cc_age_min=None,
        cc_age_break=None,
        cc_age_max=None,
    ):
        '''
        Get specific number [per M_sun of stars at formation] of core-collapse supernova events in
        input age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]

        Returns
        -------
        numbers : float or array
            specific number[s] of supernova events [per M_sun of stars at formation]
        '''
        self._parse_model(model, cc_age_min, cc_age_break, cc_age_max)

        if np.isscalar(age_maxs):
            age_maxs = [age_maxs]

        numbers = np.zeros(len(age_maxs))
        for age_i, age in enumerate(age_maxs):
            numbers[age_i] = integrate.quad(
                self.get_rate,
                age_min,
                age,
                points=[self.cc_age_min, self.cc_age_break, self.cc_age_max],
            )[0]

            # alternate method
            # age_bin_width = 0.01
            # ages = np.arange(age_min, age + age_bin_width, age_bin_width)
            # numbers[age_i] = self.get_rate(ages).sum() * age_bin_width

        if len(numbers) == 1:
            numbers = numbers[0]  # return scalar if input single max age

        return numbers

    def get_mass_loss_rate(
        self,
        ages,
        model=None,
        cc_age_min=None,
        cc_age_break=None,
        cc_age_max=None,
        element_name=None,
        metallicity=1.0,
    ):
        '''
        Get fractional mass-loss rate (ejecta mass relative to mass at formation) [Myr ^ -1] from
        core-collapse supernova ejecta at input age[s].

        Parameters
        ----------
        ages : float or array
            stellar ages [Myr]
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]
        element_name : str [optional]
            name of element to get fraction mass loss rate of
            if None or '', get fractional mass loss rate of total ejecta
        metallicity : float
            metallicity (wrt Solar) of progenitor (for Nitrogen yield in FIRE-2)

        Returns
        -------
        cc_mass_loss_rates : float or array
            fractional mass loss rate[s] (ejecta mass relative to mass at formation) [Myr ^ -1]
        '''
        self._parse_model(model, cc_age_min, cc_age_break, cc_age_max)

        if 'fire2' in self.model:
            ejecta_masses = self.ejecta_mass
        elif 'fire3' in self.model:
            if np.isscalar(ages):
                # power-law interpolation of ejecta mass
                if ages < self.cc_age_min or ages > self.cc_age_max:
                    ejecta_masses = 0
                elif ages <= self.cc_age_break:
                    ejecta_masses = self.cc_mass_max * (ages / self.cc_age_min) ** (
                        np.log(self.cc_mass_brk / self.cc_mass_max)
                        / np.log(self.cc_age_break / self.cc_age_min)
                    )
                else:
                    ejecta_masses = self.cc_mass_brk * (ages / self.cc_age_break) ** (
                        np.log(self.cc_mass_min / self.cc_mass_brk)
                        / np.log(self.cc_age_max / self.cc_age_break)
                    )
            else:
                ages = np.asarray(ages)
                ejecta_masses = np.zeros(len(ages))

                # power-law interpolation of ejecta mass
                masks = ages < self.cc_age_min
                ejecta_masses[masks] = 0
                masks = ages > self.cc_age_max
                ejecta_masses[masks] = 0
                masks = np.where(ages <= self.cc_age_break)[0]
                ejecta_masses[masks] = self.cc_mass_max * (ages[masks] / self.cc_age_min) ** (
                    np.log(self.cc_mass_brk / self.cc_mass_max)
                    / np.log(self.cc_age_break / self.cc_age_min)
                )
                masks = np.where(ages > self.cc_age_break)[0]
                ejecta_masses[masks] = self.cc_mass_brk * (ages[masks] / self.cc_age_break) ** (
                    np.log(self.cc_mass_min / self.cc_mass_brk)
                    / np.log(self.cc_age_max / self.cc_age_break)
                )

        cc_mass_loss_rates = ejecta_masses * self.get_rate(ages)

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(self.model)
            if ages is not None and not np.isscalar(ages) and 'fire3' in self.model:
                for ai, age in enumerate(ages):
                    element_yield = NucelosyntheticYield.get_element_yields(
                        'supernova.cc', metallicity, age=age
                    )
                cc_mass_loss_rates[ai] *= element_yield[element_name]
            else:
                element_yield = NucelosyntheticYield.get_element_yields(
                    'supernova.cc', metallicity, age=ages
                )
                cc_mass_loss_rates *= element_yield[element_name]

        return cc_mass_loss_rates

    def get_mass_loss(
        self,
        age_min=0,
        age_maxs=99,
        model=None,
        cc_age_min=None,
        cc_age_break=None,
        cc_age_max=None,
        element_name=None,
        metallicity=1.0,
    ):
        '''
        Get fractional mass loss (ejecta mass relative to mass at formation] from core-collapse
        supernovae across input age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        model : str
            model for core-collapse supernova rates (delay time distribution): 'fire2', 'fire3'
        cc_age_min : float
            minimum age for core-collapse supernova to occur [Myr]
        cc_age_break : float
            age at which rate of core-collapse supernova changes/breaks [Myr]
        cc_age_min : float
            maximum age for core-collapse supernova to occur [Myr]
        element_name : str [optional]
            name of element to get fractional mass loss of
            if  None or '', get mass loss fraction from total ejecta
        metallicity : float
            metallicity (wrt Solar) of progenitor stars (for Nitrogen yield in FIRE-2)

        Returns
        -------
        mass_loss_fractions : float or array
            fractional mass loss[es] (ejecta mass relative to mass at formation)
        '''
        self._parse_model(model, cc_age_min, cc_age_break, cc_age_max)

        if np.isscalar(age_maxs):
            age_maxs = [age_maxs]

        mass_loss_fractions = np.zeros(len(age_maxs))
        for age_i, age in enumerate(age_maxs):
            mass_loss_fractions[age_i] = integrate.quad(
                self.get_mass_loss_rate,
                age_min,
                age,
                (None, None, None, None, element_name, metallicity),
                points=self.ages_transition,
            )[0]

        if len(mass_loss_fractions) == 1:
            mass_loss_fractions = [mass_loss_fractions]

        return mass_loss_fractions


SupernovaCC = SupernovaCCClass()


class SupernovaIaClass(ut.io.SayClass):
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for supernovae Ia
    in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=FIRE_MODEL_DEFAULT, ia_age_min=None):
        '''
        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        '''
        self.ejecta_mass = 1.4  # ejecta mass per event, IMF-averaged [M_sun]

        self._models_available = [
            'fire2',
            'fire2.1',
            'fire2.2',
            'fire3',
            'mannucci',
            'maoz',
        ]
        self.model = None
        self.ia_age_min = None
        self.sun_massfraction = None

        self._parse_model(model, ia_age_min)

    def _parse_model(self, model, ia_age_min):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        '''
        reset_parameters = False

        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                reset_parameters = True
            self.model = model

        assert self.model in self._models_available

        if reset_parameters:
            # reset solar abundances
            self.sun_massfraction = get_sun_massfraction(self.model)

            if ia_age_min is None:
                if 'fire2' in self.model:
                    ia_age_min = 37.53  # [Myr] ensure FIRE-2 default
                    # self.say(f'input Ia model = {model}, forcing Ia age min = {ia_age_min} Myr')
                elif self.model == 'fire3':
                    ia_age_min = 44  # [Myr] ensure FIRE-3 default
                    # self.say(f'input Ia model = {model}, forcing Ia age min = {ia_age_min} Myr')
            assert ia_age_min >= 0
            self.ia_age_min = ia_age_min

    def get_rate(self, ages, model=None, ia_age_min=None):
        '''
        Get specific rate [Myr ^ -1 per M_sun of stars at formation] of supernovae Ia events.

        FIRE-2
            rates are from Mannucci, Della Valle, & Panagia 2006, for a delayed population
            (constant rate) + prompt population (Gaussian), starting 37.53 Myr after formation:
            rate / M_sun = 5.3e-14 + 1.6e-11 * exp(-0.5 * ((star_age - 5e-5) / 1e-5) ** 2) yr ^ -1

        FIRE-3
            power-law model from Maoz & Graur 2017, starting 44 Myr after formation
            normalized to 1.6 events per 1000 Msun per Hubble time:
            rate / M_sun = 2.67e-13 * (star_age / 1e6) ** (-1.1) yr ^ -1

        Parameters
        ----------
        ages : float
            age of stellar population [Myr]
        model : str
            model for supernova Ia rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
            decreasing to 10 Myr increases total number by ~50%,
            increasing to 100 Myr decreases total number by ~50%

        Returns
        -------
        rates : float or array
            specific rate[s] of supernovae Ia events [Myr ^ -1 per M_sun of stars at formation]
        '''

        def _get_rate(ages):
            if 'fire2' in self.model or self.model == 'mannucci':
                # Mannucci, Della Valle, & Panagia 2006
                rate = 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((ages - 50) / 10) ** 2)  # [Myr ^ -1]
            elif 'fire3' in self.model:
                # this normalization is 2.67e-7 [Myr ^ -1]
                rate = (
                    1.6e-3
                    * 7.94e-5
                    / ((self.ia_age_min / 100) ** -0.1 - 0.61)
                    * (ages / 1e3) ** -1.1
                )
            elif self.model == 'maoz':
                # Maoz & Graur 2017
                rate = 2.6e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1] compromise fit
                # fit to volumetric, Hubble-time-integrated Ia N/M = 1.3 +/- 0.1 per 1000 Msun
                # rate = 2.1e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1]
                # fit to field galaxies, Hubble-time-integrated Ia N/M = 1.6 +/- 0.1 per 1000 Msun
                # rate = 2.6e-7 * (ages / 1e3) ** -1.13  # [Myr ^ -1]
                # fit to galaxy clusters, Hubble-time-integrated Ia N/M = 5.4 +/- 0.1 per 1000 Msun
                # rate = 6.7e-7 * (ages / 1e3) ** -1.39  # [Myr ^ -1]

            return rate

        self._parse_model(model, ia_age_min)

        if np.isscalar(ages):
            if ages < self.ia_age_min:
                rates = 0
            else:
                rates = _get_rate(ages)
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where(ages >= self.ia_age_min)[0]
            rates[masks] = _get_rate(ages[masks])

        return rates

    def get_number(self, age_min=0, age_maxs=99, model=None, ia_age_min=None):
        '''
        Get specific number [per M_sun of stars at formation] of supernovae Ia events in input age
        interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        numbers : float or array
            specific number[s] of supernovae Ia [per M_sun of stars at formation]
        '''
        self._parse_model(model, ia_age_min)

        if np.isscalar(age_maxs):
            age_maxs = [age_maxs]

        numbers = np.zeros(len(age_maxs))
        for age_i, age in enumerate(age_maxs):
            numbers[age_i] = integrate.quad(self.get_rate, age_min, age)[0]

        if len(numbers) == 1:
            numbers = numbers[0]  # return scalar if input single max age

        return numbers

    def get_mass_loss_rate(self, ages, model=None, ia_age_min=None, element_name=''):
        '''
        Get fractional mass loss rate (ejecta mass relative to mass of stars at formation)
        [Myr ^ -1] via supernova Ia ejecta in input age interval[s].

        Parameters
        ----------
        ages : float or array
            stellar age[s] [Myr]
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        element_name : str [optional]
            name of element to get fractional mass loss of
            if None or '', get mass loss of total ejecta

        Returns
        -------
        mass_loss_rates : float or array
             fractional mass loss rate[s] (ejecta mass relative to mass of stars at formation) from
             supernovae Ia events [Myr ^ -1]
        '''
        self._parse_model(model, ia_age_min)

        mass_loss_rates = self.ejecta_mass * self.get_rate(ages)

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(self.model)
            element_yield = NucelosyntheticYield.get_element_yields('supernova.ia')
            mass_loss_rates *= element_yield[element_name]

        return mass_loss_rates

    def get_mass_loss(self, age_min=0, age_maxs=99, model=None, ia_age_min=None, element_name=''):
        '''
        Get fractional mass loss (ejecta mass relative to mass of stars at formation) via
        supernova Ia events in age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        element_name : str [optional]
            name of element to get fractional mass loss of
            if None or '', get mass loss of total ejecta

        Returns
        -------
        mass_loss_fractions : float or array
            fractional mass loss[es] (ejecta mass relative to mass of stars at formation)
            from supernovae Ia events
        '''
        self._parse_model(model, ia_age_min)

        mass_loss_fractions = self.ejecta_mass * self.get_number(age_min, age_maxs)

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(self.model)
            element_yield = NucelosyntheticYield.get_element_yields('supernova.ia')
            mass_loss_fractions *= element_yield[element_name]

        return mass_loss_fractions


SupernovaIa = SupernovaIaClass()


class MassLossClass(ut.io.SayClass):
    '''
    Compute mass loss from all channels (stellar winds, core-collapse and Ia supernovae) as
    implemented in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=FIRE_MODEL_DEFAULT):
        '''
        Parameters
        ----------
        model : str
            stellar evolution model to use: 'fire2', 'fire2.1', 'fire2.2', 'fire3'
        '''
        self._models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire3']
        self.model = None
        self.sun_massfraction = None
        self._parse_model(model)

        self.SupernovaCC = SupernovaCCClass(self.model)
        self.SupernovaIa = SupernovaIaClass(self.model)
        self.StellarWind = StellarWindClass(self.model)
        self.Spline = None
        self.AgeBin = None
        self.MetalBin = None
        self.mass_loss_fractions = None

        self._file_name = os.environ['HOME'] + '/.gizmo_star_mass_loss_spline.pkl'

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            stellar evolution model to use: 'fire2', 'fire3'
        '''
        reset_parameters = False

        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                reset_parameters = True
            self.model = model

        assert self.model in self._models_available

        if reset_parameters:
            # reset solar abundances
            self.sun_massfraction = get_sun_massfraction(self.model)

    def get_mass_loss_rate(self, ages, metallicity=1, metal_mass_fraction=None, element_name=''):
        '''
        Get rate[s] of fractional mass loss relative to mass of stars at formation [Myr ^ -1]
        from all stellar evolution channels in FIRE-2 or FIRE-3.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            metallicity [(linear) wrt Sun] of progenitor, for scaling the wind rates
            input either this or metal_mass_fraction (below)
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02]
            For FIRE-3, this should be Iron abundance [scaled to Solar := 1.38e-3]
        metal_mass_fraction : float
            optional: mass fraction of given metal in progenitor stars
            input either this or metallicity (above)
            For FIRE-2, this should be *total* metals (everything not H, He)
            For FIRE-3, this should be iron
        element_name : str [optional]
            name of element to get fractional mass loss of
            if None or '', get mass loss from all elements

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss relative to mass of stars at formation [Myr ^ -1]
        '''
        return (
            self.StellarWind.get_mass_loss_rate(
                ages, metallicity, metal_mass_fraction, element_name=element_name
            )
            + self.SupernovaCC.get_mass_loss_rate(
                ages, element_name=element_name, metallicity=metallicity
            )
            + self.SupernovaIa.get_mass_loss_rate(ages, element_name=element_name)
        )

    def get_mass_loss(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None, element_name=''
    ):
        '''
        Get fractional mass loss relative to mass of stars at formation via all stellar evolution
        channels within age interval[s] in the FIRE-2 or FIRE-3 model.

        Parameters
        ----------
        age_min : float
            min (starting) age of stellar population [Myr]
        age_maxs : float or array
            max (ending) age[s] of stellar population [Myr]
        metallicity : float
            metallicity [(linear) wrt Sun] of progenitor, for scaling the wind rates
            input either this or metal_mass_fraction (below)
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02]
            For FIRE-3, this should be Iron abundance [scaled to Solar := 1.38e-3]
        metal_mass_fraction : float
            optional: mass fraction of given metal in progenitor stars
            input either this or metallicity (above)
            For FIRE-2, this should be *total* metals (everything not H, He).
            For FIRE-3, this should be iron.
        element_name : str [optional]
            name of element to get fractional mass loss of
            if None or '', get mass loss from all elements

        Returns
        -------
        mass_loss_fractions : float or array
            fractional mass loss[es] relative to mass of stars at formation
        '''
        return (
            self.StellarWind.get_mass_loss(
                age_min, age_maxs, metallicity, metal_mass_fraction, element_name=element_name
            )
            + self.SupernovaCC.get_mass_loss(
                age_min, age_maxs, element_name=element_name, metallicity=metallicity
            )
            + self.SupernovaIa.get_mass_loss(age_min, age_maxs, element_name=element_name)
        )

    def get_mass_loss_from_spline(self, ages, metallicities=None, metal_mass_fractions=None):
        '''
        Get fractional mass loss relative to mass of stars at formation via all stellar evolution
        channels at ages and metallicities (or metal mass fractions) via 2-D (bivariate) spline.

        Parameters
        ----------
        ages : float or array
            age[s] of stellar population [Myr]
        metallicities : float or array
            metallicity [linear wrt Solar] of progenitor, for scaling the wind rates
            input either this or metal_mass_fraction (below)
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02].
            For FIRE-3, this should be iron abundance [scaled to Solar := 1.38e-3].
        metal_mass_fractions : float or array
            optional: mass fraction of given metal in progenitor stars
            input either this or metallicity (above)
            For FIRE-2, this should be *total* metals (everything not H, He).
            For FIRE-3, this should be iron.

        Returns
        -------
        mass_loss_fractions : float or array
            fractional mass loss[es] relative to mass of stars at formation
        '''
        if metal_mass_fractions is not None:
            # convert mass fraction to metallicity using Solar value assumed in FIRE
            if 'fire2' in self.model:
                metallicities = metal_mass_fractions / self.sun_massfraction['metals']
            elif 'fire3' in self.model:
                metallicities = metal_mass_fractions / self.sun_massfraction['iron']

        assert np.isscalar(ages) or np.isscalar(metallicities) or len(ages) == len(metallicities)

        if self.Spline is None:
            self._make_mass_loss_spline()

        mass_loss_fractions = self.Spline.ev(ages, metallicities)

        if np.isscalar(ages) and np.isscalar(metallicities):
            mass_loss_fractions = np.asscalar(mass_loss_fractions)

        return mass_loss_fractions

    def _make_mass_loss_spline(
        self,
        age_limits=[1, 13700],
        age_bin_number=20,
        metallicity_limits=[0.01, 3],
        metallicity_bin_number=25,
        force_remake=False,
        write_spline=False,
    ):
        '''
        Create 2-D bivariate spline (in age and metallicity) for fractional mass loss
        relative to mass of stars at formation via all stellar evolution channels.

        Parameters
        ----------
        age_limits : list
            min and max limits of age of stellar population [Myr]
        age_bin_number : int
            number of age bins within age_limits
        metallicity_limits : list
            min and max limits of (linear) metallicity
        metallicity_bin_number : float
            number of metallicity bins
        force_remake : bool
            whether to force a recalculation of the spline, even if file exists
        write_spline : bool
            whether to write the spline to a pickle file for rapid loading in the future
        '''
        from os.path import isfile
        from scipy import interpolate

        if not force_remake and isfile(self._file_name):
            self._read_mass_loss_spline()
            return

        age_min = 0

        self.AgeBin = ut.binning.BinClass(age_limits, number=age_bin_number, log_scale=True)
        self.MetalBin = ut.binning.BinClass(
            metallicity_limits, number=metallicity_bin_number, log_scale=True
        )

        self.say('* generating 2-D spline to compute stellar mass loss from age + metallicity')
        self.say(f'number of age bins = {self.AgeBin.number}')
        self.say(f'number of metallicity bins = {self.MetalBin.number}')

        self.mass_loss_fractions = np.zeros((self.AgeBin.number, self.MetalBin.number))
        for metallicity_i, metallicity in enumerate(self.MetalBin.mins):
            self.mass_loss_fractions[:, metallicity_i] = self.get_mass_loss(
                age_min, self.AgeBin.mins, metallicity
            )

        self.Spline = interpolate.RectBivariateSpline(
            self.AgeBin.mins, self.MetalBin.mins, self.mass_loss_fractions
        )

        if write_spline:
            self._write_mass_loss_spline()

    def _write_mass_loss_spline(self):
        with open(self._file_name, 'wb') as f:
            pickle.dump(self.Spline, f)
        self.say(f'wrote mass-loss spline to:  {self._file_name}')

    def _read_mass_loss_spline(self):
        with open(self._file_name, 'rb') as f:
            self.Spline = pickle.load(f)
            self.say(f'read mass-loss spline from:  {self._file_name}')


MassLoss = MassLossClass()


def plot_supernova_number_v_age(
    axis_y_kind='rate',
    axis_y_limits=None,
    axis_y_log_scale=True,
    age_limits=[1, 13700],
    age_bin_width=0.1,
    age_log_scale=True,
    file_name=False,
    directory='.',
    figure_index=1,
):
    '''
    Plot specific rates or cumulative numbers [per M_sun of stars at formation] of
    core-collapse and Ia supernova events versus stellar age [Myr].

    Parameters
    ----------
    axis_y_kind : str
        'rate' or 'number'
    axis_y_limits : list
        min and max limits to impose on y axis
    axis_y_log_scale : bool
        whether to duse logarithmic scaling for y axis
    age_limits : list
        min and max limits of age of stellar population [Myr]
    age_bin_width : float
        width of stellar age bin [Myr]
    age_log_scale : bool
        whether to use logarithmic scaling for age bins
    file_name : str
        whether to write figure to file, and set its name: True = use default naming convention
    directory : str
        where to write figure file
    figure_index : int
        index for matplotlib window
    '''
    assert axis_y_kind in ['rate', 'number']

    AgeBin = ut.binning.BinClass(age_limits, age_bin_width, include_max=True)

    CC_FIRE2 = SupernovaCCClass(model='fire2')
    CC_FIRE3 = SupernovaCCClass(model='fire3')
    Ia_FIRE2 = SupernovaIaClass(model='fire2')
    SupernovaIa3 = SupernovaIaClass(model='fire3')

    if axis_y_kind == 'rate':
        cc_fire2 = CC_FIRE2.get_rate(AgeBin.mins)
        cc_fire3 = CC_FIRE3.get_rate(AgeBin.mins)
        ia_fire2 = Ia_FIRE2.get_rate(AgeBin.mins)
        ia_fire3 = SupernovaIa3.get_rate(AgeBin.mins)
    elif axis_y_kind == 'number':
        cc_fire2 = CC_FIRE2.get_number(min(age_limits), AgeBin.maxs)
        cc_fire3 = CC_FIRE3.get_number(min(age_limits), AgeBin.maxs)
        ia_fire2 = Ia_FIRE2.get_number(min(age_limits), AgeBin.maxs)
        ia_fire3 = SupernovaIa3.get_number(min(age_limits), AgeBin.maxs)
        if axis_y_limits is None or len(axis_y_limits) == 0:
            axis_y_limits = [5e-5, 2e-2]

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, left=0.20)

    ut.plot.set_axes_scaling_limits(
        subplot,
        age_log_scale,
        age_limits,
        None,
        axis_y_log_scale,
        axis_y_limits,
        [cc_fire2, cc_fire3, ia_fire2, ia_fire3],
    )

    subplot.set_xlabel('stellar age $\\left[ {\\rm Myr} \\right]$')
    if axis_y_kind == 'rate':
        subplot.set_ylabel('SN rate $\\left[ {\\rm Myr}^{-1} {\\rm M}_\odot^{-1} \\right]$')
    elif axis_y_kind == 'number':
        subplot.set_ylabel('SN number $\\left[ {\\rm M}_\odot^{-1} \\right]$')

    colors = ut.plot.get_colors(4, use_black=False)

    subplot.plot(AgeBin.mins, cc_fire2, color=colors[0], alpha=0.8, label='CC (FIRE-2)')
    subplot.plot(AgeBin.mins, ia_fire2, color=colors[1], alpha=0.8, label='Ia (FIRE-2)')
    subplot.plot(AgeBin.mins, cc_fire3, color=colors[2], alpha=0.8, label='CC (FIRE-3)')
    subplot.plot(AgeBin.mins, ia_fire3, color=colors[3], alpha=0.8, label='Ia (FIRE-3)')

    print('CC FIRE-2')
    print('{:.4f}'.format(cc_fire2[-1]))
    print('Ia FIRE-2')
    print('{:.4f}'.format(ia_fire2[-1]))
    print('CC FIRE-3')
    print('{:.4f}'.format(cc_fire3[-1]))
    print('Ia FIRE-3')
    print('{:.4f}'.format(ia_fire3[-1]))

    ut.plot.make_legends(subplot, 'best')

    if file_name is True or file_name == '':
        if axis_y_kind == 'rate':
            file_name = 'supernova.rate_v_time'
        elif axis_y_kind == 'number':
            file_name = 'supernova.number_v_time'
    ut.plot.parse_output(file_name, directory)


def plot_mass_loss_v_age(
    mass_loss_kind='rate',
    mass_loss_limits=None,
    mass_loss_log_scale=True,
    element_name=None,
    metallicity=1,
    metal_mass_fraction=None,
    model=FIRE_MODEL_DEFAULT,
    age_limits=[1, 13700],
    age_bin_width=0.01,
    age_log_scale=True,
    file_name=False,
    directory='.',
    figure_index=1,
):
    '''
    Plot fractional mass loss relative to mass of stars at formation from all stellar evolution
    channels (stellar winds, core-collapse and Ia supernovae) versus stellar age [Myr].

    Parameters
    ----------
    mass_loss_kind : str
        'rate' or 'mass'
    mass_loss_limits : list
        min and max limits to impose on y-axis
    mass_loss_log_scale : bool
        whether to use logarithmic scaling for age bins
    element_name : str
        name of element to get yield of (if None, compute total mass loss)
    metallicity : float
        (linear) total abundance of metals wrt Solar
    metal_mass_fraction : float
        mass fration of all metals (everything not H, He)
    model : str
        model for rates: 'fire2', 'fire3'
    age_limits : list
        min and max limits of age of stellar population [Myr]
    age_bin_width : float
        width of stellar age bin [Myr]
    age_log_scale : bool
        whether to use logarithmic scaling for age bins
    file_name : str
        whether to write figure to file and its name. True = use default naming convention
    directory : str
        directory in which to write figure file
    figure_index : int
        index for matplotlib window
    '''
    mass_loss_kind = mass_loss_kind.lower()
    assert mass_loss_kind in ['rate', 'mass']

    AgeBin = ut.binning.BinClass(
        age_limits, age_bin_width, include_max=True, log_scale=age_log_scale
    )

    StellarWind = StellarWindClass(model)
    SupernovaCC = SupernovaCCClass(model)
    SupernovaIa = SupernovaIaClass(model)

    if mass_loss_kind == 'rate':
        wind = StellarWind.get_mass_loss_rate(
            AgeBin.mins, metallicity, metal_mass_fraction, element_name=element_name
        )
        supernova_cc = SupernovaCC.get_mass_loss_rate(
            AgeBin.mins, element_name=element_name, metallicity=metallicity
        )
        supernova_Ia = SupernovaIa.get_mass_loss_rate(AgeBin.mins, element_name=element_name)
    else:
        age_min = 0
        wind = StellarWind.get_mass_loss(
            age_min, AgeBin.mins, metallicity, metal_mass_fraction, element_name=element_name
        )
        supernova_cc = SupernovaCC.get_mass_loss(
            age_min, AgeBin.mins, element_name=element_name, metallicity=metallicity
        )
        supernova_Ia = SupernovaIa.get_mass_loss(age_min, AgeBin.mins, element_name=element_name)

    total = supernova_cc + supernova_Ia + wind

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot,
        age_log_scale,
        age_limits,
        None,
        mass_loss_log_scale,
        mass_loss_limits,
        [supernova_cc, supernova_Ia, wind, total],
    )

    subplot.set_xlabel('star age $\\left[ {\\rm Myr} \\right]$')
    if mass_loss_kind == 'rate':
        subplot.set_ylabel('mass loss rate $\\left[ {\\rm Myr}^{-1} \\right]$')
    else:
        axis_y_label = 'fractional mass loss'
        if element_name:
            axis_y_label = f'{element_name} yield per ${{\\rm M}}_\odot$'
        subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(3, use_black=False)

    subplot.plot(AgeBin.mins, wind, color=colors[0], alpha=0.7, label='stellar winds')
    subplot.plot(AgeBin.mins, supernova_cc, color=colors[1], alpha=0.7, label='supernova cc')
    subplot.plot(AgeBin.mins, supernova_Ia, color=colors[2], alpha=0.7, label='supernova Ia')
    subplot.plot(AgeBin.mins, total, color='black', alpha=0.8, label='total')

    print('wind')
    print('{:.4f}'.format(wind[-1]))
    print('CC')
    print('{:.4f}'.format(supernova_cc[-1]))
    print('Ia')
    print('{:.4f}'.format(supernova_Ia[-1]))
    print('total')
    print('{:.4f}'.format(total[-1]))

    ut.plot.make_legends(subplot, 'best')

    if file_name is True or file_name == '':
        if element_name is not None and len(element_name) > 0:
            file_name = f'{element_name}.yield_v_time'
            if 'rate' in mass_loss_kind:
                file_name = file_name.repace('.yield', '.yield.rate')
        else:
            file_name = 'star.mass.loss_v_time'
            if 'rate' in mass_loss_kind:
                file_name = file_name.replace('.loss', '.loss.rate')
        file_name += '_Z.{}'.format(
            ut.io.get_string_from_numbers(metallicity, digits=4, exponential=False, strip=True)
        )
    ut.plot.parse_output(file_name, directory)


def plot_nucleosynthetic_yields(
    event_kinds='wind',
    metallicity=1,
    model=FIRE_MODEL_DEFAULT,
    axis_y_limits=[1e-3, 5],
    axis_y_log_scale=True,
    file_name=False,
    directory='.',
    figure_index=1,
):
    '''
    Plot nucleosynthetic yield mass [M_sun] v element name, for input event_kind[s].

    Parameters
    ----------
    event_kinds : str or list
        stellar event: 'wind', 'supernova.cc', 'supernova.ia', 'all'
    metallicity : float
        metallicity of progenitor [linear mass fraction wrt to Solar]
    model : str
        stellar evolution model for yields: 'fire2', 'fire3'
    axis_y_limits : list
        min and max limits of y axis
    axis_y_log_scale: bool
        whether to use logarithmic scaling for y axis
    file_name : str
        whether to write figure to file and its name. True = use default naming convention
    directory : str
        directory to write figure file
    figure_index : int
        index of figure for matplotlib
    '''
    title_dict = {
        'wind': 'winds',
        'supernova.cc': 'SN CC',
        'supernova.ia': 'SN Ia',
    }

    if event_kinds == 'all':
        event_kinds = ['wind', 'supernova.cc', 'supernova.ia']
    elif np.isscalar(event_kinds):
        event_kinds = [event_kinds]

    NucleosyntheticYield = NucleosyntheticYieldClass(model)
    element_yield_dict = collections.OrderedDict()
    for element_name in NucleosyntheticYield.sun_massfraction.keys():
        if element_name != 'metals':
            element_yield_dict[element_name] = 0

    # plot ----------
    _fig, subplots = ut.plot.make_figure(
        figure_index, panel_numbers=[1, len(event_kinds)], top=0.92
    )

    colors = ut.plot.get_colors(len(element_yield_dict), use_black=False)

    for ei, event_kind in enumerate(event_kinds):
        subplot = subplots[ei]

        if 'fire2' in model:
            # get mass of element ejecta [M_sun]
            element_yield_t = NucleosyntheticYield.get_element_yields(
                event_kind, metallicity, return_mass=True
            )
            for element_name in element_yield_dict:
                element_yield_dict[element_name] = element_yield_t[element_name]

        elif 'fire3' in model:
            age_min = 0
            age_max = 13700
            if event_kind == 'wind':
                StellarWind = StellarWindClass(model)
                for element_name in element_yield_dict:
                    element_yield_dict[element_name] = StellarWind.get_mass_loss(
                        age_min, age_max, metallicity=metallicity, element_name=element_name,
                    )
            elif event_kind == 'supernova.cc':
                SupernovaCC = SupernovaCCClass(model)
                for element_name in element_yield_dict:
                    element_yield_dict[element_name] = SupernovaCC.get_mass_loss(
                        age_min, age_max, metallicity=metallicity, element_name=element_name,
                    )
            elif event_kind == 'supernova.ia':
                SupernovaIa = SupernovaIaClass(model)
                for element_name in element_yield_dict:
                    element_yield_dict[element_name] = SupernovaIa.get_mass_loss(
                        age_min, age_max, element_name=element_name,
                    )

        element_yields = [element_yield_dict[e] for e in element_yield_dict]
        element_labels = [
            str.capitalize(ut.constant.element_symbol_from_name[e]) for e in element_yield_dict
        ]
        element_indices = np.arange(len(element_yield_dict))

        ut.plot.set_axes_scaling_limits(
            subplot,
            x_limits=[element_indices.min() - 0.5, element_indices.max() + 0.5],
            y_log_scale=axis_y_log_scale,
            y_limits=axis_y_limits,
            y_values=element_yields,
        )

        # subplot.set_xticks(element_indices)
        # subplot.set_xticklabels(element_labels)
        subplot.tick_params(top=False)
        subplot.tick_params(bottom=False)
        subplot.tick_params(right=False)

        if ei == 0:
            subplot.set_ylabel('yield $\\left[ {\\rm M}_\odot \\right]$')
        if ei == 1:
            subplot.set_xlabel('element')

        for i in element_indices:
            if element_yields[i] > 0:
                subplot.plot(
                    element_indices[i], element_yields[i], 'o', markersize=10, color=colors[i]
                )
                # add element symbols near points
                subplot.text(element_indices[i] * 0.98, element_yields[i] * 0.5, element_labels[i])

        if ei == 0:
            metal_label = ut.io.get_string_from_numbers(metallicity, exponential=None, strip=True)
            ut.plot.make_label_legend(subplot, f'$Z / Z_\odot={metal_label}$')

        subplot.set_title(title_dict[event_kind])

    if file_name is True or file_name == '':
        file_name = f'{event_kind}.yields_Z.{metal_label}'
    ut.plot.parse_output(file_name, directory)
