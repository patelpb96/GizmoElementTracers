'''
This module contains the models for stellar evolution implemented in Gizmo for the
FIRE-2 and FIRE-3 models, specifically, nucleosynthetic yields and mass-loss rates for
    (1) core-collapse supernovae
    (2) Ia supernovae
    (3) stellar winds

These nucleosynthetic yields and mass-loss rates depend on progenitor metallicity:
    stellar wind: overall mass-loss rate and oxygen yield
    core-collapse supernova: nitrogen yield

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    time [Myr] (different than most other modules in this package, which default to Gyr)
    elemental abundance [(linear) mass fraction]

TODO for FIRE-3
    add solar abundances
    ensure scales to Iron
'''

import collections
import numpy as np
from scipy import integrate

import utilities as ut

# default rate + yield model to assume throughout
DEFAULT_MODEL = 'fire2'


# --------------------------------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------------------------------
def get_sun_massfraction(model=DEFAULT_MODEL):
    '''
    Get dictionary of Solar abundances (mass fractions).
    These mass fractions may differ by up to a percent from the values in utilities.constant,
    given choices of mean atomic mass.

    Parameters
    ----------
    model : str
        stellar evolution model: 'fire2', 'fire3'
    '''

    model = model.lower()
    assert model in ['fire2', 'fire2.1', 'fire2.2', 'fire2.3', 'fire3']

    sun_massfraction = {}

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
        sun_massfraction['sulphur'] = 6.44e-4
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
        sun_massfraction['sulphur'] = 3.31e-4
        sun_massfraction['calcium'] = 6.87e-5
        sun_massfraction['iron'] = 1.38e-3

    return sun_massfraction


def get_ages_critical(model=DEFAULT_MODEL):
    '''
    Get array of critical ages [Myr] that mark transition points in given stellar evolution model.
    Use to supply to numerical integrators.

    Parameters
    ----------
    model : str
        stellar evolution model: 'fire2', 'fire3'
    '''
    model = model.lower()
    assert model in ['fire2', 'fire2.1', 'fire2.2', 'fire2.3', 'fire3']

    if 'fire2' in model:
        ages_critical = np.sort([3.401, 10.37, 37.53, 1, 50, 100, 1000, 13800])  # [Myr]
    elif model == 'fire3':
        ages_critical = None  # [Myr]

    return ages_critical


# --------------------------------------------------------------------------------------------------
# nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class NucleosyntheticYieldClass:
    '''
    Nucleosynthetic yields in the FIRE-2 or FIRE-3 models.

    Metallicity dependent yields:
        for stellar wind: oxygen
        for core-collpase supernova: nitgrogen
        for Ia supernova: none
    '''

    def __init__(self, model=DEFAULT_MODEL):
        '''
        Store Solar elemental abundances, as linear mass fractions.

        FIRE-2 uses values from Anders & Grevesse 1989.
        FIRE-3 uses proto-solar values from Asplund et al 2009.

        Parameters
        ----------
        model : str
            stellar evolution model for yields: 'fire2', 'fire3'
        '''
        self.models_available = ['fire2', 'fire2.1', 'fire.2.2', 'fire3']
        self._parse_model(model)

        self.wind_yield = None
        self.sncc_yield = None
        self.snia_yield = None

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                # reset solar abundances
                self.sun_massfraction = get_sun_massfraction(model)
            self.model = model

        assert self.model in self.models_available

    def get_yields(
        self,
        event_kind='supernova.cc',
        progenitor_metallicity=1.0,
        progenitor_massfraction_dict={},
        age=None,
        model=None,
        normalize=True,
    ):
        '''
        Get stellar nucleosynthetic yields for input event_kind in the FIRE-2 or FIRE-3 model.

        This computes the *additional* nucleosynthetic yields that Gizmo adds to the star's
        existing metallicity, so these are not the actual yields that get deposited to gas.
        The total mass fraction (wrt total mass of star) returned for each element is:
            (1 - ejecta_total_mass_fraction) * star_element_mass_fraction
                + ejecta_total_mass_fraction * ejecta_element_mass_fraction

        Parameters
        ----------
        event_kind : str
            stellar event: 'wind', 'supernova.cc' or 'supernova.ii', 'supernova.ia'
        progenitor_metallicity : float
            total metallicity of progenitor stellar population
            [linear mass fraction relative to sun_mass_fraction['metals']]
        progenitor_massfraction_dict : dict or bool [optional]
            optional: dictionary that contains the mass fraction of each element in the progenitor
            if True, then assume Solar abundance ratios and use progenitor_metallicity to normalize
            use this to calculate higher-order correction of yields as used in FIRE
        normalize : bool
            whether to normalize yields to be mass fractions (wrt formation mass), instead of masses

        Returns
        -------
        element_yield : ordered dict
            stellar nucleosynthetic yield for each element,
            in mass [M_sun] or mass fraction (wrt formation mass)
        '''
        element_yield = collections.OrderedDict()
        for element_name in self.sun_massfraction:
            element_yield[element_name] = 0.0

        event_kind = event_kind.lower()
        assert event_kind in ['wind', 'supernova.cc', 'supernova.ii', 'supernova.ia']

        self._parse_model(model)

        # determine progenitor abundance[s]
        if isinstance(progenitor_massfraction_dict, dict) and len(progenitor_massfraction_dict) > 0:
            # input mass fraction for each element, use to compute higher-order corrections
            for element_name in element_yield:
                assert element_name in progenitor_massfraction_dict
            progenitor_metal_mass_fraction = progenitor_massfraction_dict['metals']
        else:
            assert progenitor_metallicity >= 0
            progenitor_metal_mass_fraction = (
                progenitor_metallicity * self.sun_massfraction['metals']
            )
            if progenitor_massfraction_dict is True:
                progenitor_massfraction_dict = {}
                # assume Solar abundance ratios and use progenitor_metallicity to normalize
                for element_name in self.sun_massfraction:
                    progenitor_massfraction_dict[element_name] = (
                        progenitor_metallicity * self.sun_massfraction[element_name]
                    )

        if event_kind == 'wind':
            ejecta_mass = 1  # stellar wind yields are intrinsically mass fractions

            if 'fire2' in self.model:
                # FIRE-2: stellar_evolution.c line 583
                # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
                # mass fractions
                element_yield['helium'] = 0.36
                element_yield['carbon'] = 0.016
                element_yield['nitrogen'] = 0.0041
                element_yield['oxygen'] = 0.0118

                if self.model != 'fire2.2':
                    # oxygen yield increases linearly with progenitor metallicity at Z/Z_sun < 1.65
                    if progenitor_metal_mass_fraction < 0.033:
                        element_yield['oxygen'] *= (
                            progenitor_metal_mass_fraction / self.sun_massfraction['metals']
                        )
                    else:
                        element_yield['oxygen'] *= 1.65

            elif self.model == 'fire3':
                # FIRE-3: stellar_evolution.c line 563
                # everything except He and CNO and S-process is well approximated by surface
                # abundances. CNO is conserved to high accuracy in sum for secondary production

                # define initial H, He, CNO fraction
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
                t = age / 1e3  # convert to [Gyr]
                # solar-scaled CNO abundance
                z_sol = f_CNO_0 / (
                    self.sun_massfraction['carbon']
                    + self.sun_massfraction['nitrogen']
                    + self.sun_massfraction['oxygen ']
                )
                # He production : this scales off of the fraction of H in IC:
                # y represents the yield of He produced by burning H, scales off availability
                t1 = 0.0028
                t2 = 0.01
                t3 = 2.3
                t4 = 3.0
                y1 = 0.4 * np.min((z_sol + 1e-3) ** 0.6, 2)
                y2 = 0.08
                y3 = 0.07
                y4 = 0.042
                if t < t1:
                    y = y1 * (t / t1) ** 3
                elif t < t2:
                    y = y1 * (t / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif t < t3:
                    y = y2 * (t / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
                elif t < t4:
                    y = y3 * (t / t3) ** (np.log(y4 / y3) / np.log(t4 / t3))
                else:
                    y = y4

                element_yield['helium'] = f_He_0 + y * f_H_0

                # secondary N production in CNO cycle: scales off of initial fraction of CNO:
                # y here represents fraction of CO mass converted to -additional- N
                t1 = 0.001
                t2 = 0.0028
                t3 = 0.05
                t4 = 1.9
                t5 = 14.0
                y1 = 0.2 * np.max(1e-4, np.min(z_sol * z_sol, 0.9))
                y2 = 0.68 * np.min((z_sol + 1e-3) ** 0.1, 0.9)
                y3 = 0.4
                y4 = 0.23
                y5 = 0.065
                if t < t1:
                    y = y1 * (t / t1) ** 3.5
                elif t < t2:
                    y = y1 * (t / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif t < t3:
                    y = y2 * (t / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
                elif t < t4:
                    y = y3 * (t / t3) ** (np.log(y4 / y3) / np.log(t4 / t3))
                elif t < t5:
                    y = y4 * (t / t4) ** (np.log(y5 / y4) / np.log(t5 / t4))
                else:
                    y = y5
                y = np.max(0, np.min(1, y))
                frac_loss_from_C = 0.5
                floss_CO = y * (f_C_0 + f_O_0)
                floss_C = np.min(frac_loss_from_C * floss_CO, 0.99 * f_C_0)
                floss_O = floss_CO - floss_C
                # convert mass from CO to N, conserving total CNO mass
                element_yield['nitrogen'] = f_N_0 + floss_CO
                element_yield['carbon'] = f_C_0 - floss_C
                element_yield['oxygen'] = f_O_0 - floss_O

                # primary C production: scales off initial H+He, generally small compared to loss
                # fraction above in SB99, large in some other models, small for early OB winds
                t1 = 0.005
                t2 = 0.04
                t3 = 10
                y1 = 1.0e-6
                y2 = 0.001
                y3 = 0.005
                if t < t1:
                    y = y1 * (t / t1) ** 3
                elif t < t2:
                    y = y1 * (t / t1) ** (np.log(y2 / y1) / np.log(t2 / t1))
                elif t < t3:
                    y = y2 * (t / t2) ** (np.log(y3 / y2) / np.log(t3 / t2))
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
                ejecta_mass = 10.5  # [M_sun]
                # mass fractions
                element_yield['metals'] = 0.19
                element_yield['helium'] = 0.369
                element_yield['carbon'] = 0.0127
                element_yield['nitrogen'] = 0.00456
                element_yield['oxygen'] = 0.111
                element_yield['neon'] = 0.0286
                if self.model in ['fire2.1', 'fire2.2', 'fire2.3']:
                    element_yield['neon'] = 0.0381  # used in later simulations
                element_yield['magnesium'] = 0.00940
                element_yield['silicon'] = 0.00889
                element_yield['sulphur'] = 0.00378
                element_yield['calcium'] = 0.000436  # Nomoto et al 2013 suggest 0.05 - 0.1 M_sun
                element_yield['iron'] = 0.00706

                if model != 'fire2.2':
                    yield_nitrogen_orig = np.float(element_yield['nitrogen'])

                    # nitrogen yield increases linearly with progenitor metallicity @ Z/Z_sun < 1.65
                    if progenitor_metal_mass_fraction < 0.033:
                        element_yield['nitrogen'] *= (
                            progenitor_metal_mass_fraction / self.sun_massfraction['metals']
                        )
                    else:
                        element_yield['nitrogen'] *= 1.65
                    # correct total metal mass for nitrogen
                    element_yield['metals'] += element_yield['nitrogen'] - yield_nitrogen_orig

            elif self.model == 'fire3':
                # FIRE-3: stellar_evolution.c line 471 (or so)
                ejecta_mass = 8.72  # [M_sun]

                # numbers for interpolation of ejecta masses
                # [must be careful here that this integrates to the correct -total- ejecta mass]
                # these break times: tmin=3.7 Myr corresponds to the first explosions
                # (Eddington-limited lifetime of the most massive stars), tbrk=6.5 Myr to the end of
                # this early phase, stars with ZAMS mass ~30+ Msun here. curve flattens both from
                # IMF but also b/c mass-loss less efficient. tmax=44 Myr to the last explosion
                # determined by lifetime of 8 Msun stars
                t_min = 0.0037
                t_brk = 0.0065
                t_max = 0.044
                t = age / 1e3
                Mmax = 35
                Mbrk = 10
                Mmin = 6
                # power-law interpolation of ejecta mass over duration of C-C phase
                if t <= tbrk:
                    Msne = Mmax * (t / t_min) ** (np.log(Mbrk / Mmax) / np.log(t_brk / t_min))
                else:
                    Msne = Mbrk * (t / t_brk) ** (np.log(Mmin / Mbrk) / np.log(t_max / t_brk))
                tvec = np.array([3.7, 8, 18, 30, 44])  # [Myr]
                fvec = np.array(
                    [
                        # He [IMF-mean y=3.67e-01] - have to remove normal solar correction and take
                        # care with winds
                        [4.61e-01, 3.30e-01, 3.58e-01, 3.65e-01, 3.59e-01],
                        # C [IMF-mean y=3.08e-02] - care needed in fitting out winds: wind=6.5e-3,
                        # ejecta_only=1.0e-3
                        [2.37e-01, 8.57e-03, 1.69e-02, 9.33e-03, 4.47e-03],
                        # N [IMF-mean y=4.47e-03] - care needed with winds, but not as essential
                        [1.07e-02, 3.48e-03, 3.44e-03, 3.72e-03, 3.50e-03],
                        # O [IMF-mean y=7.26e-02] - reasonable - generally IMF-integrated
                        # alpha-element total mass-yields lower versus fire-2 by factor ~0.7 or so
                        [9.53e-02, 1.02e-01, 9.85e-02, 1.73e-02, 8.20e-03],
                        # Ne [IMF-mean y=1.58e-02] - roughly a hybrid of fit direct to ejecta and
                        # fit to all mass as above, truncating at highest masses
                        [2.60e-02, 2.20e-02, 1.93e-02, 2.70e-03, 2.75e-03],
                        # Mg [IMF-mean y=9.48e-03] - fit directly on ejecta and ignore mass-fraction
                        # rescaling because that is not reliable at early times:
                        # this gives a reasonable vnumber.
                        # important to note that early supernovae strongly dominate Mg
                        [2.89e-02, 1.25e-02, 5.77e-03, 1.03e-03, 1.03e-03],
                        # Si [IMF-mean y=4.53e-03]
                        # lots comes from 1a's, so low here is not an issue
                        [4.12e-04, 7.69e-03, 8.73e-03, 2.23e-03, 1.18e-03],
                        # S [IMF-mean y=3.01e-03] - more from Ia's
                        [3.63e-04, 5.61e-03, 5.49e-03, 1.26e-03, 5.75e-04],
                        # Ca [IMF-mean y=2.77e-04] - Ia
                        [4.28e-05, 3.21e-04, 6.00e-04, 1.84e-04, 9.64e-05],
                        # Fe [IMF-mean y=4.11e-03] - Ia
                        [5.46e-04, 2.18e-03, 1.08e-02, 4.57e-03, 1.83e-03],
                    ]
                )
                # compare nomoto 2006:
                # y = [He: 3.69e-1, C: 1.27e-2, N: 4.56e-3, O: 1.11e-1, Ne: 3.81e-2, Mg: 9.40e-3,
                # Si: 8.89e-3, S: 3.78e-3, Ca: 4.36e-4, Fe: 7.06e-3]
                # use the fit parameters above for the piecewise power-law components to define the
                # yields at each time
                i_t = -1
                for k in range(len(tvec)):
                    if age > tvec[k]:
                        i_t = k
                for k in range(10):
                    i_y = k + 1
                    if i_t < 0:
                        yields[i_y] = fvec[k][0]
                    elif i_t >= i_tvec - 1:
                        yields[i_y] = fvec[k][i_tvec - 1]
                    else:
                        yields[i_y] = fvec[k][i_t] * (age / tvec[i_t]) ** (
                            np.log(fvec[k][i_t + 1] / fvec[k][i_t])
                            / np.log(tvec[i_t + 1] / tvec[i_t])
                        )
                # sum heavy element yields to get the total metal yield, multiplying by a small
                # correction term to account for trace species not explicitly followed above
                # [mean for CC]
                element_yield['metals'] = 0
                for element_name in element_yield:
                    if element_name not in ['metals', 'helium']:
                        # assume that there is some trace species proportional to each species,
                        # not correct in detail, but a tiny correction, so negligible
                        element_yield['metals'] += 1.0144 * element_yield[element_name]

        elif event_kind == 'supernova.ia':
            ejecta_mass = 1.4  # [M_sun]

            if 'fire2' in self.model:
                # FIRE-2: stellar_evolution.c line 498 (or so)
                # yields from Iwamoto et al 1999, W7 model, IMF averaged
                # mass fractions
                element_yield['metals'] = 1
                element_yield['helium'] = 0
                element_yield['carbon'] = 0.035
                element_yield['nitrogen'] = 8.57e-7
                element_yield['oxygen'] = 0.102
                element_yield['neon'] = 0.00321
                element_yield['magnesium'] = 0.00614
                element_yield['silicon'] = 0.111
                element_yield['sulphur'] = 0.0621
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
                element_yield['sulphur'] = 7.62e-2
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
                # Seitenzahl et al. 2013, model N100 [favored]
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
            isinstance(progenitor_massfraction_dict, dict)
            and len(progenitor_massfraction_dict) > 0
            and (self.model == 'fire2' or (self.model == 'fire2.1' and 'supernova' in event_kind))
        ):
            # FIRE-2: stellar_evolution.c line 509
            # enforce that yields obey pre-existing surface abundances
            # allow for larger abundances in the progenitor star - usually irrelevant

            # get pure (non-metal) mass fraction of star
            pure_mass_fraction = 1 - progenitor_metal_mass_fraction

            for element_name in element_yield:
                if element_yield[element_name] > 0:

                    # apply (new) yield only to pure (non-metal) mass of star
                    element_yield[element_name] *= pure_mass_fraction
                    # correction relative to solar abundance
                    element_yield[element_name] += (
                        progenitor_massfraction_dict[element_name]
                        - self.sun_massfraction[element_name]
                    )
                    element_yield[element_name] = np.clip(element_yield[element_name], 0, 1)

        if not normalize:
            # convert yield mass fractions to masses [M_sun]
            for element_name in element_yield:
                element_yield[element_name] *= ejecta_mass

        return element_yield

    def assign_yields(self, progenitor_metallicity=None, progenitor_massfraction_dict=True):
        '''
        Store nucleosynthetic yields from all stellar channels, for a fixed progenitor metallicity,
        as dictionaries with element names as kwargs and yields [M_sun] as values.
        Useful to avoid having to re-call get_yields() many times.

        Parameters
        -----------
        progenitor_metallicity : float
            metallicity [wrt Solar] for yields that depend on progenitor metallicity
        progenitor_massfraction_dict : dict or bool [optional]
            optional: dictionary that contains the mass fraction of each element in the progenitor
            if True, then assume Solar abundance ratios and use progenitor_metallicity to normalize
            use this to calculate higher-order correction of yields as used in FIRE
        '''
        # store yields from stellar winds as intrinsically mass fractions (equivalent to 1 M_sun)
        self.wind_yield = self.get_yields(
            'wind', progenitor_metallicity, progenitor_massfraction_dict
        )

        # store yields from supernova as masses [M_sun]
        self.sncc_yield = self.get_yields(
            'supernova.cc', progenitor_metallicity, progenitor_massfraction_dict, normalize=False
        )

        self.snia_yield = self.get_yields(
            'supernova.ia', progenitor_metallicity, progenitor_massfraction_dict, normalize=False
        )


NucleosyntheticYield = NucleosyntheticYieldClass()


def plot_nucleosynthetic_yields(
    event_kind='wind',
    star_metallicity=0.1,
    star_massfraction={},
    model=DEFAULT_MODEL,
    normalize=False,
    axis_y_limits=[1e-3, None],
    axis_y_log_scale=True,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot nucleosynthetic element yields, according to input event_kind.

    Parameters
    ----------
    event_kind : str
        stellar event: 'wind', 'supernova.cc' or supernova.ii', 'supernova.ia'
    star_metallicity : float
        total metallicity of star, fraction wrt to solar
    star_massfraction : dict
        dictionary of elemental mass fractions in star
        need to input this to get higher-order correction of yields
    model : str
        stellar evolution model for yields: 'fire2', 'fire3'
    normalize : bool
        whether to normalize yields to be mass fractions (instead of masses)
    axis_y_limits : list
        min and max limits of y axis
    axis_y_log_scale: bool
        whether to use logarithmic scaling for y axis
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory to write figure file
    figure_index : int
        index of figure for matplotlib
    '''
    title_dict = {
        'wind': 'stellar wind',
        'supernova.cc': 'supernova CC',
        'supernova.ii': 'supernova CC',
        'supernova.ia': 'supernova Ia',
    }
    NucleosyntheticYield = NucleosyntheticYieldClass(model)
    element_yield = NucleosyntheticYield.get_yields(
        event_kind, star_metallicity, star_massfraction, normalize=normalize
    )

    yield_indices = np.arange(1, len(element_yield))
    yield_names = np.array([k for k in element_yield])[yield_indices]
    yield_values = np.array([element_yield[k] for k in element_yield])[yield_indices]
    yield_labels = [str.capitalize(ut.constant.element_symbol_from_name[k]) for k in yield_names]
    yield_indices = np.arange(yield_indices.size)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index, top=0.92)
    subplots = [subplot]

    colors = ut.plot.get_colors(yield_indices.size, use_black=False)

    for si in range(1):
        ut.plot.set_axes_scaling_limits(
            subplots[si],
            x_limits=[yield_indices.min() - 0.5, yield_indices.max() + 0.5],
            y_log_scale=axis_y_log_scale,
            y_limits=axis_y_limits,
            y_values=yield_values,
        )

        subplots[si].set_xticks(yield_indices)
        subplots[si].set_xticklabels(yield_labels)

        if normalize:
            y_label = 'yield (mass fraction)'
        else:
            y_label = 'yield $\\left[ {\\rm M}_\odot \\right]$'
        subplots[si].set_ylabel(y_label)
        subplots[si].set_xlabel('element')

        for yi in yield_indices:
            if yield_values[yi] > 0:
                subplot.plot(
                    yield_indices[yi], yield_values[yi], 'o', markersize=10, color=colors[yi]
                )
                # add element symbols near points
                subplots[si].text(
                    yield_indices[yi] * 0.98, yield_values[yi] * 0.5, yield_labels[yi]
                )

        subplots[si].set_title(title_dict[event_kind])

        metal_label = ut.io.get_string_from_numbers(star_metallicity, exponential=None, strip=True)
        ut.plot.make_label_legend(subplots[si], f'$Z / Z_\odot={metal_label}$')

    if plot_file_name is True or plot_file_name == '':
        plot_file_name = f'{event_kind}.yields_Z.{metal_label}'
    ut.plot.parse_output(plot_file_name, plot_directory)


# --------------------------------------------------------------------------------------------------
# stellar mass loss
# --------------------------------------------------------------------------------------------------
class StellarWindClass:
    '''
    Compute mass-loss rates and cumulative mass-loss fractions (with respect to formation mass)
    for stellar winds in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=DEFAULT_MODEL):
        '''
        Parameters
        ----------
        model : str
             model for wind rate: 'fire2', 'fire3'
        '''
        self.ejecta_mass = 1.0  # for stellar winds, rates are mass fractions wrt formation mass

        self.models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire2.3', 'fire3']
        self._parse_model(model)

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                # reset solar abundances
                self.sun_massfraction = get_sun_massfraction(model)
            self.model = model

        assert self.model in self.models_available

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None, model=None):
        '''
        Get rate of fractional mass loss [Myr ^ -1, fractional wrt formation mass] from stellar
        winds in FIRE-2 or FIRE-3.
        Input either metallicity (linear, wrt Solar) or (raw) metal_mass_fraction.

        Includes all non-supernova mass-loss channels, but dominated by O, B, and AGB stars.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            metallicity [(linear) wrt Sun] of progenitor stars, for scaling the wind rates
            input either metallicity or metal_mass_fraction
            For FIRE-2, this should be *total* metallicity [scaled to Solar := 0.02]
            For FIRE-3, this should be Iron abundance [scaled to Solar := 1.38e-3]
        metal_mass_fraction : float [optional]
            mass fraction of all metals in progenitor stars, for scaling the wind rates
            input either metallicity or metal_mass_fraction
            For FIRE-2, this should be *total* metals (everything not H, He)
            For FIRE-3, this should be Iron
        model : str
            model for wind rate: 'fire2', 'fire3'

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        '''
        # min and max imposed in FIRE-2 and FIRE-3 for stellar wind rates for stability
        metallicity_min = 0.01
        metallicity_max = 3

        self._parse_model(model)

        if metal_mass_fraction is not None:
            if 'fire2' in self.model:
                metallicity = metal_mass_fraction / self.sun_massfraction['metals']
            elif self.model == 'fire3':
                metallicity = metal_mass_fraction / self.sun_massfraction['iron']

        metallicity = np.clip(metallicity, metallicity_min, metallicity_max)

        if self.model == 'fire2.3':
            metallicity = 1  # force progenitor-metallicity-independent wind rates

        if 'fire2' in self.model:
            # FIRE-2: stellar_evolution.c line 350
            if np.isscalar(ages):
                assert ages >= 0 and ages < 14001
                # FIRE-2: stellar_evolution.c line 352
                if ages <= 1:
                    rates = 4.76317  # rate [Gyr^-1]
                    if self.model == 'fire2.1':
                        rates = 4.76317 * metallicity  # used (accidentally?) in some simulations
                elif ages <= 3.5:
                    rates = 4.76317 * metallicity * ages ** (1.838 * (0.79 + np.log10(metallicity)))
                elif ages <= 100:
                    rates = 29.4 * (ages / 3.5) ** -3.25 + 0.0041987
                else:
                    rates = 0.41987 * (ages / 1e3) ** -1.1 / (12.9 - np.log(ages / 1e3))
            else:
                assert np.min(ages) >= 0 and np.max(ages) < 14000
                ages = np.asarray(ages)
                rates = np.zeros(ages.size)

                masks = np.where(ages <= 1)[0]
                rates[masks] = 4.76317  # rate [Gyr^-1]
                if self.model == 'fire2.1':
                    rates[masks] = 4.76317 * metallicity  # used (accidentally?) in some simulations

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

        elif self.model == 'fire3':
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
            t1 = 1.7  # transition ages, converted to [Myr]
            t2 = 4
            t3 = 20
            # AGB: note that essentially no models [any of the SB99 geneva or padova tracks,
            # or NuGrid, or recent other MESA models] predict a significant dependence on
            # metallicity (that shifts slightly when the 'bump' occurs, but not the overall loss
            # rate), so this term is effectively metallicity-independent
            f_agb = 0.01
            t_agb = 1000  # converted to [Myr]

            if np.isscalar(ages):
                assert ages >= 0 and ages < 14001

                if ages <= t1:
                    rates = f1
                elif ages <= t2:
                    rates = f1 * (ages / t1) ** (np.log(f2 / f1) / np.log(t2 / t1))
                elif ages <= t3:
                    rates = f2 * (ages / t2) ** (np.log(f3 / f2) / np.log(t3 / t2))
                else:
                    rates = f3 * (ages / t3) ** -3.1

                # add AGB
                rates += f_agb / ((1 + (ages / t_agb) ** 1.1) * (1 + 0.01 / (ages / t_agb)))

            else:
                assert np.min(ages) >= 0 and np.max(ages) < 14001
                ages = np.asarray(ages)
                rates = np.zeros(ages.size)

                masks = np.where(ages <= t1)[0]
                rates[masks] = f1

                masks = np.where((ages > t1) * (ages <= t2))[0]
                rates[masks] = f1 * (ages / t1) ** (np.log(f2 / f1) / np.log(t2 / t1))

                masks = np.where((ages > t2) * (ages <= t3))[0]
                rates[masks] = f2 * (ages / t2) ** (np.log(f3 / f2) / np.log(t3 / t2))

                masks = np.where(ages > t3)[0]
                rates[masks] = f3 * (ages / t3) ** -3.1

                # add AGB
                rates += f_agb / ((1 + (ages / t_agb) ** 1.1) * (1 + 0.01 / (ages / t_agb)))

        rates *= 1e-3  # convert rates to [Myr ^ -1]
        # rates *= 1.4 * 0.291175  # old: expected return fraction from stellar winds alone (~17%)

        return rates

    def get_mass_loss_fraction(
        self,
        age_min=0,
        age_maxs=99,
        metallicity=1,
        metal_mass_fraction=None,
        model=None,
        element_name='',
    ):
        '''
        Get cumulative fractional mass loss [fractional wrt formation mass] via stellar winds
        within input age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        metallicity : float
            (linear) total abundance of metals wrt Solar
        metal_mass_fraction : float
            mass fraction of all metals (everything not H, He)
        model : str
            model for wind rate: 'fire2', 'fire3'
        element_name : str
            name of element to get yield of

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [fractional wrt formation mass]

        '''
        self._parse_model(model)

        # get transitional/discontinuous ages [Myr]  to be careful around when integrating
        ages_critical = get_ages_critical(model)

        if np.isscalar(age_maxs):
            mass_loss_fractions = integrate.quad(
                self.get_rate,
                age_min,
                age_maxs,
                (metallicity, metal_mass_fraction, self.model),
                points=ages_critical,
            )[0]
        else:
            mass_loss_fractions = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                mass_loss_fractions[age_i] = integrate.quad(
                    self.get_rate,
                    age_min,
                    age,
                    (metallicity, metal_mass_fraction, self.model),
                    points=ages_critical,
                )[0]

                # this method may be more stable for piece-wise (discontinuous) function
                # age_bin_width = 0.001  # [Myr]
                # ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                # mass_loss_fractions[age_i] = self.get_rate(
                #    ages, metallicity, metal_mass_fraction).sum() * age_bin_width

        if element_name:
            NucleosyntheticYield = NucleosyntheticYieldClass(model)
            element_yield = NucleosyntheticYield.get_yields('wind', metallicity, normalize=True)
            mass_loss_fractions *= element_yield[element_name]

        return mass_loss_fractions


StellarWind = StellarWindClass()


class SupernovaCCClass:
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for core-collapse supernovae
    in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=DEFAULT_MODEL, cc_age_min=None, cc_age_break=None, cc_age_max=None):
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
        self.ejecta_mass = 10.5  # ejecta mass per event, IMF-averaged [M_sun]

        self.models_available = ['fire2', 'fire2.1', 'fire2.2', 'fire2.3', 'fire3']
        self._parse_model(model)

        if cc_age_min is None:
            if 'fire2' in self.model:
                cc_age_min = 3.401  # [Myr]
            elif self.model == 'fire3':
                cc_age_min = 3.7  # [Myr]
        assert cc_age_min >= 0
        self.cc_age_min = cc_age_min

        if cc_age_break is None or not cc_age_break:
            if 'fire2' in self.model:
                cc_age_break = 10.37  # [Myr]
            elif self.model == 'fire3':
                cc_age_break = 7  # [Myr]
        assert cc_age_break >= 0
        self.cc_age_break = cc_age_break

        if cc_age_max is None or not cc_age_max:
            if 'fire2' in self.model:
                cc_age_max = 37.53  # [Myr]
            elif self.model == 'fire3':
                cc_age_max = 44  # [Myr]
        assert cc_age_max >= 0
        self.cc_age_max = cc_age_max

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                # reset solar abundances
                self.sun_massfraction = get_sun_massfraction(model)
            self.model = model

        assert self.model in self.models_available

    def get_rate(self, ages, model=None, cc_age_min=None, cc_age_break=None, cc_age_max=None):
        '''
        Get specific rate [Myr ^ -1 per M_sun of stars at formation] of core-collapse supernovae.

        FIRE-2 model:
        Rates are from Starburst99 energetics: get rate from overall energetics assuming each
        core-collapse supernova is 10^51 erg.
        Core-collapse supernovae occur from 3.4 to 37.53 Myr after formation:
            3.4 to 10.37 Myr: rate / M_sun = 5.408e-10 yr ^ -1
            10.37 to 37.53 Myr: rate / M_sun = 2.516e-10 yr ^ -1

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
            specific rate[s] [Myr ^ -1 per M_sun at formation]
        '''

        def get_rate_fire3(star_age, cc_age_min, cc_age_break, cc_age_max, kind):
            f1 = 3.9e-4
            f2 = 5.1e-4
            f3 = 1.8e-4
            if kind == 'early':
                return f1 * (star_age / cc_age_min) ** (
                    np.log(f2 / f1) / np.log(cc_age_break / cc_age_min)
                )
            elif kind == 'late':
                return f2 * (star_age / cc_age_break) ** (
                    np.log(f3 / f2) / np.log(cc_age_max / cc_age_break)
                )

        self._parse_model(model)

        if cc_age_min is None:
            cc_age_min = self.cc_age_min
        assert cc_age_min >= 0

        if cc_age_break is None:
            cc_age_break = self.cc_age_break
        assert cc_age_break >= 0

        if cc_age_max is None:
            cc_age_max = self.cc_age_max
        assert cc_age_max >= 0

        fire2_rate_early = 5.408e-4  # [Myr ^ -1]
        fire2_rate_late = 2.516e-4  # [Myr ^ -1]

        if np.isscalar(ages):
            assert ages >= 0 and ages < 14001
            if ages < cc_age_min or ages > cc_age_max:
                rates = 0
            elif ages <= cc_age_break:
                if 'fire2' in self.model:
                    rates = fire2_rate_early
                elif self.model == 'fire3':
                    rates = get_rate_fire3(ages, cc_age_min, cc_age_break, cc_age_max, 'early')
            elif ages > cc_age_break:
                if 'fire2' in self.model:
                    rates = fire2_rate_late
                elif self.model == 'fire3':
                    rates = get_rate_fire3(ages, cc_age_min, cc_age_break, cc_age_max, 'late')
        else:
            assert np.min(ages) >= 0 and np.max(ages) < 14001
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where((ages >= cc_age_min) * (ages <= cc_age_break))[0]
            if 'fire2' in self.model:
                rates[masks] = fire2_rate_early
            elif self.model == 'fire3':
                rates[masks] = get_rate_fire3(
                    ages[masks], cc_age_min, cc_age_break, cc_age_max, 'early'
                )

            masks = np.where((ages > cc_age_break) * (ages <= cc_age_max))[0]
            if 'fire2' in self.model:
                rates[masks] = fire2_rate_late
            elif self.model == 'fire3':
                rates[masks] = get_rate_fire3(
                    ages[masks], cc_age_min, cc_age_break, cc_age_max, 'late'
                )

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
        Get specific number [per M_sun of stars at formation] of core-collapse supernovae in
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
        age_bin_width = 0.01

        self._parse_model(model)

        if np.isscalar(age_maxs):
            # numbers = integrate.quad(self.get_rate, age_min, age_maxs)[0]
            # this method is more stable for piece-wise (discontinuous) function
            ages = np.arange(age_min, age_maxs + age_bin_width, age_bin_width)
            numbers = (
                self.get_rate(ages, self.model, cc_age_min, cc_age_break, cc_age_max).sum()
                * age_bin_width
            )
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                # numbers[age_i] = integrate.quad(
                # self.get_rate, age_min, age, (model, cc_age_min, cc_age_break, cc_age_max))[0]
                ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                numbers[age_i] = (
                    self.get_rate(ages, self.model, cc_age_min, cc_age_break, cc_age_max).sum()
                    * age_bin_width
                )

        return numbers

    def get_mass_loss_fraction(
        self,
        age_min=0,
        age_maxs=99,
        model=None,
        cc_age_min=None,
        cc_age_break=None,
        cc_age_max=None,
        element_name='',
        metallicity=1.0,
    ):
        '''
        Get fractional mass loss via supernova ejecta [per M_sun of stars at formation] in input
        age interval[s].

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
            name of element to get yield of. if None/empty, get mass loss from all elements
        metallicity : float
            metallicity (wrt Solar) of progenitor stars (for Nitrogen yield)

        Returns
        -------
        mass_loss_fractions : float
            fractional mass loss (ejecta mass[es] per M_sun of stars at formation)
        '''
        self._parse_model(model)

        mass_loss_fractions = self.ejecta_mass * self.get_number(
            age_min, age_maxs, self.model, cc_age_min, cc_age_break, cc_age_max
        )

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(model)
            element_yield = NucelosyntheticYield.get_yields(
                'supernova.cc', metallicity, normalize=True
            )
            mass_loss_fractions *= element_yield[element_name]

        return mass_loss_fractions


SupernovaCC = SupernovaCCClass()


class SupernovaIaClass(ut.io.SayClass):
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for supernovae Ia
    in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=DEFAULT_MODEL, ia_age_min=None):
        '''
        Parameters
        ----------
        model : str
            stellar evolution model: 'fire2', 'fire3'
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        '''
        self.ejecta_mass = 1.4  # ejecta mass per event, IMF-averaged [M_sun]

        self.models_available = [
            'fire2',
            'fire2.1',
            'fire2.2',
            'fire2.3',
            'fire3',
            'mannucci',
            'maoz',
        ]
        self._parse_model(model)

        if ia_age_min is None:
            if 'fire2' in self.model:
                ia_age_min = 37.53  # [Myr] ensure FIRE-2 default
                # self.say(f'input Ia model = {model}, so forcing Ia age min = {ia_age_min} Myr')
            elif self.model == 'fire3':
                ia_age_min = 44  # [Myr] ensure FIRE-3 default
                # self.say(f'input Ia model = {model}, so forcing Ia age min = {ia_age_min} Myr')
        assert ia_age_min >= 0
        self.ia_age_min = ia_age_min

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                # reset solar abundances
                self.sun_massfraction = get_sun_massfraction(model)
            self.model = model

        assert self.model in self.models_available

    def get_rate(self, ages, model=None, ia_age_min=None):
        '''
        Get specific rate [Myr ^ -1 per M_sun of stars at formation] of supernovae Ia.

        Default FIRE-2 rates are from Mannucci, Della Valle, & Panagia 2006,
        for a delayed population (constant rate) + prompt population (gaussian).
        Starting 37.53 Myr after formation:
            rate / M_sun = 5.3e-14 + 1.6e-11 * exp(-0.5 * ((star_age - 5e-5) / 1e-5) ** 2) yr ^ -1

        FIRE-3 power-law model from Maoz & Graur 2017,
        normalized to 1.6 events per 1000 Msun per Hubble time,
        assuming Ia start 44 Myr after formation:
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
        rate : float
            specific rate[s] of supernovae [Myr ^ -1 per M_sun of stars at formation]
        '''

        def get_rate(ages, model):
            if model in ['mannucci', 'fire2', 'fire2.1']:
                # Mannucci, Della Valle, & Panagia 2006
                rate = 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((ages - 50) / 10) ** 2)  # [Myr ^ -1]
            elif model == 'fire3':
                # this normalization is 2.67e-7 [Myr ^ -1]
                rate = 1.6e-3 * 7.94e-5 / ((ia_age_min / 100) ** -0.1 - 0.61) * (ages / 1e3) ** -1.1
            elif model == 'maoz':
                # Maoz & Graur 2017
                rate = 2.6e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1] compromise fit
                # fit to volumetric, Hubble-time-integrated Ia N/M = 1.3 +/- 0.1 per 1000 Msun
                # rate = 2.1e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1]
                # fit to field galaxies, Hubble-time-integrated Ia N/M = 1.6 +/- 0.1 per 1000 Msun
                # rate = 2.6e-7 * (ages / 1e3) ** -1.13  # [Myr ^ -1]
                # fit to galaxy clusters, Hubble-time-integrated Ia N/M = 5.4 +/- 0.1 per 1000 Msun
                # rate = 6.7e-7 * (ages / 1e3) ** -1.39  # [Myr ^ -1]

            return rate

        self._parse_model(model)

        if ia_age_min is None:
            ia_age_min = self.ia_age_min
        assert ia_age_min >= 0

        if np.isscalar(ages):
            if ages < ia_age_min:
                rates = 0
            else:
                rates = get_rate(ages, self.model)
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where(ages >= ia_age_min)[0]
            rates[masks] = get_rate(ages[masks], self.model)

        return rates

    def get_number(self, age_min=0, age_maxs=99, model=None, ia_age_min=None):
        '''
        Get specific number [per M_sun of stars at formation] of supernovae Ia in input age
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
            specific number[s] of supernovae [per M_sun of stars at formation]
        '''
        self._parse_model(model)

        if np.isscalar(age_maxs):
            numbers = integrate.quad(self.get_rate, age_min, age_maxs, (self.model, ia_age_min))[0]
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                numbers[age_i] = integrate.quad(
                    self.get_rate, age_min, age, (self.model, ia_age_min)
                )[0]

        return numbers

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, model=None, ia_age_min=None, element_name=''
    ):
        '''
        Get fractional mass loss via supernova Ia ejecta (ejecta mass per M_sun) in age interval[s].

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
            name of element to get yield of. if None/empty, get mass loss from all elements

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [per M_sun of stars at formation]
        '''
        self._parse_model(model)

        mass_loss_fractions = self.ejecta_mass * self.get_number(
            age_min, age_maxs, self.model, ia_age_min
        )

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(self.model)
            element_yield = NucelosyntheticYield.get_yields('supernova.ia', normalize=True)
            mass_loss_fractions *= element_yield[element_name]

        return mass_loss_fractions


SupernovaIa = SupernovaIaClass()


class MassLossClass(ut.io.SayClass):
    '''
    Compute mass loss from all channels (stellar winds, core-collapse and Ia supernovae) as
    implemented in the FIRE-2 or FIRE-3 model.
    '''

    def __init__(self, model=DEFAULT_MODEL):
        '''
        Parameters
        ----------
        model : str
            stellar evolution model to use: 'fire2', 'fire3'
        '''
        self.models_available = ['fire2', 'fire2.1', 'fire3']
        self._parse_model(model)

        self.SupernovaCC = SupernovaCCClass(self.model)
        self.SupernovaIa = SupernovaIaClass(self.model)
        self.StellarWind = StellarWindClass(self.model)
        self.Spline = None
        self.AgeBin = None
        self.MetalBin = None
        self.mass_loss_fractions = None

        from os.path import expanduser

        self.file_name = expanduser('~') + '/.gizmo_mass_loss_spline.pkl'

    def _parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if not hasattr(self, 'model'):
            self.model = None

        if model is not None:
            model = model.lower()
            if model != self.model:
                # reset solar abundances
                self.sun_massfraction = get_sun_massfraction(model)
            self.model = model

        assert self.model in self.models_available

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None, model=None):
        '''
        Get rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        from all stellar evolution channels in FIRE-2 or FIRE-3.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            (linear) total abundance of metals wrt Solar
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        '''
        self._parse_model(model)

        return (
            self.SupernovaCC.get_rate(ages, self.model) * self.SupernovaCC.ejecta_mass
            + self.SupernovaIa.get_rate(ages, self.model) * self.SupernovaIa.ejecta_mass
            + self.StellarWind.get_rate(ages, metallicity, metal_mass_fraction, self.model)
        )

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None, model=None
    ):
        '''
        Get fractional mass loss via all stellar evolution channels within age interval[s]
        via direct integration.

        Parameters
        ----------
        age_min : float
            min (starting) age of stellar population [Myr]
        age_maxs : float or array
            max (ending) age[s] of stellar population [Myr]
        metallicity : float
            (linear) total abundance of metals wrt Solar
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [wrt formation mass]
        '''
        self._parse_model(model)

        return (
            self.SupernovaCC.get_mass_loss_fraction(age_min, age_maxs, self.model)
            + self.SupernovaIa.get_mass_loss_fraction(age_min, age_maxs, self.model)
            + self.StellarWind.get_mass_loss_fraction(
                age_min, age_maxs, metallicity, metal_mass_fraction, self.model
            )
        )

    def get_mass_loss_fraction_from_spline(
        self, ages=[], metallicities=[], metal_mass_fractions=None, model=None
    ):
        '''
        Get fractional mass loss via all stellar evolution channels at ages and metallicities
        (or metal mass fractions) via 2-D (bivariate) spline.

        Parameters
        ----------
        ages : float or array
            age[s] of stellar population [Myr]
        metallicities : float or array
            (linear) total abundance[s] of metals wrt Solar
        metal_mass_fractions : float or array
            mass fration[s] of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [wrt formation mass]
        '''
        self._parse_model(model)

        if metal_mass_fractions is not None:
            # convert mass fraction to metallicity using Solar value assumed in FIRE
            metallicities = metal_mass_fractions / self.StellarWind.sun_massfraction['metals']

        assert np.isscalar(ages) or np.isscalar(metallicities) or len(ages) == len(metallicities)

        if self.Spline is None:
            self._make_mass_loss_fraction_spline(model=self.model)

        mass_loss_fractions = self.Spline.ev(ages, metallicities)

        if np.isscalar(ages) and np.isscalar(metallicities):
            mass_loss_fractions = np.asscalar(mass_loss_fractions)

        return mass_loss_fractions

    def _make_mass_loss_fraction_spline(
        self,
        age_limits=[1, 13800],
        age_bin_number=20,
        metallicity_limits=[0.01, 3],
        metallicity_bin_number=25,
        model=None,
        force_remake=False,
        save_spline=False,
    ):
        '''
        Create 2-D bivariate spline (in age and metallicity) for fractional mass loss
        [wrt formation mass] via all stellar evolution channels.

        Parameters
        ----------
        age_limits : list
            min and max limits of age of stellar population [Myr]
        age_bin_number : int
            number of age bins within age_limits
        metallicity_limits : list
            min and max limits of (linear) total abundance of all metals wrt Solar
        metallicity_bin_number : float
            number of metallicity bin
        model : str
            model for rates: 'fire2', 'fire3'
        force_remake : bool
            force a recalculation of the spline, even if file exists
        save_spline : bool
            save the spline to a pickle file for rapid loading in the future
        '''
        from os.path import isfile

        if not force_remake and isfile(self.file_name):
            self._load_mass_fraction_spline()
            return

        from scipy import interpolate

        self._parse_model(model)

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
            self.mass_loss_fractions[:, metallicity_i] = self.get_mass_loss_fraction(
                age_min, self.AgeBin.mins, metallicity, model=self.model
            )

        self.Spline = interpolate.RectBivariateSpline(
            self.AgeBin.mins, self.MetalBin.mins, self.mass_loss_fractions
        )

        if save_spline:
            self._save_mass_fraction_spline()

    def _save_mass_fraction_spline(self):
        import pickle

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.Spline, f)
        print(f'saved spline as {self.file_name}')

    def _load_mass_fraction_spline(self):
        import pickle

        with open(self.file_name, 'rb') as f:
            self.Spline = pickle.load(f)
            print(f'loaded spline from {self.file_name}')


MassLoss = MassLossClass()


def plot_supernova_v_age(
    age_limits=[1, 4000],
    age_bin_width=0.1,
    age_log_scale=True,
    axis_y_kind='rate',
    axis_y_limits=[None, None],
    axis_y_log_scale=True,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Plot specific rates or cumulative numbers of supernovae (core-collapse + Ia) [per M_sun]
    versus stellar age.

    Parameters
    ----------
    age_limits : list
        min and max limits of age of stellar population [Myr]
    age_bin_width : float
        width of stellar age bin [Myr]
    age_log_scale : bool
        whether to use logarithmic scaling for age bins
    axis_y_limits : str
        'rate' or 'number'
    axis_y_limits : list
        min and max limits to impose on y axis
    axis_y_log_scale : bool
        whether to use logarithmic scaling for y axis
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        where to write figure file
    figure_index : int
        index for matplotlib window
    '''
    assert axis_y_kind in ['rate', 'number']

    AgeBin = ut.binning.BinClass(age_limits, age_bin_width, include_max=True)

    if axis_y_kind == 'rate':
        cc_fire2 = SupernovaCC.get_rate(AgeBin.mins, model='fire2')
        cc_fire3 = SupernovaCC.get_rate(AgeBin.mins, model='fire3')
        ia_fire2 = SupernovaIa.get_rate(AgeBin.mins, model='fire2')
        ia_fire3 = SupernovaIa.get_rate(AgeBin.mins, model='fire3')
    elif axis_y_kind == 'number':
        cc_fire2 = SupernovaCC.get_number(min(age_limits), AgeBin.maxs, model='fire2')
        cc_fire3 = SupernovaCC.get_number(min(age_limits), AgeBin.maxs, model='fire3')
        ia_fire2 = SupernovaIa.get_number(min(age_limits), AgeBin.maxs, model='fire2')
        ia_fire3 = SupernovaIa.get_number(min(age_limits), AgeBin.maxs, model='fire3')
        axis_y_limits[0] = 4e-5

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot,
        age_log_scale,
        age_limits,
        None,
        axis_y_log_scale,
        axis_y_limits,
        [cc_fire2, cc_fire3, ia_fire2, ia_fire3],
    )

    subplot.set_xlabel('star age $\\left[ {\\rm Myr} \\right]$')
    if axis_y_kind == 'rate':
        subplot.set_ylabel('SN rate $\\left[ {\\rm Myr}^{-1} {\\rm M}_\odot^{-1} \\right]$')
    elif axis_y_kind == 'number':
        subplot.set_ylabel('SN number $\\left[ {\\rm M}_\odot^{-1} \\right]$')

    colors = ut.plot.get_colors(4, use_black=False)

    subplot.plot(AgeBin.mins, cc_fire2, color=colors[0], label='CC (FIRE-2)')
    subplot.plot(AgeBin.mins, cc_fire3, color=colors[1], label='CC (FIRE-3)')
    subplot.plot(AgeBin.mins, ia_fire2, color=colors[2], label='Ia (FIRE-2)')
    subplot.plot(AgeBin.mins, ia_fire3, color=colors[3], label='Ia (FIRE-3)')

    ut.plot.make_legends(subplot, 'best')

    if plot_file_name is True or plot_file_name == '':
        if axis_y_kind == 'rate':
            plot_file_name = 'supernova.rate_v_time'
        elif axis_y_kind == 'number':
            plot_file_name = 'supernova.number.cum_v_time'
    ut.plot.parse_output(plot_file_name, plot_directory)


def plot_mass_loss_v_age(
    age_limits=[1, 10000],
    age_bin_width=0.01,
    age_log_scale=True,
    mass_loss_kind='rate',
    mass_loss_limits=[None, None],
    mass_loss_log_scale=True,
    element_name=None,
    metallicity=1,
    metal_mass_fraction=None,
    model=DEFAULT_MODEL,
    plot_file_name=False,
    plot_directory='.',
    figure_index=1,
):
    '''
    Compute mass loss from all channels (core-collapse supernovae, supernovae Ia, stellar winds)
    versus stellar age.

    Parameters
    ----------
    age_limits : list
        min and max limits of age of stellar population [Myr]
    age_bin_width : float
        width of stellar age bin [Myr]
    age_log_scale : bool
        whether to use logarithmic scaling for age bins
    mass_loss_kind : str
        'rate' or 'cumulative'
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
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        index for matplotlib window
    '''
    mass_loss_kind = mass_loss_kind.lower()
    assert mass_loss_kind in ['rate', 'cumulative']

    AgeBin = ut.binning.BinClass(
        age_limits, age_bin_width, include_max=True, log_scale=age_log_scale
    )

    SupernovaCC = SupernovaCCClass(model)
    SupernovaIa = SupernovaIaClass(model)
    StellarWind = StellarWindClass(model)

    if mass_loss_kind == 'rate':
        supernova_cc = SupernovaCC.get_rate(AgeBin.mins) * SupernovaCC.ejecta_mass
        supernova_Ia = SupernovaIa.get_rate(AgeBin.mins) * SupernovaIa.ejecta_mass
        wind = StellarWind.get_rate(AgeBin.mins, metallicity, metal_mass_fraction)
    else:
        supernova_cc = SupernovaCC.get_mass_loss_fraction(
            0, AgeBin.mins, metallicity=metallicity, element_name=element_name
        )
        supernova_Ia = SupernovaIa.get_mass_loss_fraction(0, AgeBin.mins, element_name=element_name)
        wind = StellarWind.get_mass_loss_fraction(
            0, AgeBin.mins, metallicity, metal_mass_fraction, element_name=element_name
        )

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

    subplot.plot(AgeBin.mins, supernova_cc, color=colors[0], label='supernova cc')
    subplot.plot(AgeBin.mins, supernova_Ia, color=colors[1], label='supernova Ia')
    subplot.plot(AgeBin.mins, wind, color=colors[2], label='stellar winds')
    subplot.plot(AgeBin.mins, total, color='black', linestyle=':', label='total')

    ut.plot.make_legends(subplot, 'best')

    if plot_file_name is True or plot_file_name == '':
        mass_loss_kind = mass_loss_kind.replace('cumulative', 'cum')
        if element_name is not None and len(element_name) > 0:
            plot_file_name = f'{element_name}.yield.{mass_loss_kind}_v_time'
        else:
            plot_file_name = f'star.mass.loss.{mass_loss_kind}_v_time'
        plot_file_name += '_Z.{}'.format(
            ut.io.get_string_from_numbers(metallicity, digits=4, exponential=False, strip=True)
        )
    ut.plot.parse_output(plot_file_name, plot_directory)
