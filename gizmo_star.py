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
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Myr] (this is different than most other modules in the package, which default to Gyr)
'''

import collections
import numpy as np
from scipy import integrate

import utilities as ut

# default rate + yield model to assume throughout
DEFAULT_MODEL = 'fire2'


# --------------------------------------------------------------------------------------------------
# nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
class NucleosyntheticYieldClass:
    '''
    Nucleosynthetic yields in the FIRE-2 or FIRE-3 models.

    Metallicity dependence:
        for stellar wind: oxygen
        for core-collpase supernova: nitgrogen
        for Ia supernova: none
    '''

    def __init__(self, model=DEFAULT_MODEL):
        '''
        Store Solar elemental abundances, as linear mass fractions.

        FIRE-2 uses Anders & Grevesse 1989 for Solar.
        FIRE-3 uses Asplund et al 2009 for Solar.

        Parameters
        ----------
        model : str
            stellar evolution model for yields: 'fire2', 'fire3'
        '''
        self.models_possible = ['fire2', 'fire3']
        self.model = model.lower()
        assert self.model in self.models_possible

        self.sun_massfraction = {}
        if self.model == 'fire2':
            # FIRE-2 uses Anders & Grevesse 1989 for Solar
            self.sun_massfraction['metals'] = 0.02  # total metal mass fraction
            self.sun_massfraction['helium'] = 0.28
            self.sun_massfraction['carbon'] = 3.26e-3
            self.sun_massfraction['nitrogen'] = 1.32e-3
            self.sun_massfraction['oxygen'] = 8.65e-3
            self.sun_massfraction['neon'] = 2.22e-3
            self.sun_massfraction['magnesium'] = 9.31e-4
            self.sun_massfraction['silicon'] = 1.08e-3
            self.sun_massfraction['sulphur'] = 6.44e-4
            self.sun_massfraction['calcium'] = 1.01e-4
            self.sun_massfraction['iron'] = 1.73e-3
        elif self.model == 'fire3':
            raise ValueError(f'not yet support {model}')

        self.sncc_yield = None
        self.snia_yield = None
        self.wind_yield = None

    def get_yields(
        self,
        event_kind='supernova.cc',
        progenitor_metallicity=1.0,
        progenitor_massfraction_dict={},
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
            [*linear* mass fraction relative to sun_metal_mass_fraction]
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

        # determine progenitor abundance(s)
        if progenitor_massfraction_dict is not None and len(progenitor_massfraction_dict) > 0:
            # input mass fraction for each element, use to compute higher-order corrections
            for element_name in element_yield:
                assert element_name in progenitor_massfraction_dict
            progenitor_metal_mass_fraction = progenitor_massfraction_dict['metals']
        else:
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
            # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
            # treat AGB and O-star yields in more detail for light elements
            ejecta_mass = 1.0  # these yields are intrinsically mass fractions (wrt formation mass)

            element_yield['helium'] = 0.36
            element_yield['carbon'] = 0.016
            element_yield['nitrogen'] = 0.0041
            element_yield['oxygen'] = 0.0118

            # oxygen yield depends linearly on metallicity of progenitor star
            if progenitor_metal_mass_fraction < 0.033:
                element_yield['oxygen'] *= (
                    progenitor_metal_mass_fraction / self.sun_massfraction['metals']
                )
            else:
                element_yield['oxygen'] *= 1.65

            # sum total metal mass (not including hydrogen or helium)
            for k in element_yield:
                if k != 'helium':
                    element_yield['metals'] += element_yield[k]

        elif event_kind in ['supernova.cc' or 'supernova.ii']:
            # yields from Nomoto et al 2006, IMF averaged
            # rates from Starburst99
            # in FIRE-2, core-collapse occur 3.4 to 37.53 Myr after formation
            # from 3.4 to 10.37 Myr, rate / M_sun = 5.408e-10 yr ^ -1
            # from 10.37 to 37.53 Myr, rate / M_sun = 2.516e-10 yr ^ -1
            ejecta_mass = 10.5  # [M_sun]

            element_yield['metals'] = 2.0  # [M_sun]
            element_yield['helium'] = 3.87
            element_yield['carbon'] = 0.133
            element_yield['nitrogen'] = 0.0479
            element_yield['oxygen'] = 1.17
            element_yield['neon'] = 0.30
            element_yield['magnesium'] = 0.0987
            element_yield['silicon'] = 0.0933
            element_yield['sulphur'] = 0.0397
            element_yield['calcium'] = 0.00458  # Nomoto et al 2013 suggest 0.05 - 0.1 M_sun
            element_yield['iron'] = 0.0741

            yield_nitrogen_orig = np.float(element_yield['nitrogen'])

            # nitrogen yield depends linearly on metallicity of progenitor star
            if progenitor_metal_mass_fraction < 0.033:
                element_yield['nitrogen'] *= (
                    progenitor_metal_mass_fraction / self.sun_massfraction['metals']
                )
            else:
                element_yield['nitrogen'] *= 1.65
            # correct total metal mass for nitrogen
            element_yield['metals'] += element_yield['nitrogen'] - yield_nitrogen_orig

        elif event_kind == 'supernova.ia':
            # yields from Iwamoto et al 1999, W7 model, IMF averaged
            # rates from Mannucci, Della Valle & Panagia 2006
            # in Gizmo, these occur starting 37.53 Myr after formation, with rate / M_sun =
            # 5.3e-14 + 1.6e-11 * exp(-0.5 * ((age - 0.05) / 0.01) * ((age - 0.05) / 0.01)) yr^-1
            ejecta_mass = 1.4  # [M_sun]

            element_yield['metals'] = 1.4  # [M_sun]
            element_yield['helium'] = 0.0
            element_yield['carbon'] = 0.049
            element_yield['nitrogen'] = 1.2e-6
            element_yield['oxygen'] = 0.143
            element_yield['neon'] = 0.0045
            element_yield['magnesium'] = 0.0086
            element_yield['silicon'] = 0.156
            element_yield['sulphur'] = 0.087
            element_yield['calcium'] = 0.012
            element_yield['iron'] = 0.743

        if len(progenitor_massfraction_dict) > 0:
            # enforce that yields obey pre-existing surface abundances
            # allow for larger abundances in the progenitor star - usually irrelevant

            # get pure (non-metal) mass fraction of star
            pure_mass_fraction = 1 - progenitor_metal_mass_fraction

            for element_name in element_yield:
                if element_yield[element_name] > 0:
                    element_yield[element_name] /= ejecta_mass

                    # apply (new) yield only to pure (non-metal) mass of star
                    element_yield[element_name] *= pure_mass_fraction
                    # correction relative to solar abundance
                    element_yield[element_name] += (
                        progenitor_massfraction_dict[element_name]
                        - self.sun_massfraction[element_name]
                    )
                    element_yield[element_name] = np.clip(element_yield[element_name], 0, 1)

                    element_yield[element_name] *= ejecta_mass

        if normalize:
            # convert yield masses to mass fraction wrt total ejecta
            for element_name in element_yield:
                element_yield[element_name] /= ejecta_mass

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
            'wind', progenitor_metallicity, progenitor_massfraction_dict,
        )

        # store yields from supernova as masses [M_sun]
        self.sncc_yield = self.get_yields(
            'supernova.cc', progenitor_metallicity, progenitor_massfraction_dict, normalize=False,
        )

        self.snia_yield = self.get_yields(
            'supernova.ia', progenitor_metallicity, progenitor_massfraction_dict, normalize=False,
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

        self.models_available = ['fire2', 'fire3']
        self.model = model.lower()
        assert self.model in self.models_available

        # stellar wind rates depend on progenitor metallicity
        # FIRE-2 and FIRE-3 assume different Solar abundances
        if model == 'fire2':
            self.solar_metal_mass_fraction = 0.02  # FIRE-2 uses Anders & Grevesse 1989 for Solar
        elif model == 'fire3':
            self.solar_metal_mass_fraction = 0.0142  # FIRE-3 uses Asplund et al 2009 for Solar

    def parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            stellar evolution model: 'fire2', 'fire3'
        '''
        if model is None:
            model = self.model
        else:
            model = model.lower()
        assert model in self.models_available

        return model

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None, model=None):
        '''
        Get rate of fractional mass loss [Myr ^ -1, fractional wrt formation mass] from stellar
        winds in FIRE-2 or FIRE-3.
        Input either metallicity (linear, wrt self.solar_metal_mass_fraction) or
        (raw) metal_mass_fraction.

        Includes all non-supernova mass-loss channels, but dominated by O, B, and AGB stars.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            total abundance of metals wrt self.solar_metal_mass_fraction
            [for Solar, FIRE-2 assumes 0.02, FIRE-3 assumes 0.0142]
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        model : str
            model for wind rate: 'fire2', 'fire3'

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        '''
        metallicity_min = 0.01  # min and max imposed in FIRE-2 for stellar wind rates for stability
        metallicity_max = 3

        if metal_mass_fraction is not None:
            metallicity = metal_mass_fraction / self.solar_metal_mass_fraction

        metallicity = np.clip(metallicity, metallicity_min, metallicity_max)

        model = self.parse_model(model)

        if np.isscalar(ages):
            assert ages >= 0 and ages < 16000
            if model == 'fire2':
                if ages <= 1:
                    rates = 11.6846
                elif ages <= 3.5:
                    rates = 11.6846 * metallicity * ages ** (1.838 * (0.79 + np.log10(metallicity)))
                elif ages <= 100:
                    rates = 72.1215 * (ages / 3.5) ** -3.25 + 0.0103
                else:
                    rates = 1.03 * (ages / 1e3) ** -1.1 / (12.9 - np.log(ages / 1e3))
            elif model == 'fire3':
                raise ValueError(f'not yet support {model}')
        else:
            assert np.min(ages) >= 0 and np.max(ages) < 16000

            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            if model == 'fire2':
                masks = np.where(ages <= 1)[0]
                rates[masks] = 11.6846

                masks = np.where((ages > 1) * (ages <= 3.5))[0]
                rates[masks] = (
                    11.6846 * metallicity * ages[masks] ** (1.838 * (0.79 + np.log10(metallicity)))
                )

                masks = np.where((ages > 3.5) * (ages <= 100))[0]
                rates[masks] = 72.1215 * (ages[masks] / 3.5) ** -3.25 + 0.0103

                masks = np.where(ages > 100)[0]
                rates[masks] = (
                    1.03 * (ages[masks] / 1e3) ** -1.1 / (12.9 - np.log(ages[masks] / 1e3))
                )
            elif model == 'fire3':
                raise ValueError(f'not yet support {model}')

        rates *= 1e-3  # convert rate[s] to [Myr ^ -1]

        rates *= 1.4 * 0.291175  # give expected return fraction from stellar winds alone (~17%)

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
            total abundance of metals wrt solar_metal_mass_fraction
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
        model = self.parse_model(model)

        if np.isscalar(age_maxs):
            mass_loss_fractions = integrate.quad(
                self.get_rate, age_min, age_maxs, (metallicity, metal_mass_fraction, model)
            )[0]
        else:
            mass_loss_fractions = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                mass_loss_fractions[age_i] = integrate.quad(
                    self.get_rate, age_min, age, (metallicity, metal_mass_fraction, model)
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

        self.models_available = ['fire2', 'fire3']
        self.model = model.lower()
        assert self.model in self.models_available

        if cc_age_min is None:
            if model == 'fire2':
                cc_age_min = 3.4  # [Myr]
            elif model == 'fire3':
                cc_age_min = 3.7  # [Myr]
        assert cc_age_min >= 0
        self.cc_age_min = cc_age_min

        if cc_age_break is None or not cc_age_break:
            if model == 'fire2':
                cc_age_break = 10.37  # [Myr]
            elif model == 'fire3':
                cc_age_break = 7  # [Myr]
        assert cc_age_break >= 0
        self.cc_age_break = cc_age_break

        if cc_age_max is None or not cc_age_max:
            if model == 'fire2':
                cc_age_max = 37.53  # [Myr]
            elif model == 'fire3':
                cc_age_max = 44  # [Myr]
        assert cc_age_max >= 0
        self.cc_age_max = cc_age_max

    def parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            stellar evolution model: 'fire2', 'fire3'
        '''
        if model is None:
            model = self.model
        else:
            model = model.lower()
        assert model in self.models_available

        return model

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
                    np.log10(f2 / f1) / np.log10(cc_age_break / cc_age_min)
                )
            elif kind == 'late':
                return f2 * (star_age / cc_age_min) ** (
                    np.log10(f3 / f2) / np.log10(cc_age_max / cc_age_break)
                )

        model = self.parse_model(model)

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
            if ages < cc_age_min or ages > cc_age_max:
                rates = 0
            elif ages <= cc_age_break:
                if model == 'fire2':
                    rates = fire2_rate_early
                elif model == 'fire3':
                    rates = get_rate_fire3(ages, cc_age_min, cc_age_break, cc_age_max, 'early')
            elif ages > cc_age_break:
                if model == 'fire2':
                    rates = fire2_rate_late
                elif model == 'fire3':
                    rates = get_rate_fire3(ages, cc_age_min, cc_age_break, cc_age_max, 'late')
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where((ages >= cc_age_min) * (ages <= cc_age_break))[0]
            if model == 'fire2':
                rates[masks] = fire2_rate_early
            elif model == 'fire3':
                rates[masks] = get_rate_fire3(
                    ages[masks], cc_age_min, cc_age_break, cc_age_max, 'early'
                )

            masks = np.where((ages <= cc_age_max) * (ages > cc_age_break))[0]
            if model == 'fire2':
                rates[masks] = fire2_rate_late
            elif model == 'fire3':
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

        model = self.parse_model(model)

        if np.isscalar(age_maxs):
            # numbers = integrate.quad(self.get_rate, age_min, age_maxs)[0]
            # this method is more stable for piece-wise (discontinuous) function
            ages = np.arange(age_min, age_maxs + age_bin_width, age_bin_width)
            numbers = (
                self.get_rate(ages, model, cc_age_min, cc_age_break, cc_age_max).sum()
                * age_bin_width
            )
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                # numbers[age_i] = integrate.quad(
                # self.get_rate, age_min, age, (model, cc_age_min, cc_age_break, cc_age_max))[0]
                ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                numbers[age_i] = (
                    self.get_rate(ages, model, cc_age_min, cc_age_break, cc_age_max).sum()
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
        model = self.parse_model(model)

        mass_loss_fractions = self.ejecta_mass * self.get_number(
            age_min, age_maxs, model, cc_age_min, cc_age_break, cc_age_max
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

        self.models_available = ['fire2', 'fire3', 'mannucci', 'maoz']
        self.model = model.lower()
        assert self.model in self.models_available

        if ia_age_min is None:
            if model == 'fire2':
                ia_age_min = 37.53  # [Myr] ensure FIRE-2 default
                # self.say(f'input Ia model = {model}, so forcing Ia age min = {ia_age_min} Myr')
            elif model == 'fire3':
                ia_age_min = 44  # [Myr] ensure FIRE-3 default
                # self.say(f'input Ia model = {model}, so forcing Ia age min = {ia_age_min} Myr')
        assert ia_age_min >= 0
        self.ia_age_min = ia_age_min

    def parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            model for rate (delay time distribution):
                'fire2' or 'mannucci' (FIRE-2 default), 'fire3' (FIRE-3 default), 'maoz' (power law)
        '''
        if model is None:
            model = self.model
        else:
            model = model.lower()
        assert model in self.models_available

        return model

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
            if model in ['mannucci', 'fire2']:
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

        model = self.parse_model(model)

        if ia_age_min is None:
            ia_age_min = self.ia_age_min
        assert ia_age_min >= 0

        if np.isscalar(ages):
            if ages < ia_age_min:
                rates = 0
            else:
                rates = get_rate(ages, model)
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where(ages >= ia_age_min)[0]
            rates[masks] = get_rate(ages[masks], model)

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
        model = self.parse_model(model)

        if np.isscalar(age_maxs):
            numbers = integrate.quad(self.get_rate, age_min, age_maxs, (model, ia_age_min))[0]
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                numbers[age_i] = integrate.quad(self.get_rate, age_min, age, (model, ia_age_min))[0]

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
        model = self.parse_model(model)

        mass_loss_fractions = self.ejecta_mass * self.get_number(
            age_min, age_maxs, model, ia_age_min
        )

        if element_name:
            NucelosyntheticYield = NucleosyntheticYieldClass(model)
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
        self.models_available = ['fire2', 'fire3']
        self.model = model.lower()
        assert self.model in self.models_available

        self.SupernovaCC = SupernovaCCClass(model)
        self.SupernovaIa = SupernovaIaClass(model)
        self.StellarWind = StellarWindClass(model)
        self.Spline = None
        self.AgeBin = None
        self.MetalBin = None
        self.mass_loss_fractions = None

        from os.path import expanduser

        self.file_name = expanduser('~') + '/.gizmo_mass_loss_spline.pkl'

    def parse_model(self, model):
        '''
        Parse input model.

        Parameters
        ----------
        model : str
            stellar evolution model: 'fire2', 'fire3'
        '''
        if model is None:
            model = self.model
        else:
            model = model.lower()
        assert model in self.models_available

        return model

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None, model=None):
        '''
        Get rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        from all stellar evolution channels in FIRE-2 or FIRE-3.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss [Myr ^ -1, fractional wrt formation mass]
        '''
        model = self.parse_model(model)

        return (
            self.SupernovaCC.get_rate(ages, model) * self.SupernovaCC.ejecta_mass
            + self.SupernovaIa.get_rate(ages, model) * self.SupernovaIa.ejecta_mass
            + self.StellarWind.get_rate(ages, metallicity, metal_mass_fraction, model)
        )

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None, model=None,
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
            total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [wrt formation mass]
        '''
        model = self.parse_model(model)

        return (
            self.SupernovaCC.get_mass_loss_fraction(age_min, age_maxs, model)
            + self.SupernovaIa.get_mass_loss_fraction(age_min, age_maxs, model)
            + self.StellarWind.get_mass_loss_fraction(
                age_min, age_maxs, metallicity, metal_mass_fraction, model
            )
        )

    def get_mass_loss_fraction_from_spline(
        self, ages=[], metallicities=[], metal_mass_fractions=None, model=None,
    ):
        '''
        Get fractional mass loss via all stellar evolution channels at ages and metallicities
        (or metal mass fractions) via 2-D (bivariate) spline.

        Parameters
        ----------
        ages : float or array
            age[s] of stellar population [Myr]
        metallicities : float or array
            total abundance[s] of metals wrt solar_metal_mass_fraction
        metal_mass_fractions : float or array
            mass fration[s] of all metals (everything not H, He)
        model : str
            model for rates: 'fire2', 'fire3'

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] [wrt formation mass]
        '''
        model = self.parse_model(model)

        if metal_mass_fractions is not None:
            # convert mass fraction to metallicity using Solar value assumed in FIRE
            metallicities = metal_mass_fractions / self.StellarWind.solar_metal_mass_fraction

        assert np.isscalar(ages) or np.isscalar(metallicities) or len(ages) == len(metallicities)

        if self.Spline is None:
            self._make_mass_loss_fraction_spline(model=model)

        mass_loss_fractions = self.Spline.ev(ages, metallicities)

        if np.isscalar(ages) and np.isscalar(metallicities):
            mass_loss_fractions = np.asscalar(mass_loss_fractions)

        return mass_loss_fractions

    def _make_mass_loss_fraction_spline(
        self,
        age_limits=[1, 14000],
        age_bin_width=0.2,
        metallicity_limits=[0.01, 3],
        metallicity_bin_width=0.1,
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
        age_bin_width : float
            log width of age bin [Myr]
        metallicity_limits : list
            min and max limits of metal abundance wrt solar_metal_mass_fraction
        metallicity_bin_width : float
            width of metallicity bin
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

        model = self.parse_model(model)

        age_min = 0

        self.AgeBin = ut.binning.BinClass(
            age_limits, age_bin_width, include_max=True, log_scale=True,
        )
        self.MetalBin = ut.binning.BinClass(
            metallicity_limits, metallicity_bin_width, include_max=True, log_scale=True,
        )

        self.say('* generating 2-D spline to compute stellar mass loss from age + metallicity')
        self.say(f'number of age bins = {self.AgeBin.number}')
        self.say(f'number of metallicity bins = {self.MetalBin.number}')

        self.mass_loss_fractions = np.zeros((self.AgeBin.number, self.MetalBin.number))
        for metallicity_i, metallicity in enumerate(self.MetalBin.mins):
            self.mass_loss_fractions[:, metallicity_i] = self.get_mass_loss_fraction(
                age_min, self.AgeBin.mins, metallicity, model=model
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
        total abundance of metals wrt solar_metal_mass_fraction
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
    # StellarWind = StellarWindClass(model)

    if mass_loss_kind == 'rate':
        supernova_cc = SupernovaCC.get_rate(AgeBin.mins) * SupernovaCC.ejecta_mass
        supernova_Ia = SupernovaIa.get_rate(AgeBin.mins) * SupernovaIa.ejecta_mass
        wind = StellarWind.get_rate(AgeBin.mins, metallicity, metal_mass_fraction)
    else:
        supernova_cc = SupernovaCC.get_mass_loss_fraction(
            0, AgeBin.mins, metallicity=metallicity, element_name=element_name,
        )
        supernova_Ia = SupernovaIa.get_mass_loss_fraction(0, AgeBin.mins, element_name=element_name)
        # wind = StellarWind.get_mass_loss_fraction(
        #    0, AgeBin.mins, metallicity, metal_mass_fraction, element_name=element_name
        # )
    wind = np.zeros(supernova_cc.size)

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
    ut.plot.parse_output(plot_file_name, plot_directory)
