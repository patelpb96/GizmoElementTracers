'''
Analyze stellar evolution, including supernova rates, stellar mass loss, nucleosynthetic yields,
as implemented in Gizmo simulations.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Myr]
'''

import collections
import numpy as np
from scipy import integrate

import utilities as ut


# elemental abundances (mass fraction) of the Sun that Gizmo uses for FIRE-2
sun_massfraction = {}
sun_massfraction['metals'] = 0.02  # total metal mass fraction
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


# --------------------------------------------------------------------------------------------------
# nucleosynthetic yields
# --------------------------------------------------------------------------------------------------
def get_nucleosynthetic_yields(
    event_kind='supernova.cc', star_metallicity=1.0, star_massfraction={}, normalize=True
):
    '''
    Get nucleosynthetic element yields, according to input event_kind.
    This computes the *additional* nucleosynthetic yields that Gizmo adds to the star's existing
    metallicity, so these are not the actual yields that get deposited to gas.
    The total mass fraction (wrt total mass of star) returned for each element is:
        (1 - ejecta_total_mass_fraction) * star_element_mass_fraction
            + ejecta_total_mass_fraction * ejecta_element_mass_fraction

    Parameters
    ----------
    event_kind : str
        stellar event: 'wind', 'supernova.cc' or 'supernova.ii', 'supernova.ia'
    star_metallicity : float
        total metallicity of star prior to event, specifically, *linear* mass fraction relative to
        solar (solar := sun_metal_mass_fraction)
    star_massfraction : dict
        optional: dictionary of elemental mass fractions of stars
        use to get higher-order correction of yields that Gizmo includes
    normalize : bool
        whether to normalize yields to be mass fractions (instead of masses)

    Returns
    -------
    star_yield : ordered dict
        stellar yield for each element, in mass [M_sun] or mass fraction
    '''
    star_yield = collections.OrderedDict()
    star_yield['metals'] = 0.0
    star_yield['helium'] = 0.0
    star_yield['carbon'] = 0.0
    star_yield['nitrogen'] = 0.0
    star_yield['oxygen'] = 0.0
    star_yield['neon'] = 0.0
    star_yield['magnesium'] = 0.0
    star_yield['silicon'] = 0.0
    star_yield['sulphur'] = 0.0
    star_yield['calcium'] = 0.0
    star_yield['iron'] = 0.0

    event_kind = event_kind.lower()

    assert event_kind in ['wind', 'supernova.cc', 'supernova.ii', 'supernova.ia']

    if star_massfraction is not None and len(star_massfraction) > 0:
        # input full array of stellar elemental mass fractions
        for element_name in star_yield:
            assert element_name in star_massfraction
        star_metal_mass_fraction = star_massfraction['metals']
    else:
        star_metal_mass_fraction = star_metallicity * sun_massfraction['metals']

    if event_kind == 'wind':
        # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
        # treat AGB and O-star yields in more detail for light elements
        ejecta_mass = 1.0  # these yields already are mass fractions

        star_yield['helium'] = 0.36
        star_yield['carbon'] = 0.016
        star_yield['nitrogen'] = 0.0041
        star_yield['oxygen'] = 0.0118

        # oxygen yield depends linearly on metallicity of progenitor star
        if star_metal_mass_fraction < 0.033:
            star_yield['oxygen'] *= star_metal_mass_fraction / sun_massfraction['metals']
        else:
            star_yield['oxygen'] *= 1.65

        # sum total metal mass (not including hydrogen or helium)
        for k in star_yield:
            if k != 'helium':
                star_yield['metals'] += star_yield[k]

    elif event_kind in ['supernova.cc' or 'supernova.ii']:
        # yields from Nomoto et al 2006, IMF averaged
        # rates from Starburst99
        # in Gizmo core-collapse occur 3.4 to 37.53 Myr after formation
        # from 3.4 to 10.37 Myr, rate / M_sun = 5.408e-10 yr ^ -1
        # from 10.37 to 37.53 Myr, rate / M_sun = 2.516e-10 yr ^ -1
        ejecta_mass = 10.5  # [M_sun]

        star_yield['metals'] = 2.0  # [M_sun]
        star_yield['helium'] = 3.87
        star_yield['carbon'] = 0.133
        star_yield['nitrogen'] = 0.0479
        star_yield['oxygen'] = 1.17
        star_yield['neon'] = 0.30
        star_yield['magnesium'] = 0.0987
        star_yield['silicon'] = 0.0933
        star_yield['sulphur'] = 0.0397
        star_yield['calcium'] = 0.00458  # Nomoto et al 2013 suggest 0.05 - 0.1 M_sun
        star_yield['iron'] = 0.0741

        yield_nitrogen_orig = np.float(star_yield['nitrogen'])

        # nitrogen yield depends linearly on metallicity of progenitor star
        if star_metal_mass_fraction < 0.033:
            star_yield['nitrogen'] *= star_metal_mass_fraction / sun_massfraction['metals']
        else:
            star_yield['nitrogen'] *= 1.65
        # correct total metal mass for nitrogen
        star_yield['metals'] += star_yield['nitrogen'] - yield_nitrogen_orig

    elif event_kind == 'supernova.ia':
        # yields from Iwamoto et al 1999, W7 model, IMF averaged
        # rates from Mannucci, Della Valle & Panagia 2006
        # in Gizmo, these occur starting 37.53 Myr after formation, with rate / M_sun =
        # 5.3e-14 + 1.6e-11 * exp(-0.5 * ((age - 0.05) / 0.01) * ((age - 0.05) / 0.01)) yr^-1
        ejecta_mass = 1.4  # [M_sun]

        star_yield['metals'] = 1.4  # [M_sun]
        star_yield['helium'] = 0.0
        star_yield['carbon'] = 0.049
        star_yield['nitrogen'] = 1.2e-6
        star_yield['oxygen'] = 0.143
        star_yield['neon'] = 0.0045
        star_yield['magnesium'] = 0.0086
        star_yield['silicon'] = 0.156
        star_yield['sulphur'] = 0.087
        star_yield['calcium'] = 0.012
        star_yield['iron'] = 0.743

    if len(star_massfraction) > 0:
        # enforce that yields obey pre-existing surface abundances
        # allow for larger abundances in the progenitor star - usually irrelevant

        # get pure (non-metal) mass fraction of star
        pure_mass_fraction = 1 - star_metal_mass_fraction

        for element_name in star_yield:
            if star_yield[element_name] > 0:
                star_yield[element_name] /= ejecta_mass

                # apply (new) yield only to pure (non-metal) mass of star
                star_yield[element_name] *= pure_mass_fraction
                # correction relative to solar abundance
                star_yield[element_name] += (
                    star_massfraction[element_name] - sun_massfraction[element_name]
                )
                star_yield[element_name] = np.clip(star_yield[element_name], 0, 1)

                star_yield[element_name] *= ejecta_mass

    if normalize:
        # convert yield masses to mass fraction wrt total ejecta
        for element_name in star_yield:
            star_yield[element_name] /= ejecta_mass

    return star_yield


def plot_nucleosynthetic_yields(
    event_kind='wind',
    star_metallicity=0.1,
    star_massfraction={},
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

    event_kind = event_kind.lower()

    star_yield = get_nucleosynthetic_yields(
        event_kind, star_metallicity, star_massfraction, normalize=normalize
    )

    yield_indices = np.arange(1, len(star_yield))
    yield_names = np.array([k for k in star_yield])[yield_indices]
    yield_values = np.array([star_yield[k] for k in star_yield])[yield_indices]
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
class SupernovaCCClass:
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for core-collapse supernovae,
    as implemented in Gizmo.
    '''

    def __init__(self):
        self.ejecta_mass = 10.5  # ejecta mass per event, IMF-averaged [M_sun]

    def get_rate(self, ages):
        '''
        Get specific rate [Myr ^ -1 M_sun ^ -1] of core-collapse supernovae.

        Rates are from Starburst99 energetics: assume each core-collapse is 10^51 erg, derive rate.
        Core-collapse supernovae occur from 3.4 to 37.53 Myr after formation:
            3.4 to 10.37 Myr: rate / M_sun = 5.408e-10 yr ^ -1
            10.37 to 37.53 Myr: rate / M_sun = 2.516e-10 yr ^ -1

        Parameters
        ----------
        ages : float or array
            age[s] of stellar population [Myr]

        Returns
        -------
        rates : float or array
            specific rate[s] [Myr ^ -1 M_sun ^ -1]
        '''
        star_age_min = 3.4  # [Myr]
        star_age_transition = 10.37  # [Myr]
        star_age_max = 37.53  # [Myr]

        rate_early = 5.408e-4  # [Myr ^ -1]
        rate_late = 2.516e-4  # [Myr ^ -1]

        if np.isscalar(ages):
            if ages < star_age_min or ages > star_age_max:
                rates = 0
            elif ages <= star_age_transition:
                rates = rate_early
            elif ages > star_age_transition:
                rates = rate_late
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where((ages >= star_age_min) * (ages <= star_age_transition))[0]
            rates[masks] = rate_early
            masks = np.where((ages <= star_age_max) * (ages > star_age_transition))[0]
            rates[masks] = rate_late

        return rates

    def get_number(self, age_min=0, age_maxs=99):
        '''
        Get specific number [per M_sun] of supernovae in input age interval.

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]

        Returns
        -------
        numbers : float or array
            specific number[s] of supernova events [per M_sun]
        '''
        age_bin_width = 0.01

        if np.isscalar(age_maxs):
            # numbers = integrate.quad(self.get_rate, age_min, age_maxs)[0]
            # this method is more stable for piece-wise (discontinuous) function
            ages = np.arange(age_min, age_maxs + age_bin_width, age_bin_width)
            numbers = self.get_rate(ages).sum() * age_bin_width
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                # numbers[age_i] = integrate.quad(self.get_rate, age_min, age)[0]
                ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                numbers[age_i] = self.get_rate(ages).sum() * age_bin_width

        return numbers

    def get_mass_loss_fraction(self, age_min=0, age_maxs=99, element_name='', metallicity=1.0):
        '''
        Get fractional mass loss via supernova ejecta (ejecta mass per M_sun) in age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        element_name : bool
            name of element to get yield of
        metallicity : float
            metallicity of star (for Nitrogen yield)

        Returns
        -------
        mass_loss_fractions : float
            fractional mass loss (ejecta mass[es] per M_sun)
        '''
        mass_loss_fractions = self.ejecta_mass * self.get_number(age_min, age_maxs)

        if element_name:
            element_yields = get_nucleosynthetic_yields('supernova.cc', metallicity, normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


SupernovaCC = SupernovaCCClass()


class SupernovaIaClass:
    '''
    Compute rates, cumulative numbers, and cumulative ejecta masses for supernovae Ia,
    as implemented in Gizmo.
    '''

    def __init__(self):
        self.ejecta_mass = 1.4  # ejecta mass per event, IMF-averaged [M_sun]

    def get_rate(self, ages, ia_kind='mannucci', ia_age_min=37.53):
        '''
        Get specific rate [Myr ^ -1 M_sun ^ -1] of supernovae Ia.

        Default rates are from Mannucci, Della Valle, & Panagia 2006,
        for a delayed population (constant rate) + prompt population (gaussian).
        Starting 37.53 Myr after formation:
            rate / M_sun = 5.3e-14 + 1.6e-11 * exp(-0.5 * ((star_age - 5e-5) / 1e-5) ** 2) yr ^ -1

        Updated power-law model (to update Gizmo to eventually) from Maoz & Graur 2017,
        normalized assuming Ia start 40 Myr after formation:
            rate / M_sun = 2e-13 * (star_age / 1e6) ** (-1.1) yr ^ -1

        Parameters
        ----------
        ages : float
            age of stellar population [Myr]
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
            decreasing to 10 Myr increases total number by ~50%,
            increasing to 100 Myr decreases total number by ~50%

        Returns
        -------
        rate : float
            specific rate of supernovae [Myr ^ -1 M_sun ^ -1]
        '''

        def get_rate(ages, kind):
            if kind == 'mannucci':
                # Mannucci, Della Valle, & Panagia 2006
                rate = 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((ages - 50) / 10) ** 2)  # [Myr ^ -1]
            elif kind == 'maoz':
                # Maoz & Graur 2017
                rate = 2.6e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1], my compromise fit
                # fit to volumetric, Hubble-time-integrated Ia N/M = 1.3 +/- 0.1 per 1000 Msun
                # rate = 2.1e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1]
                # fit to field galaxies, Hubble-time-integrated Ia N/M = 1.6 +/- 0.1 per 1000 Msun
                # rate = 2.6e-7 * (ages / 1e3) ** -1.13  # [Myr ^ -1]
                # fit to galaxy clusters, Hubble-time-integrated Ia N/M = 5.4 +/- 0.1 per 1000 Msun
                # rate = 6.7e-7 * (ages / 1e3) ** -1.39  # [Myr ^ -1]
            return rate

        assert ia_kind in ['mannucci', 'maoz']

        if np.isscalar(ages):
            if ages < ia_age_min:
                rates = 0
            else:
                rates = get_rate(ages, ia_kind)
        else:
            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            masks = np.where(ages >= ia_age_min)[0]
            rates[masks] = get_rate(ages[masks], ia_kind)

        return rates

    def get_number(self, age_min=0, age_maxs=99, ia_kind='mannucci', ia_age_min=37.53):
        '''
        Get specific number [per M_sun] of supernovae Ia in given age interval.

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        numbers : float or array
            specific number[s] of supernovae [M_sun ^ -1]
        '''
        if np.isscalar(age_maxs):
            numbers = integrate.quad(self.get_rate, age_min, age_maxs, (ia_kind, ia_age_min))[0]
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                numbers[age_i] = integrate.quad(self.get_rate, age_min, age, (ia_kind, ia_age_min))[
                    0
                ]

        return numbers

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, ia_kind='mannucci', ia_age_min=37.53, element_name=''
    ):
        '''
        Get fractional mass loss via supernova ejecta (ejecta mass per M_sun) in age interval[s].

        Parameters
        ----------
        age_min : float
            min age of stellar population [Myr]
        age_maxs : float or array
            max age[s] of stellar population [Myr]
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        element_name : str
            name of element to get yield of

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s] (ejecta mass per M_sun)
        '''
        mass_loss_fractions = self.ejecta_mass * self.get_number(
            age_min, age_maxs, ia_kind, ia_age_min
        )

        if element_name:
            element_yields = get_nucleosynthetic_yields('supernova.ia', normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


SupernovaIa = SupernovaIaClass()


class StellarWindClass:
    '''
    Compute mass loss rates rates and cumulative mass loss fractions for stellar winds,
    as implemented in Gizmo for FIRE-2.
    '''

    def __init__(self):
        self.ejecta_mass = 1.0  # for stellar winds, the values below are mass fractions
        self.solar_metal_mass_fraction = 0.02  # Gizmo assumes this

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None):
        '''
        Get rate of fractional mass loss [Myr ^ -1] from stellar winds.

        Includes all non-supernova mass-loss channels, but dominated by O, B, and AGB stars.

        Note: Gizmo assumes solar abundance (total metal mass fraction) of 0.02,
        while Asplund et al 2009 is 0.0134.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)

        Returns
        -------
        rates : float or array
            rate[s] of fractional mass loss [Myr ^ -1]
        '''
        metallicity_min = 0.01  # min and max imposed in Gizmo for stellar wind rates for stability
        metallicity_max = 3

        if metal_mass_fraction is not None:
            metallicity = metal_mass_fraction / self.solar_metal_mass_fraction

        metallicity = np.clip(metallicity, metallicity_min, metallicity_max)

        if np.isscalar(ages):
            assert ages >= 0 and ages < 16000
            # get rate
            if ages <= 1:
                rates = 11.6846
            elif ages <= 3.5:
                rates = 11.6846 * metallicity * ages ** (1.838 * (0.79 + np.log10(metallicity)))
            elif ages <= 100:
                rates = 72.1215 * (ages / 3.5) ** -3.25 + 0.0103
            else:
                rates = 1.03 * (ages / 1e3) ** -1.1 / (12.9 - np.log(ages / 1e3))
        else:
            assert np.min(ages) >= 0 and np.max(ages) < 16000

            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            # get rate
            masks = np.where(ages <= 1)[0]
            rates[masks] = 11.6846

            masks = np.where((ages > 1) * (ages <= 3.5))[0]
            rates[masks] = (
                11.6846 * metallicity * ages[masks] ** (1.838 * (0.79 + np.log10(metallicity)))
            )

            masks = np.where((ages > 3.5) * (ages <= 100))[0]
            rates[masks] = 72.1215 * (ages[masks] / 3.5) ** -3.25 + 0.0103

            masks = np.where(ages > 100)[0]
            rates[masks] = 1.03 * (ages[masks] / 1e3) ** -1.1 / (12.9 - np.log(ages[masks] / 1e3))

        rates *= 1e-3  # convert to [Myr ^ -1]

        rates *= 1.4 * 0.291175  # give expected return fraction from stellar winds alone (~17%)

        return rates

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None, element_name=''
    ):
        '''
        Get cumulative fractional mass loss via stellar winds within age interval[s].

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
        element_name : str
            name of element to get yield of

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s]
        '''
        if np.isscalar(age_maxs):
            mass_loss_fractions = integrate.quad(
                self.get_rate, age_min, age_maxs, (metallicity, metal_mass_fraction)
            )[0]
        else:
            mass_loss_fractions = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                mass_loss_fractions[age_i] = integrate.quad(
                    self.get_rate, age_min, age, (metallicity, metal_mass_fraction)
                )[0]

                # this method may be more stable for piece-wise (discontinuous) function
                # age_bin_width = 0.001  # [Myr]
                # ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                # mass_loss_fractions[age_i] = self.get_rate(
                #    ages, metallicity, metal_mass_fraction).sum() * age_bin_width

        if element_name:
            element_yields = get_nucleosynthetic_yields('wind', metallicity, normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


StellarWind = StellarWindClass()


class MassLossClass(ut.io.SayClass):
    '''
    Compute mass loss from all channels (stellar winds, core-collapse and Ia supernovae) as
    implemented in Gizmo for FIRE-2.
    '''

    def __init__(self):
        self.SupernovaCC = SupernovaCCClass()
        self.SupernovaIa = SupernovaIaClass()
        self.StellarWind = StellarWindClass()
        self.Spline = None
        self.AgeBin = None
        self.MetalBin = None
        self.mass_loss_fractions = None

        from os.path import expanduser

        self.filename = expanduser('~') + '/.gizmo_mass_loss_spline.pkl'

    def get_rate(
        self, ages, metallicity=1, metal_mass_fraction=None, ia_kind='mannucci', ia_age_min=37.53
    ):
        '''
        Get rate of fractional mass loss [Myr ^ -1] from all stellar evolution channels.

        Parameters
        ----------
        age : float or array
            age[s] of stellar population [Myr]
        metallicity : float
            total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float
            mass fration of all metals (everything not H, He)
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        rates : float or array
            fractional mass loss rate[s] [Myr ^ -1]
        '''
        return (
            self.SupernovaCC.get_rate(ages) * self.SupernovaCC.ejecta_mass
            + +self.SupernovaIa.get_rate(ages, ia_kind, ia_age_min) * self.SupernovaIa.ejecta_mass
            + self.StellarWind.get_rate(ages, metallicity, metal_mass_fraction)
        )

    def get_mass_loss_fraction(
        self,
        age_min=0,
        age_maxs=99,
        metallicity=1,
        metal_mass_fraction=None,
        ia_kind='mannucci',
        ia_age_min=37.53,
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
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s]
        '''
        return (
            self.SupernovaCC.get_mass_loss_fraction(age_min, age_maxs)
            + self.SupernovaIa.get_mass_loss_fraction(age_min, age_maxs, ia_kind, ia_age_min)
            + self.StellarWind.get_mass_loss_fraction(
                age_min, age_maxs, metallicity, metal_mass_fraction
            )
        )

    def get_mass_loss_fraction_from_spline(
        self, ages=[], metallicities=[], metal_mass_fractions=None
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

        Returns
        -------
        mass_loss_fractions : float or array
            mass loss fraction[s]
        '''
        if metal_mass_fractions is not None:
            # convert mass fraction to metallicity using Solar value assumed in Gizmo
            metallicities = metal_mass_fractions / self.StellarWind.solar_metal_mass_fraction

        assert np.isscalar(ages) or np.isscalar(metallicities) or len(ages) == len(metallicities)

        if self.Spline is None:
            self._make_mass_loss_fraction_spline()

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
        ia_kind='mannucci',
        ia_age_min=37.53,
        force_remake=False,
        save_spline=False,
    ):
        '''
        Create 2-D bivariate spline (in age and metallicity) for fractional mass loss via
        all stellar evolution channels.

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
        ia_kind : str
            supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float
            minimum age for supernova Ia to occur [Myr]
        force_remake : bool
            force a recalculation of the spline, even if file exists
        save_spline : bool
            save the spline to a pickle file for rapid loading in the future
        '''
        from os.path import isfile

        if not force_remake and isfile(self.filename):
            self._load_mass_fraction_spline()
            return

        from scipy import interpolate

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
                age_min, self.AgeBin.mins, metallicity, None, ia_kind, ia_age_min
            )

        self.Spline = interpolate.RectBivariateSpline(
            self.AgeBin.mins, self.MetalBin.mins, self.mass_loss_fractions
        )

        if save_spline:
            self._save_mass_fraction_spline()

    def _save_mass_fraction_spline(self):
        import pickle

        with open(self.filename, 'wb') as f:
            pickle.dump(self.Spline, f)
        print(f'saved spline as {self.filename}')

    def _load_mass_fraction_spline(self):
        import pickle

        with open(self.filename, 'rb') as f:
            self.Spline = pickle.load(f)
            print(f'loaded spline from {self.filename}')


MassLoss = MassLossClass()


def plot_supernova_v_age(
    age_limits=[1, 3000],
    age_bin_width=1,
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
        supernova_CC_rates = SupernovaCC.get_rate(AgeBin.mins)
        supernova_Ia_rates_mannucci = SupernovaIa.get_rate(AgeBin.mins, 'mannucci')
        supernova_Ia_rates_maoz = SupernovaIa.get_rate(AgeBin.mins, 'maoz')
    elif axis_y_kind == 'number':
        supernova_CC_rates = SupernovaCC.get_number(min(age_limits), AgeBin.maxs)
        supernova_Ia_rates_mannucci = SupernovaIa.get_number(
            min(age_limits), AgeBin.maxs, 'mannucci'
        )
        supernova_Ia_rates_maoz = SupernovaIa.get_number(min(age_limits), AgeBin.maxs, 'maoz')

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot,
        age_log_scale,
        age_limits,
        None,
        axis_y_log_scale,
        axis_y_limits,
        [supernova_CC_rates, supernova_Ia_rates_mannucci, supernova_Ia_rates_maoz],
    )

    subplot.set_xlabel('star age $\\left[ {\\rm Myr} \\right]$')
    if axis_y_kind == 'rate':
        subplot.set_ylabel('SN rate $\\left[ {\\rm Myr}^{-1} {\\rm M}_\odot^{-1} \\right]$')
    elif axis_y_kind == 'number':
        subplot.set_ylabel('SN number $\\left[ {\\rm M}_\odot^{-1} \\right]$')

    colors = ut.plot.get_colors(3, use_black=False)

    subplot.plot(AgeBin.mins, supernova_CC_rates, color=colors[0], label='CC')
    subplot.plot(AgeBin.mins, supernova_Ia_rates_mannucci, color=colors[1], label='Ia (manucci)')
    subplot.plot(AgeBin.mins, supernova_Ia_rates_maoz, color=colors[2], label='Ia (maoz)')

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
    plot_file_name : str
        whether to write figure to file and its name. True = use default naming convention
    plot_directory : str
        directory in which to write figure file
    figure_index : int
        index for matplotlib window
    '''
    ia_kind = 'mannucci'

    assert mass_loss_kind in ['rate', 'cumulative']

    AgeBin = ut.binning.BinClass(
        age_limits, age_bin_width, include_max=True, log_scale=age_log_scale
    )

    if mass_loss_kind == 'rate':
        supernova_cc = SupernovaCC.get_rate(AgeBin.mins) * SupernovaCC.ejecta_mass
        supernova_Ia = SupernovaIa.get_rate(AgeBin.mins, ia_kind) * SupernovaIa.ejecta_mass
        wind = StellarWind.get_rate(AgeBin.mins, metallicity, metal_mass_fraction)
    else:
        supernova_cc = SupernovaCC.get_mass_loss_fraction(0, AgeBin.mins, element_name, metallicity)
        supernova_Ia = SupernovaIa.get_mass_loss_fraction(
            0, AgeBin.mins, ia_kind, element_name=element_name
        )
        wind = StellarWind.get_mass_loss_fraction(
            0, AgeBin.mins, metallicity, metal_mass_fraction, element_name
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
        if element_name is not None and len(element_name):
            plot_file_name = f'{element_name}.yield.{mass_loss_kind}_v_time'
        else:
            plot_file_name = f'star.mass.loss.{mass_loss_kind}_v_time'
    ut.plot.parse_output(plot_file_name, plot_directory)
