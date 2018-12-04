'''
Analyze stellar evolution, including supernova rates, stellar mass loss, nucleosynthetic yields,
as implemented in Gizmo.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function
import collections
import numpy as np
from scipy import integrate
# local ----
import utilities as ut


#===================================================================================================
# stellar mass loss
#===================================================================================================
class SupernovaIIClass:
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
        Core-collapse occur from 3.4 to 37.53 Myr after formation:
            3.4 to 10.37 Myr: rate / M_sun = 5.408e-10 yr ^ -1
            10.37 to 37.53 Myr: rate / M_sun = 2.516e-10 yr ^ -1

        Parameters
        ----------
        ages : float or array : age[s] of stellar population [Myr]

        Returns
        -------
        rates : float or array : specific rate[s] [Myr ^ -1 M_sun ^ -1]
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
        age_min : float : min age of stellar population [Myr]
        age_maxs : float or array : max age[s] of stellar population [Myr]

        Returns
        -------
        numbers : float or array : specific number[s] of supernovae events [per M_sun]
        '''
        age_bin_width = 0.01

        if np.isscalar(age_maxs):
            #numbers = integrate.quad(self.get_rate, age_min, age_maxs)[0]
            # this method is more stable for piece-wise (discontinuous) function
            ages = np.arange(age_min, age_maxs + age_bin_width, age_bin_width)
            numbers = self.get_rate(ages).sum() * age_bin_width
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                #numbers[age_i] = integrate.quad(self.get_rate, age_min, age)[0]
                ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                numbers[age_i] = self.get_rate(ages).sum() * age_bin_width

        return numbers

    def get_mass_loss_fraction(self, age_min=0, age_maxs=99, element_name='', metallicity=1.0):
        '''
        Get fractional mass loss via supernova ejecta (ejecta mass per M_sun) in age interval[s].

        Parameters
        ----------
        age_min : float : min age of stellar population [Myr]
        age_maxs : float or array : max age[s] of stellar population [Myr]
        element_name : bool : name of element to get yield of
        metallicity : float : metallicity of star (for Nitrogen yield)

        Returns
        -------
        mass_loss_fractions : float : fractional mass loss (ejecta mass[es] per M_sun)
        '''
        mass_loss_fractions = self.ejecta_mass * self.get_number(age_min, age_maxs)

        if element_name:
            element_yields = get_nucleosynthetic_yields('supernova.ii', metallicity, normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


SupernovaII = SupernovaIIClass()


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
        ages : float : age of stellar population [Myr]
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]
            decreasing to 10 Myr increases total number by ~50%,
            increasing to 100 Myr decreases total number by ~50%

        Returns
        -------
        rate : float : specific rate of supernovae [Myr ^ -1 M_sun ^ -1]
        '''

        def get_rate(ages, kind):
            if kind == 'mannucci':
                # Mannucci, Della Valle, & Panagia 2006
                return 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((ages - 50) / 10) ** 2)  # [Myr ^ -1]
            elif kind == 'maoz':
                # Maoz & Graur 2017
                return 2e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1] best-fit volumetric
                #return 3e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1] hybrid
                #return 6e-7 * (ages / 1e3) ** -1.1  # [Myr ^ -1] galaxy clusters

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
        age_min : float : min age of stellar population [Myr]
        age_maxs : float or array : max age[s] of stellar population [Myr]
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        numbers : float or array : specific number[s] of supernovae [M_sun ^ -1]
        '''
        if np.isscalar(age_maxs):
            numbers = integrate.quad(self.get_rate, age_min, age_maxs, (ia_kind, ia_age_min))[0]
        else:
            numbers = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                numbers[age_i] = integrate.quad(
                    self.get_rate, age_min, age, (ia_kind, ia_age_min))[0]

        return numbers

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, ia_kind='mannucci', ia_age_min=37.53, element_name=''):
        '''
        Get fractional mass loss via supernova ejecta (ejecta mass per M_sun) in age interval[s].

        Parameters
        ----------
        age_min : float : min age of stellar population [Myr]
        age_maxs : float or array : max age[s] of stellar population [Myr]
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]
        element_name : str : name of element to get yield of

        Returns
        -------
        mass_loss_fractions : float or array : mass loss fraction[s] (ejecta mass per M_sun)
        '''
        mass_loss_fractions = (
            self.ejecta_mass * self.get_number(age_min, age_maxs, ia_kind, ia_age_min))

        if element_name:
            element_yields = get_nucleosynthetic_yields('supernova.ia', normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


SupernovaIa = SupernovaIaClass()


class StellarWindClass:
    '''
    Compute mass loss rates rates and cumulative mass loss fractions for stellar winds,
    as implemented in Gizmo.
    '''

    def __init__(self):
        self.ejecta_mass = 1.0  # these already are mass fractions
        self.solar_metal_mass_fraction = 0.02  # Gizmo assumes this

    def get_rate(self, ages, metallicity=1, metal_mass_fraction=None):
        '''
        Get rate of fractional mass loss [Myr ^ -1] from stellar winds.

        Includes all non-supernova mass-loss channels, but dominated by O, B, and AGB-star winds.

        Note: Gizmo assumes solar abundance (total metal mass fraction) of 0.02,
        while Asplund et al 2009 is 0.0134.

        Parameters
        ----------
        age : float or array : age[s] of stellar population [Myr]
        metallicity : float : total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float : mass fration of all metals (everything not H, He)

        Returns
        -------
        rates : float or array : rate[s] of fractional mass loss [Myr ^ -1]
        '''
        metallicity_min = 0.01  # min and max imposed in Gizmo for stellar wind rates for stability
        metallicity_max = 3

        if metal_mass_fraction is not None:
            metallicity = metal_mass_fraction / self.solar_metal_mass_fraction

        metallicity = np.clip(metallicity, metallicity_min, metallicity_max)

        if np.isscalar(ages):
            assert (ages >= 0 and ages < 16000)
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
            assert (np.min(ages) >= 0 and np.max(ages) < 16000)

            ages = np.asarray(ages)
            rates = np.zeros(ages.size)

            # get rate
            masks = np.where(ages <= 1)[0]
            rates[masks] = 11.6846

            masks = np.where((ages > 1) * (ages <= 3.5))[0]
            rates[masks] = (11.6846 * metallicity *
                            ages[masks] ** (1.838 * (0.79 + np.log10(metallicity))))

            masks = np.where((ages > 3.5) * (ages <= 100))[0]
            rates[masks] = 72.1215 * (ages[masks] / 3.5) ** -3.25 + 0.0103

            masks = np.where(ages > 100)[0]
            rates[masks] = 1.03 * (ages[masks] / 1e3) ** -1.1 / (12.9 - np.log(ages[masks] / 1e3))

        rates *= 1e-3  # convert to [Myr ^ -1]

        rates *= 1.4 * 0.291175  # give expected return fraction from stellar winds alone (~17%)

        return rates

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None, element_name=''):
        '''
        Get cumulative fractional mass loss via stellar winds within age interval[s].

        Parameters
        ----------
        age_min : float : min age of stellar population [Myr]
        age_maxs : float or array : max age[s] of stellar population [Myr]
        metallicity : float : total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float : mass fraction of all metals (everything not H, He)
        element_name : str : name of element to get yield of

        Returns
        -------
        mass_loss_fractions : float or array : mass loss fraction[s]
        '''
        if np.isscalar(age_maxs):
            mass_loss_fractions = integrate.quad(
                self.get_rate, age_min, age_maxs, (metallicity, metal_mass_fraction))[0]
        else:
            mass_loss_fractions = np.zeros(len(age_maxs))
            for age_i, age in enumerate(age_maxs):
                mass_loss_fractions[age_i] = integrate.quad(
                    self.get_rate, age_min, age, (metallicity, metal_mass_fraction))[0]

                # this method may be more stable for piece-wise (discontinuous) function
                #age_bin_width = 0.001  # [Myr]
                #ages = np.arange(age_min, age + age_bin_width, age_bin_width)
                #mass_loss_fractions[age_i] = self.get_rate(
                #    ages, metallicity, metal_mass_fraction).sum() * age_bin_width

        if element_name:
            element_yields = get_nucleosynthetic_yields('wind', metallicity, normalize=True)
            mass_loss_fractions *= element_yields[element_name]

        return mass_loss_fractions


StellarWind = StellarWindClass()


class MassLossClass(ut.io.SayClass):
    '''
    Compute mass loss from all channels (supernova II, Ia, stellar winds) as implemented in Gizmo.
    '''

    def __init__(self):
        self.SupernovaII = SupernovaIIClass()
        self.SupernovaIa = SupernovaIaClass()
        self.StellarWind = StellarWindClass()
        self.Spline = None

    def get_rate(
        self, ages, metallicity=1, metal_mass_fraction=None, ia_kind='mannucci', ia_age_min=37.53):
        '''
        Get rate of fractional mass loss [Myr ^ -1] from all stellar evolution channels.

        Parameters
        ----------
        age : float or array : age[s] of stellar population [Myr]
        metallicity : float : total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float : mass fration of all metals (everything not H, He)
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        rates : float or array : fractional mass loss rate[s] [Myr ^ -1]
        '''
        return (self.SupernovaII.get_rate(ages) * self.SupernovaII.ejecta_mass +
                self.SupernovaIa.get_rate(ages, ia_kind, ia_age_min) *
                self.SupernovaIa.ejecta_mass +
                self.StellarWind.get_rate(ages, metallicity, metal_mass_fraction))

    def get_mass_loss_fraction(
        self, age_min=0, age_maxs=99, metallicity=1, metal_mass_fraction=None,
        ia_kind='mannucci', ia_age_min=37.53):
        '''
        Get fractional mass loss via all stellar evolution channels within age interval[s]
        via direct integration.

        Parameters
        ----------
        age_min : float : min (starting) age of stellar population [Myr]
        age_maxs : float or array : max (ending) age[s] of stellar population [Myr]
        metallicity : float : total abundance of metals wrt solar_metal_mass_fraction
        metal_mass_fraction : float : mass fration of all metals (everything not H, He)
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]

        Returns
        -------
        mass_loss_fractions : float or array : mass loss fraction[s]
        '''
        return (self.SupernovaII.get_mass_loss_fraction(age_min, age_maxs) +
                self.SupernovaIa.get_mass_loss_fraction(age_min, age_maxs, ia_kind, ia_age_min) +
                self.StellarWind.get_mass_loss_fraction(
                    age_min, age_maxs, metallicity, metal_mass_fraction))

    def get_mass_loss_fraction_from_spline(
        self, ages=[], metallicities=[], metal_mass_fractions=None):
        '''
        Get fractional mass loss via all stellar evolution channels at ages and metallicities
        (or metal mass fractions) via 2-D (bivariate) spline.

        Parameters
        ----------
        ages : float or array : age[s] of stellar population [Myr]
        metallicities : float or array : total abundance[s] of metals wrt solar_metal_mass_fraction
        metal_mass_fractions : float or array : mass fration[s] of all metals (everything not H, He)

        Returns
        -------
        mass_loss_fractions : float or array : mass loss fraction[s]
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
        self, age_limits=[1, 14000], age_bin_width=0.2,
        metallicity_limits=[0.01, 3], metallicity_bin_width=0.1,
        ia_kind='mannucci', ia_age_min=37.53):
        '''
        Create 2-D bivariate spline (in age and metallicity) for fractional mass loss via
        all stellar evolution channels.

        Parameters
        ----------
        age_limits : list : min and max limits of age of stellar population [Myr]
        age_bin_width : float : log width of age bin [Myr]
        metallicity_limits : list :
            min and max limits of metal abundance wrt solar_metal_mass_fraction
        metallicity_bin_width : float : width of metallicity bin
        ia_kind : str : supernova Ia rate kind: 'mannucci' (Gizmo default), 'maoz' (power law)
        ia_age_min : float : minimum age for supernova Ia to occur [Myr]
        '''
        from scipy import interpolate

        age_min = 0

        self.AgeBin = ut.binning.BinClass(
            age_limits, age_bin_width, include_max=True, scaling='log')
        self.MetalBin = ut.binning.BinClass(
            metallicity_limits, metallicity_bin_width, include_max=True, scaling='log')

        self.say('* generating 2-D spline to compute stellar mass loss from age + metallicity')
        self.say('number of age bins = {}'.format(self.AgeBin.number))
        self.say('number of metallicity bins = {}'.format(self.MetalBin.number))

        self.mass_loss_fractions = np.zeros((self.AgeBin.number, self.MetalBin.number))
        for metallicity_i, metallicity in enumerate(self.MetalBin.mins):
            self.mass_loss_fractions[:, metallicity_i] = self.get_mass_loss_fraction(
                age_min, self.AgeBin.mins, metallicity, None, ia_kind, ia_age_min)

        self.Spline = interpolate.RectBivariateSpline(
            self.AgeBin.mins, self.MetalBin.mins, self.mass_loss_fractions)


MassLoss = MassLossClass()


def plot_supernova_v_age(
    age_limits=[1, 3000], age_bin_width=1, age_scaling='log',
    y_axis_kind='rate', y_axis_limits=[None, None], y_axis_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot specific rates or cumulative numbers of supernovae (core-collapse + Ia) [per M_sun]
    versus stellar age.

    Parameters
    ----------
    age_limits : list : min and max limits of age of stellar population [Myr]
    age_bin_width : float : width of stellar age bin [Myr]
    age_scaling : str : 'log' or 'linear'
    y_axis_limits : str : 'rate' or 'number'
    y_axis_limits : list : min and max limits to impose on y-axis
    y_axis_scaling : str : 'log' or 'linear'
    write_plot : bool : whether to write plot to file
    plot_directory : str : where to write plot file
    figure_index : int : index for matplotlib window
    '''
    assert y_axis_kind in ['rate', 'number']

    AgeBin = ut.binning.BinClass(age_limits, age_bin_width, include_max=True)

    if y_axis_kind == 'rate':
        supernova_II_rates = SupernovaII.get_rate(AgeBin.mins)
        supernova_Ia_rates_mannucci = SupernovaIa.get_rate(AgeBin.mins, 'mannucci')
        supernova_Ia_rates_maoz = SupernovaIa.get_rate(AgeBin.mins, 'maoz')
    elif y_axis_kind == 'number':
        supernova_II_rates = SupernovaII.get_number(min(age_limits), AgeBin.maxs)
        supernova_Ia_rates_mannucci = SupernovaIa.get_number(
            min(age_limits), AgeBin.maxs, 'mannucci')
        supernova_Ia_rates_maoz = SupernovaIa.get_number(
            min(age_limits), AgeBin.maxs, 'maoz')

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot, age_scaling, age_limits, None, y_axis_scaling, y_axis_limits,
        [supernova_II_rates, supernova_Ia_rates_mannucci, supernova_Ia_rates_maoz])

    subplot.set_xlabel('star age $\\left[ {\\rm Myr} \\right]$')
    if y_axis_kind == 'rate':
        subplot.set_ylabel('SN rate $\\left[ {\\rm Myr}^{-1} {\\rm M}_\odot^{-1} \\right]$')
    elif y_axis_kind == 'number':
        subplot.set_ylabel('SN number $\\left[ {\\rm M}_\odot^{-1} \\right]$')

    colors = ut.plot.get_colors(3, use_black=False)

    subplot.plot(AgeBin.mins, supernova_II_rates, color=colors[0], label='II')
    subplot.plot(AgeBin.mins, supernova_Ia_rates_mannucci, color=colors[1], label='Ia (manucci)')
    subplot.plot(AgeBin.mins, supernova_Ia_rates_maoz, color=colors[2], label='Ia (maoz)')

    ut.plot.make_legends(subplot, 'best')

    if y_axis_kind == 'rate':
        plot_name = 'supernova.rate_v_time'
    elif y_axis_kind == 'number':
        plot_name = 'supernova.number.cum_v_time'
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


def plot_mass_loss_v_age(
    age_limits=[1, 10000], age_bin_width=0.01, age_scaling='log',
    mass_loss_kind='rate', mass_loss_limits=[None, None], mass_loss_scaling='log',
    element_name='',
    metallicity=1, metal_mass_fraction=None,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Compute mass loss from all channels (supernova II, Ia, stellar winds) versus stellar age.

    Parameters
    ----------
    age_limits : list : min and max limits of age of stellar population [Myr]
    age_bin_width : float : width of stellar age bin [Myr]
    age_scaling : str : 'log' or 'linear'
    mass_loss_kind : str : 'rate' or 'cumulative'
    mass_loss_limits : list : min and max limits to impose on y-axis
    mass_loss_scaling : str : 'log' or 'linear'
    element_name : str : name of element to get yield of (if None, compute total mass loss)
    metallicity : float : total abundance of metals wrt solar_metal_mass_fraction
    metal_mass_fraction : float : mass fration of all metals (everything not H, He)
    write_plot : bool : whether to write plot to file
    plot_directory : str : where to write plot file
    figure_index : int : index for matplotlib window
    '''
    ia_kind = 'mannucci'

    assert mass_loss_kind in ['rate', 'cumulative']

    AgeBin = ut.binning.BinClass(age_limits, age_bin_width, scaling=age_scaling, include_max=True)

    if mass_loss_kind == 'rate':
        supernova_II = SupernovaII.get_rate(AgeBin.mins) * SupernovaII.ejecta_mass
        supernova_Ia = SupernovaIa.get_rate(AgeBin.mins, ia_kind) * SupernovaIa.ejecta_mass
        wind = StellarWind.get_rate(AgeBin.mins, metallicity, metal_mass_fraction)
    else:
        supernova_II = SupernovaII.get_mass_loss_fraction(0, AgeBin.mins, element_name, metallicity)
        supernova_Ia = SupernovaIa.get_mass_loss_fraction(
            0, AgeBin.mins, ia_kind, element_name=element_name)
        wind = StellarWind.get_mass_loss_fraction(
            0, AgeBin.mins, metallicity, metal_mass_fraction, element_name)

    total = supernova_II + supernova_Ia + wind

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot, age_scaling, age_limits, None, mass_loss_scaling, mass_loss_limits,
        [supernova_II, supernova_Ia, wind, total])

    subplot.set_xlabel('star age $\\left[ {\\rm Myr} \\right]$')
    if mass_loss_kind == 'rate':
        subplot.set_ylabel('mass loss rate $\\left[ {\\rm Myr}^{-1} \\right]$')
    else:
        y_axis_label = 'fractional mass loss'
        if element_name:
            y_axis_label = '{} yield per ${{\\rm M}}_\odot$'.format(element_name)
        subplot.set_ylabel(y_axis_label)

    colors = ut.plot.get_colors(3, use_black=False)

    subplot.plot(AgeBin.mins, supernova_II, color=colors[0], label='supernova II')
    subplot.plot(AgeBin.mins, supernova_Ia, color=colors[1], label='supernova Ia')
    subplot.plot(AgeBin.mins, wind, color=colors[2], label='stellar winds')
    subplot.plot(AgeBin.mins, total, color='black', linestyle=':', label='total')

    ut.plot.make_legends(subplot, 'best')

    plot_name = 'star.mass.loss.' + mass_loss_kind + '_v_time'
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


#===================================================================================================
# nucleosynthetic yields
#===================================================================================================
def get_nucleosynthetic_yields(
    event_kind='supernova.ii', star_metallicity=1.0, normalize=True):
    '''
    Get nucleosynthetic element yields, according to input event_kind.
    This is the *additional* nucleosynthetic yields that are added to a stellar population's
    existing abundances (so these are not the total elemental masses that get deposited to gas).

    Parameters
    ----------
    event_kind : str : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float :
        total metallicity of star prior to event, relative to solar = sun_metal_mass_fraction
    normalize : bool : whether to normalize yields to be mass fractions (instead of masses)

    Returns
    -------
    yields : ordered dictionary : yield mass [M_sun] or mass fraction for each element
        can covert to regular dictionary via dict(yields) or list of values via yields.values()
    '''
    sun_metal_mass_fraction = 0.02  # total metal mass fraction of Sun that Gizmo assumes

    yield_dict = collections.OrderedDict()
    yield_dict['metals'] = 0.0
    yield_dict['helium'] = 0.0
    yield_dict['carbon'] = 0.0
    yield_dict['nitrogen'] = 0.0
    yield_dict['oxygen'] = 0.0
    yield_dict['neon'] = 0.0
    yield_dict['magnesium'] = 0.0
    yield_dict['silicon'] = 0.0
    yield_dict['sulphur'] = 0.0
    yield_dict['calcium'] = 0.0
    yield_dict['iron'] = 0.0

    assert event_kind in ['wind', 'supernova.ii', 'supernova.ia']

    star_metal_mass_fraction = star_metallicity * sun_metal_mass_fraction

    if event_kind == 'wind':
        # include all non-supernova mass-loss channels, but dominated by O, B, and AGB-star winds
        # compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004
        # synthesized in Wiersma et al 2009b
        # treat AGB and O-star yields in more detail for light elements
        ejecta_mass = 1.0  # these yields already are mass fractions

        yield_dict['helium'] = 0.36
        yield_dict['carbon'] = 0.016
        yield_dict['nitrogen'] = 0.0041
        yield_dict['oxygen'] = 0.0118

        # oxygen yield strongly depends on initial metallicity of star
        if star_metal_mass_fraction < 0.033:
            yield_dict['oxygen'] *= star_metal_mass_fraction / sun_metal_mass_fraction
        else:
            yield_dict['oxygen'] *= 1.65

        for k in yield_dict:
            if k is not 'helium':
                yield_dict['metals'] += yield_dict[k]

    elif event_kind == 'supernova.ii':
        # yields from Nomoto et al 2006, including hypernovae, averaged over Kroupa 2001 IM
        ejecta_mass = 10.5  # [M_sun]

        yield_dict['metals'] = 2.0
        yield_dict['helium'] = 3.87
        yield_dict['carbon'] = 0.133
        yield_dict['nitrogen'] = 0.0479
        yield_dict['oxygen'] = 1.17
        yield_dict['neon'] = 0.30
        yield_dict['magnesium'] = 0.0987
        yield_dict['silicon'] = 0.0933
        yield_dict['sulphur'] = 0.0397
        yield_dict['calcium'] = 0.00458
        yield_dict['iron'] = 0.0741

        yield_nitrogen_orig = np.float(yield_dict['nitrogen'])

        # nitrogen yield depends on initial metallicity of star
        if star_metal_mass_fraction < 0.033:
            yield_dict['nitrogen'] *= star_metal_mass_fraction / sun_metal_mass_fraction
        else:
            yield_dict['nitrogen'] *= 1.65

        # correct total metal mass for nitrogen correction
        yield_dict['metals'] += yield_dict['nitrogen'] - yield_nitrogen_orig

    elif event_kind == 'supernova.ia':
        # yields from Iwamoto et al 1999, W7 model, averaged over Kroupa 2001 IM
        ejecta_mass = 1.4  # [M_sun]

        yield_dict['metals'] = 1.4
        yield_dict['helium'] = 0.0
        yield_dict['carbon'] = 0.049
        yield_dict['nitrogen'] = 1.2e-6
        yield_dict['oxygen'] = 0.143
        yield_dict['neon'] = 0.0045
        yield_dict['magnesium'] = 0.0086
        yield_dict['silicon'] = 0.156
        yield_dict['sulphur'] = 0.087
        yield_dict['calcium'] = 0.012
        yield_dict['iron'] = 0.743

    if normalize:
        for k in yield_dict:
            yield_dict[k] /= ejecta_mass

    return yield_dict


def plot_nucleosynthetic_yields(
    event_kind='wind', star_metallicity=0.1, normalize=False,
    axis_y_scaling='linear', axis_y_limits=[1e-3, None],
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot nucleosynthetic element yields, according to input event_kind.

    Parameters
    ----------
    event_kind : str : stellar event: 'wind', 'supernova.ia', 'supernova.ii'
    star_metallicity : float : total metallicity of star prior to event, relative to solar
    normalize : bool : whether to normalize yields to be mass fractions (instead of masses)
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    axis_y_limits : list : min and max limits of y-axis
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    title_dict = {
        'wind': 'Stellar Wind',
        'supernova.ii': 'Supernova: Core Collapse',
        'supernova.ia': 'Supernova: Ia',
    }

    yield_dict = get_nucleosynthetic_yields(event_kind, star_metallicity, normalize)

    yield_indices = np.arange(1, len(yield_dict))
    yield_values = np.array(yield_dict.values())[yield_indices]
    yield_names = np.array(yield_dict.keys())[yield_indices]
    yield_labels = [str.capitalize(ut.constant.element_symbol_from_name[k]) for k in yield_names]
    yield_indices = np.arange(yield_indices.size)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)
    subplots = [subplot]

    colors = ut.plot.get_colors(yield_indices.size, use_black=False)

    for si in range(1):
        subplots[si].set_xlim([yield_indices.min() - 0.5, yield_indices.max() + 0.5])
        subplots[si].set_ylim(ut.plot.get_axis_limits(yield_values, axis_y_scaling, axis_y_limits))

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
                    yield_indices[yi], yield_values[yi], 'o', markersize=14, color=colors[yi])
                subplots[si].text(
                    yield_indices[yi] * 0.98, yield_values[yi] * 0.6, yield_labels[yi])

        subplots[si].set_title(title_dict[event_kind])

        ut.plot.make_label_legend(
            subplots[si], '$\\left[ Z / {\\rm Z}_\odot={:.3f} \\right]$'.format(star_metallicity))

    plot_name = 'element.yields_{}_Z.{:.2f}'.format(event_kind, star_metallicity)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)
