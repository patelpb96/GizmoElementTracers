'''

Post-process metal abundances using the stellar age tracers passive scalars in
the FIRE-3 set of simulations.

@author: Andrew Emerick <aemerick11@gmail.com>

In addition to metallicity fields tracked natively in FIRE-3 (see gizmo_star.py),
the FIRE-3 simulations follow metal enrichment using stellar age bins. Stars
at a given age deposit into their corresponding age bin. MORE

This post-processing requires one to compute the weightings for each age-bin for
each element. This consists of a table which contains the total amount of mass
of each element produced during each time bin. This can be constructed here
using some default enrichment models, but can also be user-generated entirelly
externally to this routine, or can be generated here if provided with an object
that accepts an element name and time as parameters and returns the instantaneous
mass loss rate of that element at that time.


Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

import numpy as np

from . import gizmo_star
# import utilities as ut


def construct_yield_table(elements, agebins,
                          yield_object,
                          yf = None ,
                          yield_args = (), yield_kwargs = {}
                          rate_function = None,
                          rate_args  = (), rate_kwargs = {}):
    """
    Construct a table of weights to post-process the age tracer fields
    with individual elemental abundances. Given a function which accepts
    elements and time as arguements, builds this table by summing up the total
    yield for each element during each time bin.

    Parameters
    -----------

    elements : list
        List of elements

    yield_function : function
        A function of two arguments, where the first is an element name in all
        lowercase (e.g. o,c,fe,ba,na,eu) and the second is a time in Gyr. This
        function must return the instantaneous, specific mass loss rate for that element
        in units of (Msun / yr) per solar mass of star formation.

        Sometimes, however, yields and rates are given separely in a model. In this
        case, the rates can be provided using rate_function and yields using
        yield_function. In this case, yield_function must return the yield
        per event in Msun, such that yield_function * rate_function gives
        Msun / yr per solar mass of star formation.

    yield_args : tuple, optional
        Arguments for yield_function. Default : empty tuple

    yield_kwargs: dict, optional
        Keword arguemnts for yield function. Default : empty dict

    rate_function : function, optional
        If yields and event rates are a separate function, this function must
        accept an element name and time, returning event rate / yr. Default : None

    rate_args : tuple, optional
        Arguments for rate_function. Default: empty tuple

    rate_kwargs : dict, optional
        Keyword arguments for rate_function. Default: empty dict
    """

    for e in elements:
        assert e in yield_object.elements

    yield_table = np.zeros(  (np.size(agebins)-1, np.size(elements)))

    for i in np.arange(np.size(agebins)-1):

        if i == 0:
            min_t = 0.0
        else:
            min_t = agebins[i]
        max_t = agebins[i+1]

        for j,e in enumerate(elements):

            yield_table[i][j] = integrate.quad( yield_object.yields,
                                min_t, max_t,
                                args = (e,))



    return yield_table 


class YieldsObject ():

    def __init__(self, name = ''):
        self.name = name

        self.elements = []


        return

    def yields(self, t, element):
        pass

    def rate_function(self, t, element):
        pass



class FIRE2_yields_table(YieldsObject):

    def __init__(self, name = "FIRE2", model_Z = 1.0, Z_scaling=True):
        """
        Model_Z is solar
        """

        super().__init__(name)

        self.model_parameters = {'model_Z' : model_Z,
                                 'Z_scaling' : Z_scaling}

        self.elements = ['helium','carbon','nitrogen','oxygen',
                         'neon','magnesium','silicon','sulphur',
                         'calcium','iron']

        return

    def yields(self, t, element):
        """
        Returns the total yields for all FIRE processes. t is in Gyr
        """


        snIa_yields = gizmo_star.get_nucleosynthetic_yields('supernova.ia',
                                                  star_metallicity=self.model_parameters['model_Z'],
                                                  star_massfraction={},
                                                  normalize=False)

        snIarate = gizmo_star.SupernovaIa.get_rate(t, 'mannucci')

        # get snII yields. Returns a dictionary with element names as kwargs
        # and yields (in Msun) as values
        snII_yields = gizmo_star.get_nucleosynthetic_yields('supernova.ii',
                                              star_metallicity=self.model_parameters['model_Z'],
                                              star_massfraction={},
                                              normalize=False)

        # get snII rate at given time (in units of 1/Myr per Msun)
        snIIrate = gizmo_star.SupernovaII.get_rate(t*1000.0)

        wind_yields = gizmo_star.get_nucleosynthetic_yields('wind',
                                              star_metallicity=self.model_parameters['model_Z'],
                                              star_massfraction={},
                                              normalize=False)

        windrate = gizmo_star.get_rate(t, metallicity=self.model_parameters['model_Z'])



        y =  ( (wind_yields[element] * windrate) +\
               (snIa_yields[element] * snIarate) +\
               (snII_yields[element] * snIIrate)) * 1.0E6 # in Msun / yr

        return y
