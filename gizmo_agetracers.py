'''

Post-process metal abundances using the stellar age tracers passive scalars in
the FIRE-3 set of simulations.

@author: Andrew Emerick <aemerick11@gmail.com>

In addition to metallicity fields tracked natively in FIRE-3 (see gizmo_star.py),
the FIRE-3 simulations follow metal enrichment using stellar age bins. Stars
at a given age deposit into their corresponding age bin.

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
from scipy import integrate

from . import gizmo_star
import utilities as ut

# --------------------------------------------------------------------------------------------------
# Agetracers
# --------------------------------------------------------------------------------------------------
class AgetracerClass(dict, ut.io.SayClass):
    '''
    Dictionary class to store information about age tracer bins.
    '''

    def __init__(self, verbose=True):
        self.info = {'flag_agetracers' : 0} # assume off initially
        self.verbose = verbose

        # left edges (in Myr) of stellar age bins plus right edge
        # of last bin
        self.age_bins = None

        return

    def determine_tracer_info(self, directory='.'):
        '''
        Checks for GIZMO_config.h in directory or the 'gizmo' subdirectory in
        directory (if present), or checks for 'gizmo.out' in directory. One of
        these is needed to read in the compile-time flags which determine the
        presence and number of age tracer time bins contained in the output.

        Parameters
        -----------
        directory : str : top-level work directory for files (optional). Default: '.'
        '''
        # use a few different methods to find information
        # search for either GIZMO_config.h or gizmo.out

        possible_files = ["GIZMO_config.h","gizmo_config.h","gizmo.out","gizmo.out.txt"]
        possible_paths = [ut.io.get_path(directory), ut.io.get_path(directory + 'gizmo'),
                          ut.io.get_path(directory) + 'code']

        path_file_name = None
        for fname in possible_files:
            for pname in possible_paths:
                if os.path.exists( pname + fname):
                    path_file_name = pname + fname
                    break

        if path_file_name is None:
            print("Cannot find GIZMO_config.h or gizmo.out in {}".format(directory)+\
                  " or in {}".format(directory + '/gizmo'))
            print("Assuming no age tracer information")
            self.info['flag_agetracers'] = 0
            self.info['flag_agetracers_custom'] = 0
            return

        if "GIZMO_config.h" in path_file_name:
            delimiter = " "
        else:
            delimiter = "="

        self.info['flag_agetracers']        = 0
        self.info['flag_agetracers_custom'] = 0

        self._logbins = True   # assume by default
        count         = 0


        self.info['metallicity_start'] = -1
        for line in open(path_file_name, 'r'):

            #
            # ----------------------  WARNING: ---------------------------------
            #
            #          If the number of age tracers is ever controlled outside
            #          of this flag (e.g. some other flag turns on a default
            #          value), this WILL need to be changed to account for that.
            #

            if "GALSF_FB_FIRE_AGE_TRACERS"+delimiter in line:
                self.info['num_tracers'] = int(line.strip("\n").split(delimiter)[-1])
                self.info['flag_agetracers'] = 1

            if "GALSF_FB_FIRE_AGE_TRACERS_CUSTOM" in line:
                self.info['flag_agetracers_custom'] = 1
                self._logbins = False

            #
            # ----------------------  WARNING: ---------------------------------
            #
            # if number of default elements and r process elements
            # changes from 11 (10 + metallicity) and 4 respecitvely, then
            # this cannot be hard-coded like this any longer. Will need some trickier
            # if statements too in this case to ensure backwards compatability
            #
            # hard-coded values:  11 = number of metals tracked
            #                      4 = number of rprocess fields
            if "FIRE_PHYSICS_DEFAULTS" in line:
                if self.info['metallicity_start'] == -1:
                    self.info['metallicity_start'] = 11 + 4 # first age tracer is field Metallicity_XX
                else:
                    self.info['metallicity_start'] = self.info['metallicity_start'] + 11

            # if this is specified, then more than 4 are being tracked.
            # check this.
            if "GALSF_FB_FIRE_RPROCESS" in line:
                numr = int(line.split(delimiter)[-1])
                if self.info['metallicity_start'] == -1:
                    self.info['metallicity_start'] = 0 + numr
                else:
                    self.info['metallicity_start'] = self.info['metallicity_start'] - 4 + numr

            if count > 400: # arbitrary limit
                break
            count = count + 1

        self.info['metallicity_end'] = self.info['metallicity_start']
        if self.info['flag_agetracers']:
            self.info['metallicity_end'] += self.info['num_tracers']

        return

    def generate_age_bins(self, file_name = 'age_bins.txt', directory = '.'):
        '''
        If age tracers are contained in the output file (as determined by
        determine_tracer_info), generates or reads in the age tracer bins used
        by the simulation. This requires either the presence of params.txt-usedvalues,
        parameters-usedvalues, or gizmo.out in the top-level directory or
        the 'output' subdirectory if log-spaced bins are used. Otherwise, this requires
        the presence of the 'age-bins.txt' input file to read in custom-spaced bins.

        Parameters
        ----------
        file_name : str : Only used if custom space age tracers are used. Name of
                          bin file, assumed to be a single column of bin left edges
                          plus the right edge of last bin (number of lines should be
                          number of tracers + 1). Default : 'age_bins.txt'

        directory : str : Top-level work directory. Default : '.'
        '''

        if self.info['flag_agetracers'] == 0:
            self.age_bins = None

            return

        if self._logbins:

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

                        raise OSError("Cannot find 'params.txt-usedvalues' or 'gizmo.out' in {}".format(directory))

            self.say('* reading age bin information from :  {}\n'.format(path_file_name.strip('./')),
                     verbose=self.verbose)

            for line in open(path_file_name):
                if 'AgeTracerBinStart' in line:
                    self.info['agetracer_bin_start'] = float(line.split(" ")[-1])
                if 'AgeTracerBinEnd'   in line:
                    self.info['agetracer_bin_end']   = float(line.split(" ")[-1])

            binstart = np.log10(self.info['agetracer_bin_start'])
            binend   = np.log10(self.info['agetracer_bin_end'])
            self.age_bins = np.logspace(binstart,binend, self.info['num_tracers'] + 1)
            self.age_bins[0] = 0 # always left edge at zero

        else:

            self.read_age_bins(file_name=file_name, directory=directory)

        if len(self.age_bins) > self.info['num_tracers'] + 1:
            raise RuntimeError("Number of age bins implies there should be more tracers. Something is wrong here.")
        elif len(self.age_bins) < self.info['num_tracers'] + 1:
            raise RuntimeError("Number of age bins implies there should be less tracers. Something is wrong here")

        return

    def read_age_bins(self, file_name = 'age_bins.txt', directory = '.'):
        '''
        Parameters
        ----------
        file_name : str : Only used if custom space age tracers are used. Name of
                          bin file, assumed to be a single column of bin left edges
                          plus the right edge of last bin (number of lines should be
                          number of tracers + 1). Default : 'age_bins.txt'

        directory : str : Top-level work directory. Default : '.'
        '''

        if self.info['flag_agetracers'] == 0:
            self.age_bins = None
            return

        try:
            path_file_name = ut.io.get_path(directory) + file_name
            self.age_bins = np.genfromtxt(path_file_name)
        except OSError:
            raise OSError("cannot find file of age tracer bins: {}".format(path_file_name))


        return


Agetracers = AgetracerClass()

def read_agetracer_times(directory='.', filename = 'age_bins.txt', verbose=True):
    '''
    Within input directory, search for and read stellar age tracer field
    bin times (if custom sized age tracer fields used). Return as
    an array.

    Parameters
    -----------
    directory : str : directory where snapshot times/scale-factor file is

    Returns
    -------
    Agetracers : dictionary class : age tracer information
    '''

    Agetracers = AgetracerClass(verbose=verbose)

    Agetracers.determine_tracer_info(directory = directory)

    Agetracers.generate_age_bins(filename, directory)

    return Agetracers

#
#
# Functions and classes for generating post-processing yield tables
#
#


def construct_yield_table(yield_object, agebins,
                          elements = None, integration_points = None):
    """
    Construct a table of weights to post-process the age tracer fields
    with individual elemental abundances. Given a function which accepts
    elements and time as arguements, builds this table by summing up the total
    yield for each element during each time bin. This integrates over the
    `yield` function in the passed `yield_object` (see below) for each time
    bin.

    Parameters
    -----------

    yield_object : obj
        An object with a required method "yield" that is a function of two arguments:
        the first is time in Gyr and the second is an element
        name in all lowercase (e.g. 'oxygen','carbon'). This
        function must return the instantaneous, specific mass loss rate for that element
        in units of (Msun / Gyr) per solar mass of star formation.

        This object must also have an attribute 'elements' which is the list
        of all elements able to be generated by this yield model. If making
        your own object to generate a yields table,
        for convenience, one can refer to the 'element_*' dictionaries
        in the `utilities` package in `basic/constant.py`.

        If total metallicity is to be computed, 'metals' MUST be one of these elements.

        If `yield_object` contains the attribute 'integration_points', these
        will be passed to the integrator (scipy.integrate.quad). Otherwise these
        can be provided as a separate argument.


    elements : list, optional
        List of elements to generate for the table if only a subset of
        the available elements need to be generated. If None, all possible
        elements in yield_object.elements will be used. Default : None

    integration_points : (sequence of floats, ints), optional
        Points to be passed to scipy.integrate.quad to be careful around. This
        can also be passed if 'integration_points' is an attribute of
        the yields_object. This argument overrides the yield_object
        attribute if provided. Default : None

    Returns
    -------

    yield_table : np.ndarray
        2D array containing the weights for each element in each age-tracer
        time bin with dimensions: N_tracer x N_elements. Each value represents
        the mass (in Msun) of each element produced during each time bin
        per solar mass of star formation.

    """

    # assume to generate this for all elements
    if elements is None:
        elements = yield_object.elements
    else:
        for e in elements:
            assert e in yield_object.elements

    yield_table = np.zeros(  (np.size(agebins)-1, np.size(elements)))


    # grab points to be carful around for integration (if available)
    points = None
    if not (integration_points is None):
        points = integration_points
    else:
        if hasattr(yield_object, 'integration_points'):
            if len(yield_object.integration_points) > 0:
                points = yield_object.integration_points

    # generate yield weights for each age bin
    for i in np.arange(np.size(agebins)-1):

        if i == 0: # ensure min time starts at 0
            min_t = 0.0
        else:
            min_t = agebins[i]

        max_t = agebins[i+1]

        for j,e in enumerate(elements):
            yield_table[i][j] = integrate.quad( yield_object.yields,
                                min_t, max_t,
                                args = (e,), points = points)[0]
    return yield_table


class YieldsObject ():

    def __init__(self, name = ''):
        self.name = name

        self.elements = []


        return

    def yields(self, t, element):
        pass


# ------------------------------------------------------------------------------
# NuGrid yield class object for generating yield tables for age-tracer
# postprocessing using the Sygma module
# ------------------------------------------------------------------------------

try:
    import sygma as _sygma
    NuPyCEE_loaded = True
except:
    NuPyCEE_loaded = False


class NuGrid_yields(YieldsObject):
    '''
    Object designed for use with the construct_yield_table method. This
    object proves the yields from the NuGrid collaboration using the
    Sygma interface. The NuPyCEE code must be installed (and its dependencies)
    in the python path for this object to work.

    https://nugrid.github.io/NuPyCEE/index.html

    '''

    def __init__(self, name = "NuGrid",
                       **kwargs):

        if not NuPyCEE_loaded:
            print("Cannot load the NuPyCEE module Sygma. "+\
                  "This is necessary to use the NuGrid yields."+\
                  "Make sure this is installed in your python path. "+\
                  "For more info: https://nugrid.github.io/NuPyCEE/index.html")

            raise RuntimeError



        super().__init__(name)

        #if elements is None
        #if isinstance(elements,str):
        #    elements = [elements]
        #self.elements = elements

        #if not 'metals' in elements:
        #    self.elements = ['metals'] + self.elements

        self.model_parameters = {}
        # set some defaults:
        self.model_parameters = {'dt' : 1.0E5,  # first dt
                                 'special_timesteps':1000, # number of timesteps (using logspacing alg)
                                 'tend' : 1.4E10, # end time in years
                                 'mgal' : 1.0} # galaxy / SSP mass in solar masses


        for k in kwargs:
            self.model_parameters[k] = kwargs[k]


        # pre-compute sygma model
        self.compute_yields()

        self.elements = self._sygma_model.history.elements
        self.elements = ['metals'] + self.elements

        return

    def compute_yields(self, **kwargs):
        """
        Runs through the Sygma model given model parameters.
        This function is not strictly necessary in a YieldsObject,
        but exists here to pre-compute information needed for
        the `yields` function so it is not re-computed each time
        it is called.
        """

        for k in kwargs:
            self.model_parameters[k] = kwargs[k]

        if not 'iniZ' in self.model_parameters:
            self.model_parameters['iniZ'] = 0.02
            print("Using default iniZ %8.5f"%(self.model_parameters['iniZ']))

        if self.model_parameters['mgal'] != 1.0:
            print("Galaxy mass is not 1 solar mass. Are you sure?")
            print("This will throw off scalings")

        self._sygma_model = _sygma.sygma(**self.model_parameters)

        # get yields. This is the "ISM" mass fraction of all elements. But since
        # mgal above is 1.0, this is the solar masses of each element in the ISM
        # as a function of time. The diff of this is dM. Need to skip first
        # which is the initial values
        skip_index = 1
        skip_h_he  = 2

        self._model_yields = np.array(self._sygma_model.history.ism_elem_yield)[skip_index:,:]
        # construct total metals (2: skips H and He)
        self._total_metals = np.sum(self._model_yields[:,skip_h_he:], axis=1)
        self._model_time   = np.array(self._sygma_model.history.age[1:]) / 1.0E9 # in yr -> Gyr

        # in Msun / Gyr
        #print(np.shape(dt), np.shape(self._model_yields), np.shape(self._model_total_metal_rate))
        dt = np.diff(self._model_time)
        self._model_yield_rate = (np.diff(self._model_yields,axis=0).transpose() / dt).transpose()
        self._model_total_metal_rate = (np.diff(self._total_metals) / dt).transpose()

        return

    def yields(self, t, element):
        """

        Returns the total yields for all yield channels in NuGrid.
        This method is REQUIRED by construct_yield_table.

        Parameters
        -----------
        t    : float or np.ndarray
            Time (in Gyr) to compute instantaneous yield rate
        element : str : Must be in self.elements
            Element name

        Returns
        -----------
        y : float or np.ndarray
            Total yields at a given time for desired element in units of
            Msun / Gyr per Msun of star formation.
        """

        assert element in self.elements

        x = 0.5 * (self._model_time[1:] + self._model_time[:-1])

        if element == 'metals':
            y = self._model_total_metal_rate
        else:
            element_index = self._sygma_model.history.elements.index(element)
            y = self._model_yield_rate[:,element_index]

        return np.interp(t, x, y)

# ------------------------------------------------------------------------------
# FIRE2 Yield Class object for generating yield tables for age-tracer
# post-processing. This serves as an example
# ------------------------------------------------------------------------------

class FIRE2_yields(YieldsObject):
    '''
    Object desigend for use with the construct_yield_table method. This object
    Provides the yields for the default FIRE2 chemical evolution model. This
    model uses some metallicity depended yields, determined by two paramters.
    '''
    def __init__(self, name = "FIRE2", model_Z = 1.0, Z_scaling=True):
        """
        Initialize object and pre-load some things for convenience.

        Parameters
        -----------
        name  : str, optional
            Optional name for this table. Default : FIRE2
        model_Z : float, Optional
            Metallicity (in solar units) for metallicity dependent yield
            scalings in the FIRE2 model. Default: 1.0 (solar)
        Z_scaling : bool, optional
            Apply the FIRE2 metallicity scalings in approximate fashion.
            Default : True
        """

        super().__init__(name)

        # Not required. Specific parameters for this model
        self.model_parameters = {'model_Z' : model_Z,
                                 'Z_scaling' : Z_scaling}

        # Not required, but useful
        self.elements = ['metals','helium','carbon','nitrogen','oxygen',
                         'neon','magnesium','silicon','sulphur',
                         'calcium','iron']


        # Not required
        # to use for metallicity dependent corrections on the yields
        # this just assumes that all yields come from stars with metallicities
        # and individual abundances scaled to the solar abundance pattern
        # this isn't accurate in practice but gives better agreement between
        # post-processed yields and native simulated yields in the FIRE-2 model.
        self._star_massfraction = {}
        for e in self.elements:
            self._star_massfraction[e]    = self.model_parameters['model_Z'] *\
                                      gizmo_star.sun_massfraction[e]

        # pre-load yields since they are constants in time.
        # in general, this probably cannot be done if they are time-varying
        # and would have to make separete function calls or something in
        # the yields method
        self.compute_yields()

        #
        # Points (in Gyr) to be careful around during integration. These are
        # all
        self.integration_points =  np.sort([0.003401, 0.010370, 0.03753,
                                            0.001, 0.05, 0.10, 1.0, 14.0])


        return

    def compute_yields(self, model_Z = None):
        """
        This function is not necessary in a YieldsObject but exists here
        to generate the yields from different channels from the underlying
        model in gizmo_analysis in order keep the `yields` function
        cleaner and to avoid having to re-compute this every time the
        yields function is called.
        """

        if not (model_Z is None):
            self.model_parameters['model_Z'] = model_Z

        if self.model_parameters['model_Z'] is None:
            self.model_parameters['model_Z'] = 1.0    # default

        #  Yields here is a dictionary with element names as kwargs
        # and yields (in Msun) as values
        self.snIa_yields = gizmo_star.get_nucleosynthetic_yields('supernova.ia',
                                                  star_metallicity=self.model_parameters['model_Z'],
                                                  star_massfraction=self._star_massfraction,
                                                  normalize=False)

        self.snII_yields = gizmo_star.get_nucleosynthetic_yields('supernova.ii',
                                              star_metallicity=self.model_parameters['model_Z'],
                                              star_massfraction=self._star_massfraction,
                                              normalize=False)
        #    wind yields do not have quantized rates. These are mass fraction
        #
        self.wind_yields = gizmo_star.get_nucleosynthetic_yields('wind',
                                              star_metallicity=self.model_parameters['model_Z'],
                                              star_massfraction=self._star_massfraction,
                                              normalize=False)
        return

    def yields(self, t, element):
        """

        Returns the total yields for all FIRE processes. This method is REQUIRED
        by construct_yield_table.

        Parameters
        -----------
        t    : float or np.ndarray
            Time (in Gyr) to compute instantaneous yield
        element : str : Must be in self.elements
            Element name

        Returns
        -----------
        y : float or np.ndarray
            Total yields at a given time for desired element in units of
            Msun / Gyr per Msun of star formation.
        """

        assert element in self.elements

        # get SNIa rate at a given time (in units of 1/Myr per Msun of SF)
        snIarate = gizmo_star.SupernovaIa.get_rate(t*1000.0, 'mannucci')

        # get snII rate at given time (in units of 1/Myr per Msun of SF)
        snIIrate = gizmo_star.SupernovaII.get_rate(t*1000.0)

        # get widnd rate at given time (in units of Msun/Myr per Msun of SF)
        #   this is the only model in the FIRE default that is Z dependent
        windrate = gizmo_star.StellarWind.get_rate(t*1000.0, metallicity=self.model_parameters['model_Z'])

        y =  ( (self.wind_yields[element] * windrate) +\
               (self.snIa_yields[element] * snIarate) +\
               (self.snII_yields[element] * snIIrate))  # in Msun / Myr

        return y * 1000.0 # Msun / Gyr
