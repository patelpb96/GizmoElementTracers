'''
Accomplishes the same as gizmo_star.py, but with added generality and the ability to perturb the main models 
for use with the agetracers. 

Currently, only SNe Ia can be perturbed (normalization and delay time).

See gizmo_star.py for additional details. 

----------
@author: Preet Patel <pbpa@ucdavis.edu>
----------
Units

Unless otherwise noted, all quantities are in (combinations of)
    mass [M_sun]
    time [Myr] (note: most other modules in this GizmoAnalysis package default to Gyr)
    elemental abundance [linear mass fraction]
'''

from http.client import NOT_ACCEPTABLE
from logging import error
from operator import iadd
import numpy as np
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt

import utilities as ut

global Z_0
Z_0 = 1.0

feedback_type = ['wind', 'ia', 'mannucci', 'maoz', 'cc']

#timespan = np.logspace(0, 4.1367, 3000)

# Yield Dictionaries for feedback types
element_yield_wind = {'metals' : 0.0, 'helium' : 0.36, 'carbon' : 0.016, 'nitrogen' : 0.0041, 'oxygen' : 0.0118, 'neon' : 0,
            'magnesium' : 0, 'silicon' : 0, 'sulfur' : 0, 'calcium' : 0, 'iron' : 0}

element_yield_cc = {'metals' : 0.19, 'helium' : 0.369, 'carbon' : 0.0127, 'neon' : 0.00456, 'oxygen' : 0.111, 'neon' : 0.0381,
            'magnesium' : 0.00940, 'silicon' : 0.00889, 'sulfur' : 0.00378, 'calcium' : 0.000436, 'iron' : 0.00706, 'nitrogen' : 1.32e-3}
element_yield_mannucci = {'metals' : 1.0, 'helium' : 0.0, 'carbon' : 0.035, 'nitrogen' : 8.57e-7, 'oxygen' : 0.102, 'neon' : 0.00321,
            'magnesium' : 0.00614, 'silicon' : 0.111, 'sulfur' : 0.0621, 'calcium' : 0.00857, 'iron' : 0.531}
element_yield_ia = {'metals' : 1.0, 'helium' : 0.0, 'carbon' : 0.035, 'nitrogen' : 8.57e-7, 'oxygen' : 0.102, 'neon' : 0.00321,
            'magnesium' : 0.00614, 'silicon' : 0.111, 'sulfur' : 0.0621, 'calcium' : 0.00857, 'iron' : 0.531}

# Ejecta Masses for Each Event 
ejecta_masses = {'wind' : 1,
                 'ia': 1.4,
                 'cc' : 10.5} # corresponding directly to the above

def get_simulation_directory(dirkey = False):

    dirset = dirkey.lower()

    if dirset == 'stampede2':
        dirs = { 'm11b' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m11b_res260' ,
            'm11b_2100' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m11b_res2100_no-swb_contaminated',
            'm11q' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m11q_res880',
            'm11h' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m11h_res880',
            'm09_30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m09_res30',
            'm11b_new' : '/scratch/projects/xsede/GalaxiesOnFIRE/core/m11b_res2100',
            'm11h_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/cr_suite/m11h/cr_700',
            'm09_2e2_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_m2e2/core',
            'm09_res30_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res30',
            'm09_res250_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res250' ,
            'm09_250_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/uv_background/m09_res250_uvb-late',
            'm09_250_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/core/m09_res250',
            'm09_250_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/snia_variation/m09_res250_uvb-late_snia-maoz',
            'm09_30_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/snia_variation/m09_res30_uvb-late_snia-maoz',
            'm10q' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res250',
            'm10_2e2' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m10q_m2e2/core',
            'm10_res30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res30',
            'm10_res250' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res250',
            'm10_res30_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m10q_res30',
            'm10_res250_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m10q_res250',
            'test' : 'test'}

        return dirs

    if dirset == 'epsilon':
        dirs = {'m09_30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m09_res30',
            'm09_2e2_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_m2e2/core',
            'm09_res30_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res30',
            'm09_res250_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res250' ,
            'm09_250_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/core/m09_res250',
            'm09_250_new' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_r250',
            'm09_30_new' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_r30',
            'm10_2e2' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m10q_m2e2/core',
            'm10_res30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res30',
            'm10_res250' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res250',
            'm10_res30_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m10q_res30'
            }

        return dirs

    if dirset == 'm09':
        dirs = {'m09_30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m09_res30',
            'm09_2e2_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_m2e2/core',
            'm09_res30_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res30',
            'm09_res250_cw' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m09_res250' ,
            #'m09_250_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/uv_background/m09_res250_uvb-late',
            'm09_250_core' : '/scratch/projects/xsede/GalaxiesOnFIRE/core/m09_res250',
            'm09_250_new' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_r250',
            'm09_30_new' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m09_r30'
            #'m09_250_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/snia_variation/m09_res250_uvb-late_snia-maoz',
            #'m09_30_uvb' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/snia_variation/m09_res30_uvb-late_snia-maoz'
            }

        return dirs

    if dirset == 'm10':
        dirs = {'m10_2e2' : '/scratch/projects/xsede/GalaxiesOnFIRE/fire3/m10q_m2e2/core',
            'm10_res30' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res30',
            'm10_res250' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res250',
            'm10_res30_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m10q_res30',
            #'m10_res250_cr' : '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/cr_heating_fix/m10q_res250'
            }

        return dirs

    if dirset == 'peloton':
        print("In development, sorry")
        return 0

    print("Something went wrong in directory loading - i.e. are you passing machine == 'stampede2' ? ")



def get_sun_massfraction(model='fire2'):

    model = model.lower()
    assert model in ['fire2', 'fire2.1', 'fire2.2', 'fire3']

    if model == 'fire2':
        solar = {'metals' : 0.02,  # total of all metals (everything not H, He)
            'helium' : 0.28,
            'carbon' : 3.26e-3,
            'nitrogen' : 1.32e-3,
            'oxygen' : 8.65e-3,
            'neon' : 2.22e-3,
            'magnesium' : 9.31e-4,
            'silicon' : 1.08e-3,
            'sulfur' : 6.44e-4,
            'calcium' : 1.01e-4,
            'iron' : 1.73e-3}

        return solar

def element_yields(source = None, includeZ = False, plot = False):

    '''
    returns a dictionary with yields for different feedback event types (CCSN, SNe Ia, Winds)

    source: one of 'ccsn', 'mannucci', 'maoz', or 'winds'
    includeZ: progenitor metallicity dependence (set to False by default)
    plot: will include a plot of the yields for a given feedback event type. Set True to plot, False is default.
    '''
    
    if source.lower() is None:
        raise Exception("Source not defined - i.e. - pass element_yields('cc') for CCSN")
    if source.lower() not in feedback_type:
        raise Exception("Please use one of " + str(feedback_type) + " for source (match case).")
    
    string = "element_yield_"+str(source)
    if(plot == True):
        print("Plotting enabled.")
        x,y = zip(*globals()[string].items())

        plt.figure(figsize = (8,6))
        plt.xticks(size = 15)
        plt.title("nucleosynthetic yields: " + str(source))

        if(includeZ or source == 'wind'):
            plt.scatter(x,y, s = 50)
            if(includeZ):
                print("Z_" + str(source) + ": " + str(y[0]))

        if(source != 'wind'):
            plt.scatter(x[1:],y[1:], s = 50)

        plt.ylabel(r"M/M$_\odot$")
        plt.grid(ls = "-.", alpha = 0.5)

        plt.show()
        
    return globals()[string] #globals() converts the [string] into a variable (or function, if needed). Works like eval()

def _to_num(num):

    list_i = isinstance(num, (list, np.ndarray))
    num_i = isinstance(num, (int, float))

    if list_i:
        return float(num[0])
    if num_i:
        return float(num)

class feedback:

    def __init__(self, 
    time_span = None, 
    source = 'any', 
    elem_name = False, 
    t_w = [1.0, 3.5, 100], 
    t_cc = [3.4, 10.37, 37.53], 
    t_ia = [37.53], 
    n_ia = 1.6e-5, 
    n_cc = False,
    n_w = False,
    ia_model = 'mannucci'):

        '''
        source: one of 'wind', 'ia', or 'cc'
        t_w: transition times for winds (list)
        t_cc: transition times for CCSN
        t_ia: transition times for SNe Ia
        n_ia: type ia normalization 
        
        ia_model: 'maoz' or 'mannucci' (mannucci by default)

        '''
        self.source = source
        self.element = elem_name

        if time_span is not None:
            if len(time_span) > 1:
                print("!! This input only accepts a list of length 1 for time_span - using the first index (time_span[0] = " + str (time_span[0]) + ") only. !!")
            if time_span[0] > 1:
                #print(time_span[0])
                self.timespan = np.array(time_span)
            if time_span[0] <= 1:
                #print("time_span[0] = " + str(time_span[0]))
                self.timespan = time_span
        else:
            self.timespan = np.logspace(0, 4.1367, 1500)
                

        self.trans_w = np.array(t_w) # transition ages of winds
        self.trans_ia = np.array(t_ia) # transition ages of SNe Ia
        self.trans_cc = np.array(t_cc) # transition ages of CCSNe
        self.ia_model = ia_model.lower()
        self.ia_norm = n_ia
        self.cc_norm = n_cc 
        self.w_norm = n_w


    def get_rate_wind(self, Z = Z_0, massloss = True, metal_mass_fraction = None,  plot = False):

        '''
        Returns the rates versus stellar age for stellar winds in FIRE-2. 

        Z: progenitor metallicity. Set to 1 by default, do not change for FIRE-2
        massloss: not sure why this is here, will be removed.
        metal_mass_fraction: was going to add as a potential fix to an inherent metallicity dependence built into FIRE-2. I did not implement this, but I could.
        plot: set to True to plot the results. 


        '''
    
        transition_ages = self.trans_w

        # Imposed mins and maxes based on FIRE-2 and FIRE-3. For stability or something
        metallicity_min = 0.01
        metallicity_max = 3
        age_min = 0  # Myr
        age_max = 13700

        # MODEL(s) BELOW
        if len(self.timespan) == 1:
            t = _to_num(self.timespan)
            ##########################################################
            # Piecewise function for continuous function integration #
            ##########################################################

            if t <= transition_ages[0]:
                r_wind = 4.76317 * Z

            elif t <= transition_ages[1]:
                r_wind = 4.76317 * Z * t ** (1.838 * (0.79 + np.log10(Z)))

            elif t <= transition_ages[2]:
                r_wind = 29.4 * (t / 3.5) ** -3.25 + 0.0041987

            else:
                r_wind = 0.41987 * (t / 1e3) ** -1.1 / (12.9 - np.log(t / 1e3))

            if self.element:
                return element_yields(self.source)[self.element]*r_wind/1e3, self.timespan, transition_ages
            
            else:
                return r_wind/1e3, self.timespan, transition_ages

        ###########################################################################################
        # Piecewise function for pre-generating a table of values and then performing operations. #
        ###########################################################################################

        mask1 = [True if 0 < i <= transition_ages[0] else False for i in self.timespan]
        mask2 = [True if transition_ages[0] <= i <= transition_ages[1] else False for i in self.timespan]
        mask3 = [True if transition_ages[1] <= i <= transition_ages[2] else False for i in self.timespan]
        mask4 = [True if transition_ages[2] <= i else False for i in self.timespan]

        #func1 = 4.76317 * Z * (self.timespan[mask1]/self.timespan[mask1]) # FIRE-2


        func1 = 4.76317 * (self.timespan[mask1]/self.timespan[mask1]) # FIRE-2.1
        func2 = 4.76317 * Z * self.timespan[mask2] ** (1.838 * (0.79 + np.log10(Z)))
        func3 = 29.4 * (self.timespan[mask3] / 3.5) ** -3.25 + 0.0041987
        func4 = 0.41987 * (self.timespan[mask4] / 1e3) ** -1.1 / (12.9 - np.log(self.timespan[mask4] / 1e3))

        # MODEL ABOVE

        r_wind = np.array([*func1, *func2, *func3, *func4], dtype = 'object')/1e3 # y-axis: rate
        a_wind = np.array([*self.timespan[mask1], *self.timespan[mask2], *self.timespan[mask3], *self.timespan[mask4]], dtype = 'object') # x-axis: age

        if self.element:
            #print("Selected " + str(self.element) + " yields for " + str(self.source))
            #print(element_yields(self.source)[self.element])

            if plot:
                plt.loglog(a_wind, element_yields(self.source)[self.element]*r_wind, label = "Wind")
            
            return element_yields(self.source)[self.element]*r_wind, a_wind, transition_ages

        if plot:
            plt.loglog(a_wind, r_wind, label = "Wind")

        return r_wind, a_wind, transition_ages

    def get_rate_cc(self, Z = Z_0, massloss = True, metal_mass_fraction = None, plot = False):
    
        transition_ages = self.trans_cc

        if len(self.timespan) == 1:
            t = _to_num(self.timespan)
            ##########################################################
            # Piecewise function for continuous function integration #
            ##########################################################

            if 0 <= t <= transition_ages[0]:
                r_cc = 0

            if transition_ages[0] <= t <= transition_ages[1]:
                r_cc = 5.408e-4 * t
            if transition_ages[1] <= t <= transition_ages[2]:
                r_cc = 2.516e-4 * t

            if transition_ages[2] <= t:
                r_cc = 0

            if self.element:
                return ejecta_masses[self.source]*element_yields(self.source)[self.element]*r_cc, self.timespan, transition_ages
            
            else:
                return ejecta_masses[self.source]*r_cc, self.timespan, transition_ages
    
        mask1 = [True if 0 <= i <= transition_ages[0] else False for i in self.timespan]
        mask2 = [True if transition_ages[0] <= i <= transition_ages[1] else False for i in self.timespan]
        mask3 = [True if transition_ages[1] <= i <= transition_ages[2] else False for i in self.timespan]
        mask4 = [True if transition_ages[2] <= i else False for i in self.timespan]

        func1 = 0*(self.timespan[mask1]/self.timespan[mask1])
        func2 = 5.408e-4*(self.timespan[mask2]/self.timespan[mask2])
        func3 = 2.516e-4*(self.timespan[mask3]/self.timespan[mask3])
        func4 = 0*(self.timespan[mask4]/self.timespan[mask4])

        r_cc = np.array([*func1, *func2, *func3, *func4], dtype = 'object') # y-axis: rate
        a_cc = np.array([*self.timespan[mask1], *self.timespan[mask2], *self.timespan[mask3], *self.timespan[mask4]], dtype = 'object') # x-axis: age

        if massloss == True:
            r_cc *= ejecta_masses[self.source]

            if self.element is not False:
                #print("Selected " + str(self.element) + " yields for " + str(self.source))
                #print(element_yields(self.source)[self.element])

                if plot:
                    plt.loglog(a_cc, element_yields(self.source)[self.element]*r_cc, label = "CCSN")

                return element_yields(self.source)[self.element]*r_cc, a_cc, transition_ages

        if self.element:
            #print("Selected " + str(self.element) + " yields for " + str(self.source))
            #print(element_yields(self.source)[self.element])

            return element_yields(self.source)[self.element]*r_cc, a_cc, transition_ages

        if plot:
            plt.loglog(a_cc, r_cc, label = "CCSN")

        return r_cc, a_cc, transition_ages

    def get_rate_ia(self, Z = Z_0, massloss = True, metal_mass_fraction = None, plot = False):

        transition_ages = self.trans_ia
        model_version = self.ia_model
        ia_norm = self.ia_norm
        t = _to_num(self.timespan)

        if model_version == 'mannucci':

            if len(self.timespan) == 1:
                if t < transition_ages[0]:
                    r_ia = 0
                else:
                    r_ia = 5.3e-8 + ia_norm * np.exp(-0.5 * ((t - 50) / 10) ** 2)

                if self.element:
                    #print("at t = " + str(t) + ", r_ia = ")
                    rfin = ejecta_masses[self.source]*element_yields(self.source)[self.element]*r_ia
                    #print("at t = " + str(t) + ", r_ia = " + str(rfin))

                    return rfin, self.timespan, transition_ages
                else:
                    return ejecta_masses[self.source]*r_ia, self.timespan, transition_ages

            mask1 = [True if 0 <= i <= transition_ages[0] else False for i in self.timespan]
            mask2 = [True if transition_ages[0] <= i else False for i in self.timespan]

            func1 = 0*(self.timespan[mask1]/self.timespan[mask1])
            func2 = 5.3e-8 + ia_norm * np.exp(-0.5 * ((self.timespan[mask2] - 50) / 10) ** 2)

            a_ia = np.array([*self.timespan[mask1], *self.timespan[mask2]], dtype = 'object') # x-axis: age
            r_ia = ejecta_masses[self.source]*np.array([*func1, *func2], dtype = 'object') # y-axis: rate

            if self.element:
                #print("Selected " + str(self.element) + " yields for " + str(self.source)) # diagnostic - are you selecting the right element?
                #print(element_yields(self.source)[self.element]) # diagnostic - reading the yield values from the yield_dictionary

                if plot:
                    plt.loglog(a_ia, element_yields(self.source)[self.element]*r_ia, label = "Mannucci")

                return element_yields(self.source)[self.element]*r_ia, a_ia, transition_ages

        if model_version == 'maoz':
            #print("Used Maoz for Rates")

            if len(self.timespan) == 1:
                if 0 <= self.timespan < transition_ages[0]:
                    r_ia = 0 * self.timespan
                if transition_ages[0] <= self.timespan:
                    r_ia = 2.6e-7 * (self.timespan / 1e3) ** -1.1

                if self.element:
                    return ejecta_masses[self.source]*element_yields(self.source)[self.element]*r_ia, self.timespan, transition_ages
                else:
                    return ejecta_masses[self.source]*r_ia, self.timespan, transition_ages

            mask1 = [True if 0 <= i <= transition_ages[0] else False for i in self.timespan]
            mask2 = [True if transition_ages[0] <= i else False for i in self.timespan]

            func1 = 0*(self.timespan[mask1]/self.timespan[mask1])
            func2 = 2.6e-7 * (self.timespan[mask2] / 1e3) ** -1.1

            a_ia = np.array([*self.timespan[mask1], *self.timespan[mask2]], dtype = 'object') # x-axis: age
            r_ia = ejecta_masses[self.source]*np.array([*func1, *func2], dtype = 'object') # y-axis: rate

            if self.element:
                #print("Selected " + str(self.element) + " yields for " + str(self.source)) # diagnostic - are you selecting the right element?
                #print(element_yields(self.source)[self.element]) # diagnostic - reading the yield values from the yield_dictionary

                if plot:
                    plt.loglog(a_ia, element_yields(self.source)[self.element]*r_ia, label = "Maoz")

                return element_yields(self.source)[self.element]*r_ia, a_ia, transition_ages

            if plot:
                plt.loglog(a_ia, r_ia, label = "Maoz")

            return r_ia, a_ia, transition_ages

        if plot:
            plt.loglog(a_ia, r_ia, label = "Mannucci")

        return r_ia, a_ia, transition_ages

    def integrate_massloss(self, Z = Z_0, metal_mass_fraction = None, plot = False, ageBins = None, ia_ver = 'mannucci'):
        elem = self.element
        ia_model = ia_ver

        #print(str(self.source) + " selected.") #diagnostic

        if self.source == 'wind':
            

            #For use with agetracers 
            if ageBins is not None:
                r_w, a_w, t_w = self.get_rate_wind()

                mask = np.logical_and(ageBins[0] <= a_w, a_w <= ageBins[1])


                i_w = integrate.trapz(r_w[mask]/len(r_w[mask]), x = [ageBins[0], ageBins[1]])#, a_w[mask])

                return a_w[1:], i_w

            r_w, a_w, t_w = self.get_rate_wind()
            # For raw plotting purposes (or whatever other relevant use case comes up)
            i_w = integrate.cumtrapz(r_w, a_w)

            return a_w[1:], i_w

        if self.source == 'ia':
            print("INTEGRATOR IS USING " + str(ia_model))
            r_ia, a_ia, t_ia = self.get_rate_ia()

            #For use with agetracers 
            if ageBins is not None:
                mask = np.logical_and(ageBins[0] <= a_ia, a_ia <= ageBins[1])
                i_ia = integrate.trapz(r_ia[mask]/len(r_ia[mask]), x = [ageBins[0], ageBins[1]])#, a_ia[mask])
                return a_ia[1:], i_ia


            print("Shouldn't be here for Agetracers")
            i_ia = integrate.cumtrapz(r_ia, a_ia)
            return a_ia[1:], i_ia

        if self.source == 'cc':
            r_cc, a_cc, t_cc = self.get_rate_cc()

            #For use with agetracers 
            if ageBins is not None:
                mask = np.logical_and(ageBins[0] <= a_cc, a_cc <= ageBins[1])
                i_cc = integrate.trapz(r_cc[mask]/len(r_cc[mask]), x = [ageBins[0], ageBins[1]])#, a_cc[mask])
                return a_cc[1:], i_cc

            i_cc = integrate.cumtrapz(r_cc, a_cc)
            return a_cc[1:], i_cc

class analysis:
    def __init__(self, 
    z = 0):
        self.dirs = get_simulation_directory('stampede2')
        self.redshift = z

    def get_metrics(self, directory):
        metrics = {}

        return metrics
