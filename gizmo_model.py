'''
Accomplishes the same as gizmo_star.py, but with added generality and the (eventual) ability to perturb the main models 
for use with the agetracers. 

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

import numpy as np
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt

import utilities as ut

global Z_0
Z_0 = 1

feedback_type = ['wind', 'ia', 'mannucci', 'maoz', 'cc']

#timespan = np.logspace(0, 4.1367, 3000)

# Yield Dictionaries for feedback types
element_yield_wind = {'He' : 0.36, 'C' : 0.016, 'N' : 0.0041, 'O' : 0.0118, 'Ne' : 0,
            'Mg' : 0, 'Si' : 0, 'S' : 0, 'Ca' : 0, 'Fe' : 0}
element_yield_cc = {'Z' : 0.19, 'He' : 0.369, 'C' : 0.0127, 'N' : 0.00456, 'O' : 0.111, 'Ne' : 0.0381,
            'Mg' : 0.00940, 'Si' : 0.00889, 'S' : 0.00378, 'Ca' : 0.000436, 'Fe' : 0.00706}
element_yield_mannucci = {'Z' : 1.0, 'He' : 0.0, 'C' : 0.035, 'N' : 8.57e-7, 'O' : 0.102, 'Ne' : 0.00321,
            'Mg' : 0.00614, 'Si' : 0.111, 'S' : 0.0621, 'Ca' : 0.00857, 'Fe' : 0.531}
element_yield_ia = {'Z' : 1.0, 'He' : 0.0, 'C' : 0.035, 'N' : 8.57e-7, 'O' : 0.102, 'Ne' : 0.00321,
            'Mg' : 0.00614, 'Si' : 0.111, 'S' : 0.0621, 'Ca' : 0.00857, 'Fe' : 0.531}

# Ejecta Masses for Each Event 
ejecta_masses = {'wind' : 1,
                 'ia': 1.4,
                 'cc' : 10.5} # corresponding directly to the above


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
    
    cc = element_yields('cc')

class feedback:

    def __init__(self, source = 'any', element_name = False, t_w = [1.0, 3.5, 100], t_cc = [3.4, 10.37, 37.53], t_ia = [37.53]):

        '''
        Init allows a user to specify a source (wind, ccsn, ia, etc.), specify a particular element name, and modify the transition ages (t_w, t_cc, t_ia) of feedback rates.
        The FIRE-2 defaults (mannucci) are implemented natively, but can be modified with input lists.
        '''
        self.source = source
        self.element = element_name
        self.timespan = np.logspace(0, 4.1367, 3000)

        self.trans_w = np.array(t_w) # transition age of winds
        self.trans_ia = np.array(t_ia) # transition age of SNe Ia
        self.trans_cc = np.array(t_cc) # transition age of CCSNe

    def get_rate_wind(self, Z = Z_0, massloss = True, metal_mass_fraction = None,  plot = False):

        '''
        Returns the rates versus stellar age for stellar winds in FIRE-2. 
        '''
    
        transition_ages = self.trans_w

        # Imposed mins and maxes based on FIRE-2 and FIRE-3. For stability or something
        metallicity_min = 0.01
        metallicity_max = 3
        age_min = 0  # Myr
        age_max = 13700

        # MODEL BELOW

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
            print("Selected " + str(self.element) + " yields for " + str(self.source))
            print(element_yields(self.source)[self.element])

            if plot:
                plt.loglog(a_wind, element_yields(self.source)[self.element]*r_wind)
            
            return element_yields(self.source)[self.element]*r_wind, a_wind, transition_ages

        if plot:
            plt.loglog(a_wind, r_wind)

        return r_wind, a_wind, transition_ages

    def get_rate_cc(self, Z = Z_0, massloss = True, metal_mass_fraction = None, plot = False):
    
        transition_ages = np.array([3.4, 10.37, 37.53])
    
        mask1 = [True if 0 < i <= transition_ages[0] else False for i in self.timespan]
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

            if self.element:
                print("Selected " + str(self.element) + " yields for " + str(self.source))
                print(element_yields(self.source)[self.element])

                if plot:
                    plt.loglog(a_cc, element_yields(self.source)[self.element]*r_cc)

                return element_yields(self.source)[self.element]*r_cc, a_cc, transition_ages

        if self.element:
            print("Selected " + str(self.element) + " yields for " + str(self.source))
            print(element_yields(self.source)[self.element])

            return element_yields(self.source)[self.element]*r_cc, a_cc, transition_ages

        if plot:
            plt.loglog(a_cc, r_cc)

        return r_cc, a_cc, transition_ages

    def get_rate_ia(self, Z = Z_0, massloss = True, metal_mass_fraction = None, plot = False, type = "mannucci"):

        transition_ages = self.trans_ia

        mask1 = [True if 0 < i <= transition_ages[0] else False for i in self.timespan]
        mask2 = [True if transition_ages[0] <= i else False for i in self.timespan]

        func1 = 0*(self.timespan[mask1]/self.timespan[mask1])
        func2 = 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((self.timespan[mask2] - 50) / 10) ** 2)

        r_ia = np.array([*func1, *func2], dtype = 'object') # y-axis: rate
        a_ia = np.array([*self.timespan[mask1], *self.timespan[mask2]], dtype = 'object') # x-axis: age

        if massloss == True:
            r_ia = ejecta_masses[self.source]*r_ia

            if self.element:
                print("Selected " + str(self.element) + " yields for " + str(self.source))
                print(element_yields(self.source)[self.element])

                if plot:
                    plt.loglog(a_ia, element_yields(self.source)[self.element]*r_ia)
                    plt.show()

                return element_yields(self.source)[self.element]*r_ia, a_ia, transition_ages

            if plot:
                plt.loglog(a_ia, r_ia)
                plt.show()

            return r_ia, a_ia, transition_ages

        if plot:
            plt.loglog(a_ia, r_ia)
            plt.show()

        return r_ia, a_ia, transition_ages

    def integrate_massloss(self, ages, Z = Z_0, massloss = True, metal_mass_fraction = None, source = 'wind', element_name = False):
        elem = element_name

        a,b,c = self.eval("get_rate_" + str(source) + "(self.timespan, Z = Z_0, element_name = " + str(elem) + ")")
        d = integrate.cumtrapz(a, b)

        return b, d