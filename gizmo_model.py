'''
Accomplishes the same as gizmo_star.py, but with added generality and the (eventual) ability to modulate the main models 
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

feedback_type = ['wind', 'mannucci', 'maoz', 'cc']
timespan = np.logspace(0, 4.1367, 3000)

# Yield Dictionaries for feedback types
element_yield_wind = {'He' : 0.36, 'C' : 0.016, 'N' : 0.0041, 'O' : 0.0118}
element_yield_cc = {'Z' : 0.19, 'He' : 0.369, 'C' : 0.0127, 'N' : 0.00456, 'O' : 0.111, 'Ne' : 0.0381,
            'Mg' : 0.00940, 'Si' : 0.00889, 'S' : 0.00378, 'Ca' : 0.000436, 'Fe' : 0.00706}
element_yield_mannucci = {'Z' : 1.0, 'He' : 0.0, 'C' : 0.035, 'N' : 8.57e-7, 'O' : 0.102, 'Ne' : 0.00321,
            'Mg' : 0.00614, 'Si' : 0.111, 'S' : 0.0621, 'Ca' : 0.00857, 'Fe' : 0.531}
element_yield_maoz = {'Z' : 1.0, 'He' : 0.0, 'C' : 0.035, 'N' : 8.57e-7, 'O' : 0.102, 'Ne' : 0.00321,
            'Mg' : 0.00614, 'Si' : 0.111, 'S' : 0.0621, 'Ca' : 0.00857, 'Fe' : 0.531}

# Ejecta Masses for Each Event 
ejecta_masses = {'wind' : 1,
                 'mannucci': 1.4,
                 'maoz' : 1.4,
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
        raise Exception("Source not Defined - i.e. - pass element_yields('cc') for CCSN")
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
        
    return globals()[string] #globals() converts the indexed string into a variable (or function, if needed). Works like eval()
    
    cc = element_yields('cc')

class feedback:

    Z_0 = 1 # solar metallicity 

    def get_rate_wind(ages = timespan, Z = Z_0, massloss = True, metal_mass_fraction = None, model = 'wind', element_name = False):

        '''
        Returns the rates versus stellar age for stellar winds in FIRE-2. 
        '''
    
        transition_ages = np.array([1.0, 3.5, 100])

        # Imposed mins and maxes based on FIRE-2 and FIRE-3. For stability or something
        metallicity_min = 0.01
        metallicity_max = 3
        age_min = 0  # Myr
        age_max = 13700

        # MODEL BELOW

        mask1 = [True if 0 < i <= transition_ages[0] else False for i in ages]
        mask2 = [True if transition_ages[0] <= i <= transition_ages[1] else False for i in ages]
        mask3 = [True if transition_ages[1] <= i <= transition_ages[2] else False for i in ages]
        mask4 = [True if transition_ages[2] <= i else False for i in ages]

        func1 = 4.76317 * Z * (ages[mask1]/ages[mask1]) # FIRE-2
        func1 = 4.76317 * (ages[mask1]/ages[mask1]) # FIRE-2.1
        func2 = 4.76317 * Z * ages[mask2] ** (1.838 * (0.79 + np.log10(Z)))
        func3 = 29.4 * (ages[mask3] / 3.5) ** -3.25 + 0.0041987
        func4 = 0.41987 * (ages[mask4] / 1e3) ** -1.1 / (12.9 - np.log(ages[mask4] / 1e3))

        # MODEL ABOVE

        r_wind = np.array([*func1, *func2, *func3, *func4], dtype = 'object')/1e3 # y-axis: rate
        a_wind = np.array([*timespan[mask1], *timespan[mask2], *timespan[mask3], *timespan[mask4]], dtype = 'object') # x-axis: age

        if element_name:
            print("Selected " + str(element_name) + " yields for " + str(model))
            print(element_yields(model)[element_name])
            return element_yields(model)[element_name]*r_wind, a_wind, transition_ages

        return r_wind, a_wind, transition_ages

    def get_rate_cc(ages, Z = Z_0, massloss = True, metal_mass_fraction = None, model = 'cc', element_name = False):
    
        transition_ages = np.array([3.4, 10.37, 37.53])
    
        mask1 = [True if 0 < i <= transition_ages[0] else False for i in ages]
        mask2 = [True if transition_ages[0] <= i <= transition_ages[1] else False for i in ages]
        mask3 = [True if transition_ages[1] <= i <= transition_ages[2] else False for i in ages]
        mask4 = [True if transition_ages[2] <= i else False for i in ages]

        func1 = 0*(ages[mask1]/ages[mask1])
        func2 = 5.408e-4*(ages[mask2]/ages[mask2])
        func3 = 2.516e-4*(ages[mask3]/ages[mask3])
        func4 = 0*(ages[mask4]/ages[mask4])

        r_cc = np.array([*func1, *func2, *func3, *func4], dtype = 'object') # y-axis: rate
        a_cc = np.array([*timespan[mask1], *timespan[mask2], *timespan[mask3], *timespan[mask4]], dtype = 'object') # x-axis: age

        if massloss == True:
            r_cc *= ejecta_masses[model]

            if element_name:
                print("Selected " + str(element_name) + " yields for " + str(model))
                print(element_yields(model)[element_name])
                return element_yields(model)[element_name]*r_cc, a_cc, transition_ages

        if element_name:
            print("Selected " + str(element_name) + " yields for " + str(model))
            print(element_yields(model)[element_name])
            return element_yields(model)[element_name]*r_cc, a_cc, transition_ages

        return r_cc, a_cc, transition_ages

    def get_rate_ia(ages, Z = Z_0, massloss = True, metal_mass_fraction = None, model = 'mannucci', element_name = False):

        transition_ages = np.array([37.53])

        mask1 = [True if 0 < i <= transition_ages[0] else False for i in ages]
        mask2 = [True if transition_ages[0] <= i else False for i in ages]

        func1 = 0*(ages[mask1]/ages[mask1])
        func2 = 5.3e-8 + 1.6e-5 * np.exp(-0.5 * ((ages[mask2] - 50) / 10) ** 2)

        r_ia = np.array([*func1, *func2], dtype = 'object') # y-axis: rate
        a_ia = np.array([*timespan[mask1], *timespan[mask2]], dtype = 'object') # x-axis: age

        if massloss == True:
            r_ia = ejecta_masses[model]*r_ia

            if element_name:
                print("Selected " + str(element_name) + " yields for " + str(model))
                print(element_yields(model)[element_name])
                return element_yields(model)[element_name]*r_ia, a_ia, transition_ages

            return r_ia, a_ia, transition_ages

        return r_ia, a_ia, transition_ages


    