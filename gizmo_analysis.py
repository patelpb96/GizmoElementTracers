'''
@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
# local ----
#from utilities import utility as ut


def get_center_position_zoom(part, cen_pos=[0, 0, 0], radius_max=1e10):
    '''
    .
    '''
    radius_bins = np.array([1e10, 4000, 2000, 1000, 500, 250, 125, 70, 50, 25, 10, 5, 2.5, 1, 0.5])
    radius_bins = radius_bins[radius_bins <= radius_max]
    poss = np.array(part['position'])
    cen_pos = np.array(cen_pos, dtype=np.float32)

    #import ipdb; ipdb.set_trace()

    for rbin_i in xrange(len(radius_bins)):
        poss -= cen_pos
        rads = np.sqrt(poss[:, 0] ** 2 + poss[:, 1] ** 2 + poss[:, 2] ** 2)
        masks = (rads < radius_bins[rbin_i])
        poss = poss[masks]
        if poss.shape[0] <= 1:
            break
        cen_pos += np.median(poss, 0)

    return cen_pos


def plot_sfr_history(part, pis=None, redshift_lim=[0, 1], aexp_wid=0.001, write_plot=False):
    '''
    .
    '''
    times, sfrs = get_sfr_history(part, pis, redshift_lim, aexp_wid)

    # plot ----------
    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.14, hspace=0.03)

    #subplot.xscale('linear')
    #subplot.yscale('log')
    subplot.set_xlabel(r'time [Gyr]')
    #pylab.ylabel(r'${\rm SFR}\ \ \dot{M}_{\ast}\ \  [{\rm M_{\odot}\,yr^{-1}}]$')
    subplot.set_ylabel(r'${\rm SFR}\,[{\rm M_{\odot}\,yr^{-1}}]$')
    subplot.semilogy(times, sfrs, linewidth=2.0, color='r')

    #plt.tight_layout(pad=0.02)
    if write_plot:
        plt.savefig('sfr_v_time.pdf', format='pdf')


def get_sfr_history(part, pis=None, redshift_lim=[0, 1], aexp_wid=0.001):
    '''
    .
    '''
    if pis is None:
        pis = np.arange(part['mass'].size, dtype=np.int32)
    pis_sort = np.argsort(part['form.time'])
    star_form_aexps = part['form.time'][pis_sort]
    #star_form_redshifts = 1 / star_form_aexps - 1
    star_masses = part['mass'][pis_sort]
    star_masses_cum = np.cumsum(star_masses)

    if redshift_lim:
        redshift_lim = np.array(redshift_lim)
        aexp_lim = np.sort(1 / (1 + redshift_lim))
    else:
        aexp_lim = [np.min(star_form_aexps), np.max(star_form_aexps)]

    aexp_bins = np.arange(aexp_lim.min(), aexp_lim.max(), aexp_wid)
    redshift_bins = 1 / aexp_bins - 1
    time_bins = part.Cosmo.age(redshift_bins)
    #time_bins = part.Cosmo.age(0) - time_bins
    time_bins *= 1e9    # {yr}

    star_mass_cum_bins = np.interp(aexp_bins, star_form_aexps, star_masses_cum)
    dm_dts = np.diff(star_mass_cum_bins) / np.diff(time_bins) / 0.7    # account for mass loss

    time_mids = time_bins[0: time_bins.size - 1] + np.diff(time_bins)    # midpoints

    time_mids /= 1e9

    return time_mids, dm_dts
