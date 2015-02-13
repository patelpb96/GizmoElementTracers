'''
Analysis of gizmo/gadget simulations.

Masses in {M_sun}, positions in {kpc comoving}, distances & radii in {kpc physical}.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# local ----
from utilities import utility as ut
from utilities import particle
from utilities import plot


#===================================================================================================
# tests
#===================================================================================================
def test_contamination(
    part, center_pos=[], distance_lim=[10, 3000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', vir_radius=None, scale_vir=False,
    center_species=['star'], write_plot=False, plot_directory='.'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
        note: if not input, generate
    distance limits: list or array
    distance scaling: float
        options: log, lin
    '''
    species_test = ['dark.2', 'dark.3', 'dark.4', 'gas', 'star']

    species_ref = 'dark'

    Say = ut.io.SayClass(test_contamination)

    species_test_t = []
    for spec_test in species_test:
        if spec_test in part:
            species_test_t.append(spec_test)
        else:
            Say.say('! no %s in particle dictionary' % spec_test)
    species_test = species_test_t

    if center_pos is None or not len(center_pos):
        center_pos = particle.get_center_position_species(part, center_species)

    x_lim = np.array(distance_lim)
    if vir_radius and scale_vir:
        x_lim *= vir_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    pros = {species_ref: {}}
    for spec in species_test:
        pros[spec] = {}

    ratios = {}

    for spec in pros:
        dists = ut.coord.distance('scalar', part[spec]['position'], center_pos,
                                  part.info['box.length'])
        pros[spec] = DistanceBin.get_mass_profile(dists, part[spec]['mass'], get_spline=False)

    for spec in species_test:
        Say.say(spec)
        mass_ratio_bin = pros[spec]['mass'] / pros[species_ref]['mass']
        mass_ratio_cum = pros[spec]['mass.cum'] / pros[species_ref]['mass.cum']
        ratios[spec] = {'bin': mass_ratio_bin, 'cum': mass_ratio_cum}
        """
        for dist_bin_i in xrange(DistanceBin.num):
            dist_bin_lim = DistanceBin.get_bin_limit('lin', dist_bin_i)
            Say.say('dist = [%.3f, %.3f]: mass ratio (bin, cum) = (%.5f, %.5f)' %
                    (dist_bin_lim[0], dist_bin_lim[1],
                     mass_ratio_bin[dist_bin_i], mass_ratio_cum[dist_bin_i]))
            if mass_ratio_bin[dist_bin_i] >= 1.0:
                break
        """

    # plot ----------
    colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if vir_radius and scale_vir:
        xs /= vir_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    #subplot.set_ylim([0, 0.1])
    subplot.set_ylim([0.001, 3])
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    subplot.set_ylabel('$M_{\\rm spec} / M_{\\rm %s}$' % species_ref, fontsize=20)
    if scale_vir:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if vir_radius:
        if scale_vir:
            x_ref = 1
        else:
            x_ref = vir_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    #import ipdb; ipdb.set_trace()

    for spec_i, spec in enumerate(species_test):
        plot_func(xs, ratios[spec]['bin'], color=colors[spec_i], alpha=0.6, label=spec)

    legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    legend.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_safe_path(plot_directory)
        dist_name = 'dist'
        if vir_radius and scale_vir:
            dist_name += '.200m'
        plot_name = 'mass.ratio_v_%s_z.%.1f.pdf' % (dist_name, part.snap['redshift'])
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


def test_metal_v_distance(
    part, center_pos=[], distance_lim=[10, 3000], distance_bin_num=100,
    distance_scaling='log', y_scaling='log', vir_radius=None, scale_vir=False,
    center_species=['star'],
    plot_kind='metalicity', write_plot=False, plot_directory='.'):
    '''
    Test lower resolution particle contamination around center.

    Parameters
    ----------
    catalog of particles: dict
    position of galaxy center: array
        note: if not input, generate
    distance limits: list or array
    distance scaling: float
        options: log, lin
    '''
    metal_index = 0    # overall metalicity

    Say = ut.io.SayClass(test_metal_v_distance)

    if center_pos is None or not len(center_pos):
        center_pos = particle.get_center_position_species(part, center_species)

    x_lim = np.array(distance_lim)
    if vir_radius and scale_vir:
        x_lim *= vir_radius

    DistanceBin = ut.bin.DistanceBinClass(distance_scaling, x_lim, distance_bin_num)

    dists = ut.coord.distance('scalar', part['gas']['position'], center_pos,
                              part.info['box.length'])
    metal_masses = part['gas']['metal'][:, metal_index] * part['gas']['mass'] / 0.02    # solar

    pro_metal = DistanceBin.get_mass_profile(dists, metal_masses, get_spline=False)
    if plot_kind == 'metalicity':
        pro_mass = DistanceBin.get_mass_profile(dists, part['gas']['mass'], get_spline=False)
        ys = pro_metal['mass'] / pro_mass['mass']
        y_lim = np.clip(plot.get_limits(ys), 0.0001, 10)
    elif plot_kind == 'metal.mass':
        ys = pro_metal['frac.cum']
        y_lim = [0.001, 1]

    # plot ----------
    #colors = plot.get_colors(len(species_test), use_black=False)
    xs = DistanceBin.mids
    if vir_radius and scale_vir:
        xs /= vir_radius

    plt.close()
    plt.minorticks_on()
    fig, subplot = plt.subplots(1, 1, sharex=True)
    subplot.set_xlim(distance_lim)
    #subplot.set_ylim([0, 0.1])
    subplot.set_ylim(y_lim)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.14, hspace=0.03)

    if plot_kind == 'metalicity':
        subplot.set_ylabel('$Z \, / \, Z_\odot$', fontsize=20)
    elif plot_kind == 'metal.mass':
        subplot.set_ylabel('$M_{\\rm Z}(< r) \, / \, M_{\\rm Z,tot}$', fontsize=20)
    if scale_vir:
        x_label = '$d \, / \, R_{\\rm 200m}$'
    else:
        x_label = 'distance [$\\rm kpc\,comoving$]'
    subplot.set_xlabel(x_label, fontsize=20)

    plot_func = plot.get_plot_function(subplot, distance_scaling, y_scaling)

    if vir_radius:
        if scale_vir:
            x_ref = 1
        else:
            x_ref = vir_radius
        plot_func([x_ref, x_ref], [1e-6, 1e6], color='black', linestyle=':', alpha=0.6)

    #import ipdb; ipdb.set_trace()

    plot_func(xs, ys, color='blue', alpha=0.6)

    #legend = subplot.legend(loc='best', prop=FontProperties(size=12))
    #legend.get_frame().set_alpha(0.7)

    #plt.tight_layout(pad=0.02)

    if write_plot:
        plot_directory = ut.io.get_safe_path(plot_directory)
        dist_name = 'dist'
        if vir_radius and scale_vir:
            dist_name += '.200m'
        plot_name = plot_kind + '_v_' + dist_name + '_z.%.1f.pdf' % part.info['redshift']
        plt.savefig(plot_directory + plot_name, format='pdf')
        Say.say('wrote %s' % plot_directory + plot_name)
    else:
        plt.show(block=False)


#===================================================================================================
# analysis
#===================================================================================================
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
