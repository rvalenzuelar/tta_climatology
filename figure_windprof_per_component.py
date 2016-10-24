# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:54:04 2016

@author: raul
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse_data
import matplotlib.gridspec as gridspec
import string
import axis_builder as axb
from matplotlib import rcParams
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import interp1d
from rv_utilities import discrete_cmap

# if seaborn-style plot shows up need
# to use:
# %matplotlib inline

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['mathtext.default'] = 'sf'


def sin(arg):
    return np.sin(np.radians(arg))


def cos(arg):
    return np.cos(np.radians(arg))


# years = [1998]
years = [1998] + range(2001, 2013)

# max_hgt_gate = 15  # 1450 [m]
# max_hgt_gate = 21  # 2000 [m]
max_hgt_gate = 40  # 3750 [m] max top


try:
    WS
except NameError:

    WD = [pd.Series(), pd.Series()]
    WS = [pd.Series(), pd.Series()]

    for year in years:
        wpr = parse_data.windprof(year=year)
        bby = parse_data.surface('bby', year=year)
        czd = parse_data.surface('czd', year=year)

        hgt = wpr.hgt

        ' find common time period '
        first_bby = bby.dframe.index[0]
        first_czd = czd.dframe.index[0]
        first_wpr = wpr.dframe.index[0]
        last_bby = bby.dframe.index[-1]
        last_czd = czd.dframe.index[-1]
        last_wpr = wpr.dframe.index[-1]
        first = max(first_bby, first_czd, first_wpr)
        last = min(last_bby, last_czd, last_wpr)

        ' reduce time interval so all start and end at same time '
        wpr = wpr.dframe.loc[first:last]
        bby = bby.dframe.loc[first:last]
        czd = czd.dframe.loc[first:last]

        ' append surface values to windprof to make entire profile '
        surf_wsp = iter(bby.wspd.values.tolist())
        surf_wdr = iter(bby.wdir.values.tolist())
        wsp = wpr.wspd.map(lambda x: [surf_wsp.next()] + x)
        wdr = wpr.wdir.map(lambda x: [surf_wdr.next()] + x)

        ' check nans on precip '
        precip = pd.concat([bby.precip, czd.precip], axis=1)
        precip_nans = precip.apply(lambda x: x.isnull().any(),
                                   axis=1, reduce=True)
        precip_nans.name = 'precip_nan'
        tx = 'year:{}, any_precip_nan:{:4d}'
        print(tx.format(year, precip_nans.sum()))

        ' check entire profile nans ( same for ws and wd)'
        prof_nans = wsp.apply(lambda x: np.isnan(x).all())
        prof_nans.name = 'prof_nan'

        ' include only hours when surf and the entire' \
        ' profile is non-missing (profile is allowed to have' \
        ' at least one non-missing)'
        nan_df = pd.concat([precip_nans, prof_nans], axis=1)
        any_nan = nan_df.apply(lambda x: x.any(), axis=1, reduce=True)
        include = ~any_nan

        ' rainy days at CZD '
        rain_czd = czd.precip > 0
        rain_dates = rain_czd.loc[rain_czd.values].index

        ' reduce and save to big Series '
        wdr = wdr[include]
        wsp = wsp[include]
        wdr_rain = wdr[rain_czd]
        wsp_rain = wsp[rain_czd]
        WD[0] = WD[0].append(wdr)
        WS[0] = WS[0].append(wsp)
        WD[1] = WD[1].append(wdr_rain)
        WS[1] = WS[1].append(wsp_rain)

''' component analysis '''
wind_mean = [dict(), dict()]
wind_std = [dict(), dict()]
wind_spd = {0: [], 1: []}
wind_dir = {0: [], 1: []}
wind_spd_std = {0: [], 1: []}
wind_dir_std = {0: [], 1: []}

wind_dir_p05 = {0: [], 1: []}
wind_dir_p95 = {0: [], 1: []}

nans_per_level = [np.array([]).astype(int),
                  np.array([]).astype(int)]

U_stack2 = dict()
V_stack2 = dict()

' for each all/rainy category '
for n in range(2):
    sin_WD = WD[n].apply(lambda x: sin(x))
    cos_WD = WD[n].apply(lambda x: cos(x))

    U_df = -1 * WS[n].multiply(sin_WD)
    V_df = -1 * WS[n].multiply(cos_WD)

    U_stack = np.array(U_df.tolist())
    V_stack = np.array(V_df.tolist())

    if n == 0:
        m = 'all'
    else:
        m = 'rain'
    U_stack2[m] = U_stack
    V_stack2[m] = V_stack

    ' statistics per level '
    wind_mean[n][90] = np.nanmean(U_stack, axis=0)
    wind_std[n][90] = np.nanstd(U_stack, axis=0)
    wind_mean[n][180] = np.nanmean(V_stack, axis=0)
    wind_std[n][180] = np.nanstd(V_stack, axis=0)

    nans_per_level[n] = np.append(nans_per_level[n],
                                  np.isnan(V_stack).sum(axis=0))

cmap = discrete_cmap(7, base_cmap='Set1')
colors = [cmap(0), cmap(1), (0, 0, 0)]

dz = np.array([160] + [92] * (max_hgt_gate - 1))

y = np.array([0])
y = np.append(y, hgt[:max_hgt_gate])
ydz = y[:-1] + (y[1:] - y[:-1]) / 2.

" percentage of obs for each category "
wind_mean[0][0] = 100 * (1 - (nans_per_level[0] / float(WD[0].index.size)))
wind_mean[1][0] = 100 * (1 - (nans_per_level[1] / float(WD[1].index.size)))
wind_std[0][0] = 100 * (1 - (nans_per_level[0] / float(WD[0].index.size)))
wind_std[1][0] = 100 * (1 - (nans_per_level[1] / float(WD[1].index.size)))

out = 'mean-wind'

if out == 'mean-comp':
    fig, axs = axb.specs(rows=2, cols=2,
                         id_panels=True,
                         hide_xlabels_in=[0, 1],
                         hide_ylabels_in=[1, 3],
                         hspace=0.07,
                         wspace=0.1,
                         left=0.15,
                         right=0.9,
                         show_grid=False,
                         figsize=(7, 8))
elif out == 'shear':
    fig, axs = axb.specs(rows=2, cols=3,
                         id_panels=True,
                         hide_xlabels_in=[0, 1, 2],
                         hide_ylabels_in=[1, 2, 4, 5],
                         hspace=0.07,
                         wspace=0.15,
                         left=0.1,
                         right=0.95,
                         show_grid=False,
                         figsize=(9, 8))

elif out == 'distr-wind':
    fig = plt.figure(figsize=(7, 8))
elif out == 'mean-wind':
    fig,axs = axb.specs(rows=2, cols=3,
                        id_panels=True,
                        col_ratio=[2,2,1],
                        hide_xlabels_in=[0,1,2],
                        hide_ylabels_in=[1,2,4,5],
                        hspace=0.07,
                        wspace=0.1,
                        left=0.1,
                        right=0.95,
                        show_grid=False,
                        figsize=(9, 8))
elif out == 'shear-mod':
    fig = plt.figure(figsize=(6, 8))

lw = 3

if out == 'mean-comp':

    for n, ax in enumerate(axs):

        if n == 0:
            group, comp = [0,90]
            cl = colors[0]
        elif n == 1:
            group, comp = [0,180]
            cl = colors[1]
        elif n == 2:
            group, comp = [1,90]
            cl = colors[0]
        elif n == 3:
            group, comp = [1, 180]
            cl = colors[1]

        x = wind_mean[group][comp]
        std = wind_std[group][comp]
        ' plot mean '
        ax.plot(x, y, color=cl, lw=lw)
        ' plot std_dev'
        ax.fill_betweenx(y, x - std,
                         x2=x + std,
                         where=x - std < x + std,
                         color=cl,
                         alpha=0.2)

        ax.set_xlim([-10, 20])
        ax.set_ylim([0, 3000])

        if n in [2,3]:
            ax.set_xlabel('$[m\,s^{-1}]$')
        anotU, anotV = ['U', 'V']

        if n == 0:
            ax.set_ylabel('Altitude [m] MSL')

        " vertical line "
        ax.vlines(0, 0, 3000, color=(0.4, 0.4, 0.4),
                  lw=lw, linestyle='--')

        anot_grid_pos = [0, 1, 1, 3]
        panel_name_loc = 0.8

elif out == 'shear':

    for n, ax in enumerate(axs):

        if n == 0:
            x = wind_mean[0][90]
            dx = x[1:] - x[:-1]
            dxdz = dx / dz
        elif n == 1:
            x = wind_mean[0][180]
            dx = x[1:] - x[:-1]
            dxdz = dx / dz
        elif n == 2:
            x = wind_mean[0][90]
            dx = x[1:] - x[:-1]
            dx1dz = dx / dz
            x = wind_mean[0][180]
            dx = x[1:] - x[:-1]
            dx2dz = dx / dz
            dxdz = np.sqrt(dx1dz**2+dx2dz**2)
        elif n == 3:
            x = wind_mean[1][90]
            dx = x[1:] - x[:-1]
            dxdz = dx / dz
        elif n == 4:
            x = wind_mean[1][180]
            dx = x[1:] - x[:-1]
            dxdz = dx / dz
        elif n == 5:
            x = wind_mean[1][90]
            dx = x[1:] - x[:-1]
            dx1dz = dx / dz
            x = wind_mean[1][180]
            dx = x[1:] - x[:-1]
            dx2dz = dx / dz
            dxdz = np.sqrt(dx1dz**2+dx2dz**2)

        if n in [0, 3]:
            cl = cmap(0)
        elif n in [1, 4]:
            cl = cmap(1)
        else:
            cl = (0, 0, 0)

        ynew = np.linspace(ydz.min(), int(ydz.max()), 500)
        s = Spline(ydz, dxdz)
        xnew = s(ynew)
        ax.plot(xnew * 1e3, ynew, color=cl, lw=lw)
        # ax.plot(dxdz*1e3, ydz, color=cl, lw=lw)
        ax.set_xlim([-5, 20])
        ax.set_ylim([0, 3000])
        if n in [3, 4, 5]:
            ax.set_xlabel('$[x10^{-3}\,s^{-1}]$')
        anotU, anotV = ['$dU/dZ$', '$dV/dZ$']
        anot_grid_pos = [0, 1, 2, 5]

        if n == 2:
            tx = '$\sqrt{(dU/dZ)^{2}+(dV/dZ)^{2}}$'
            ax.text(0.5, 1.03, tx, transform=ax.transAxes,
                    ha='center')

        if n == 0:
            ax.set_ylabel('Altitude [m] MSL')


elif out == 'distr-wind':

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    axs = [ax1, ax2, ax3, ax4]

    axes[0, 2].remove()
    axes[1, 2].remove()

    WD_stack_all = np.array(WD[0].tolist())
    WS_stack_all = np.array(WS[0].tolist())
    WD_stack_rai = np.array(WD[1].tolist())
    WS_stack_rai = np.array(WS[1].tolist())

    df1 = pd.DataFrame(data=WS_stack_all)
    df2 = pd.DataFrame(data=WD_stack_all)
    df3 = pd.DataFrame(data=WS_stack_rai)
    df4 = pd.DataFrame(data=WD_stack_rai)

    whis = [5, 95]
    dfs = [df1, df2, df3, df4]
    axs = [ax1, ax2, ax3, ax4]
    for df, ax in zip(dfs, axs):
        box = df.boxplot(vert=False, grid=False, whis=whis,
                         showfliers=False,
                         showbox=False,
                         whiskerprops={'linestyle': '-'},
                         medianprops={'linestyle': '-'},
                         ax=ax)

        whisks = box['whiskers']
        w1 = whisks[0::2]
        w2 = whisks[1::2]
        for l, r in zip(w1, w2):
            ldata = l.get_xdata()
            rdata = r.get_xdata()
            x = [ldata[1], rdata[1]]
            y = l.get_ydata()
            ax.plot(x, y, color='b')

    ax1.set_xlim([-1, 28])
    ax3.set_xlim([-1, 28])

    ax2.set_xticks(range(0, 400, 60))
    ax4.set_xticks(range(0, 400, 60))
    ax2.set_xlim([-10, 370])
    ax4.set_xlim([-10, 370])

    anotU, anotV = ['speed', 'direction']
    anrows = [0, 0, 0, 1]
    ancols = [0, 1, 1, 1]

    mode = df1.mode(axis=0).max().values
    ax1.scatter(mode, range(1, 42))

    mode = df2.mode(axis=0).max().values
    ax2.scatter(mode, range(1, 42))

    mode = df3.mode(axis=0).max().values
    ax3.scatter(mode, range(1, 42))

    mode = df4.mode(axis=0).max().values
    ax4.scatter(mode, range(1, 42))

    f = interp1d(hgt.tolist(), range(1, 41))
    for ax in axs:
        ax.grid(False)
        newticks = [f(160)] + f(np.arange(570, 3500, 500)).tolist()
        ax.set_yticks(newticks)
        ax.set_yticklabels(np.arange(0, 3500, 500), fontsize=15)
        ax.set_ylim([0, f(3070)])

    ax1.set_xticklabels('')
    ax2.set_xticklabels('')
    ax2.set_yticklabels('')
    ax4.set_yticklabels('')

    ax3.set_xlabel('$[m\,s^{-1}]$')
    ax4.set_xlabel('$[degrees]$')
    ax1.set_ylabel('Altitude [m] MSL')

    panel_name_loc = 0.1


elif out == 'mean-wind':

    import wind_weber as wb

    for n, ax in enumerate(axs):

        if n in [0, 1, 3, 4]:
            if n in [1, 4]:
                mean = list()
                std = list()
                cl = cmap(1)
                if n == 1:
                    u = U_stack2['all']
                    v = V_stack2['all']
                else:
                    u = U_stack2['rain']
                    v = V_stack2['rain']

                for i in range(41):
                    mean.append(wb.angular_mean(u[:, i],v[:, i]))
                    std.append(wb.angular_stddev2(u[:, i], v[:, i]))

                x = np.array(mean)
                std = np.array(std)
            elif n in [0, 3]:
                cl = cmap(0)
                if n == 0:
                    spd = np.array(WS[0].tolist())
                    x = np.nanmean(spd,axis=0)
                    std = np.nanstd(spd,axis=0)
                else:
                    spd = np.array(WS[1].tolist())
                    x = np.nanmean(spd,axis=0)
                    std = np.nanstd(spd,axis=0)

            ' plot mean '
            ax.plot(x, y, color=cl, lw=lw)
            ' plot std_dev'
            ax.fill_betweenx(y, x - std,
                             x2=x + std,
                             where=x - std < x + std,
                             color=cl,
                             alpha=0.2)

            ax.set_ylim([0, 3000])


        else:
            if n == 2:
                x = wind_mean[0][0]
            else:
                x = wind_mean[1][0]
            ax.plot(x, y, color='k', lw=lw)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 3000])
            ax.set_xticklabels('')

    labels = ['0', '', '40', '', '80', '']
    axs[5].set_xlabel('[%]')
    axs[5].set_xticklabels(labels)

    for n in [1, 4]:
        axs[n].axvline(360, 0, 3000,
                        color=(0.3, 0.3, 0.3),
                        linestyle='--',
                        lw=2)

    for n in [0, 3]:
        axs[n].set_xlim([-1, 25])

    for n in [1, 4]:
        axs[n].set_xticks(range(60, 400, 60))
        axs[n].set_xlim([50, 410])

    anotU, anotV = ['speed', 'direction']
    anot_grid_pos = [0, 1, 2, 5]
    axs[3].set_xlabel('$[m\,s^{-1}]$')
    axs[4].set_xlabel('$[degrees]$')
    axs[0].set_ylabel('Altitude [m] MSL')

    panel_name_loc = 0.1


elif out == 'shear-mod':

    axs = axes.flatten()

    u_all = wind_mean[0][90]
    v_all = wind_mean[0][180]
    u_rai = wind_mean[1][90]
    v_rai = wind_mean[1][180]

    du = u_all[1:] - u_all[:-1]
    dudz = du / dz

    dv = v_all[1:] - v_all[:-1]
    dvdz = dv / dz

    ynew = np.linspace(ydz.min(), int(ydz.max()), 500)
    s = Spline(ydz, dudz)
    dudz_new = s(ynew)

    s = Spline(ydz, dvdz)
    dvdz_new = s(ynew)

    shear = np.sqrt(dudz_new ** 2 + dvdz_new ** 2)
    axs[0].plot(shear * 1e3, ynew, color='k', lw=lw)
    axs[0].set_xlim([0, 20])

    du = u_rai[1:] - u_rai[:-1]
    dudz = du / dz

    dv = v_rai[1:] - v_rai[:-1]
    dvdz = dv / dz

    ynew = np.linspace(ydz.min(), int(ydz.max()), 500)
    s = Spline(ydz, dudz)
    dudz_new = s(ynew)

    s = Spline(ydz, dvdz)
    dvdz_new = s(ynew)

    shear = np.sqrt(dudz_new ** 2 + dvdz_new ** 2)
    axs[3].plot(shear * 1e3, ynew, color='k', lw=lw)
    axs[3].set_xlim([0, 20])

    axs[1].remove()
    axs[2].remove()
    axs[4].remove()
    axs[5].remove()

    axs[0].set_xticklabels('')

    axs[3].set_xlabel('$[x10^{-3}\,s^{-1}]$')
    axs[0].set_ylabel('Altitude [m] MSL')
    axs[0].grid(True)
    axs[0].set_ylim([0, 3000])
    axs[3].grid(True)
    axs[3].set_ylim([0, 3000])

    anotU, anotV = ['', '']
    anrows = [0, 0, 0, 1]
    ancols = [0, 1, 0, 0]

    panel_name_loc = 0.8

''' axis annotation '''
locx = [0.5, 0.5, 1.1, 1.1]
locy = [1.05, 1.05, 0.5, 0.5]
anot = [anotU,
        anotV,
        'All (n={:d})'.format(WD[0].index.size),
        'czd-rain (n={:d})'.format(WD[1].index.size)]
rot = [0, 0, -90, -90]
for gpos, n in zip(anot_grid_pos, range(4)):
    # ax = axs[gpos]
    axs[gpos].text(locx[n],
            locy[n],
            anot[n],
            fontsize=14,
            ha='center',
            va='center',
            weight='bold',
            rotation=rot[n],
            transform=axs[gpos].transAxes)


if out == 'mean-comp':
    tx = '13-season mean and std_dev wind component profile'

elif out == 'shear':
    tx = '13-season mean vertical wind-shear profile'

elif out == 'distr-wind':
    tx = '13-season wind distribution profile'
    plt.subplots_adjust(hspace=0.1, wspace=0.15,
                        right=1.15)
elif out == 'mean-wind':
    tx = '13-season mean and std_dev wind profile'

elif out == 'shear-mod':
    tx = '13-season vertical wind shear profile'


plt.suptitle(tx, fontsize=15, weight='bold', y=0.98)

# plt.show()

place = '/Users/raulvalenzuela/Documents/'
# place = '/home/raul/Desktop/'
fname = 'windprof_components_{}.png'.format(out)
plt.savefig(place+fname, dpi=300, format='png',papertype='letter',
           bbox_inches='tight')
