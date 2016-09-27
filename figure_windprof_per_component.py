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


def comp2wind(u, v):
    wdir = 270 - (np.arctan2(v,u)*180/np.pi)
    wdir[wdir > 360] -= 360
    wspd = np.sqrt(u**2+v**2)
    return wspd,wdir


def wind_stddev2(u,v):

    '''
        Ackermann (1983,JCAM)

        u and v are time series at individual level
    '''


    m = np.vstack((u, v))
    bad = np.sum(np.isnan(m), axis=0).astype(bool)
    ug = u[~bad]
    vg = v[~bad]

    N = ug.size

    " mean "
    U = ug.sum()/N
    V = vg.sum()/N

    " variance "
    sqr_u = (ug - U)**2
    sqr_v = (vg - V)**2
    var_u = sqr_u.sum()/N
    var_v = sqr_v.sum()/N

    " covariance "
    a = (ug-U)*(vg-V)
    covar_uv = a.sum()/N

    " mean wind speed and direction "
    S = np.sqrt(U**2 + V**2)
    D = 270 - np.arctan2(V,U)*180/np.pi

    S_std = np.sqrt(U**2*var_u + \
                       V**2*var_v + \
                       2*U*V*covar_uv)/S

    D_std = np.sqrt(V**2*var_u + \
                       U**2*var_v - \
                       2*U*V*covar_uv)/S**2

    return S, S_std, D, D_std



# years = [1998]
years = [1998]+range(2001,2013)

#max_hgt_gate = 15  # 1450 [m]
#max_hgt_gate = 21  # 2000 [m]
max_hgt_gate = 40  # 3750 [m] max top

try:
    WS
except NameError:

    WD = [pd.Series(),pd.Series()]
    WS = [pd.Series(),pd.Series()]

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
        first = max(first_bby,first_czd,first_wpr)   
        last  = min(last_bby,last_czd,last_wpr)

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
        precip = pd.concat([bby.precip,czd.precip],axis=1)
        precip_nans = precip.apply(lambda x: x.isnull().any(),
                                   axis=1,reduce=True)
        precip_nans.name='precip_nan'
        tx = 'year:{}, any_precip_nan:{:4d}'
        print(tx.format(year,precip_nans.sum()))

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

' for each all/rainy category '
for n in range(2):

    sin_WD = WD[n].apply(lambda x: sin(x))
    cos_WD = WD[n].apply(lambda x: cos(x))

    U_df = -1*WS[n].multiply(sin_WD)
    V_df = -1*WS[n].multiply(cos_WD)

    U_stack = np.array(U_df.tolist())
    V_stack = np.array(V_df.tolist())

    ' statistics per level '
    wind_mean[n][90] = np.nanmean(U_stack,axis=0)
    wind_std[n][90] = np.nanstd(U_stack,axis=0)
    wind_mean[n][180] = np.nanmean(V_stack,axis=0)
    wind_std[n][180] = np.nanstd(V_stack,axis=0)

    nans_per_level[n] = np.append(nans_per_level[n],
                                  np.isnan(V_stack).sum(axis=0))

cmap = discrete_cmap(7, base_cmap='Set1')
colors = [cmap(0), cmap(1), (0, 0, 0)]

dz = np.array([160]+[92]*(max_hgt_gate-1))

y = np.array([0])
y = np.append(y, hgt[:max_hgt_gate])
ydz = y[:-1]+(y[1:]-y[:-1])/2.


" percentage of obs for each category "
wind_mean[0][0] = 100*(1-(nans_per_level[0]/float(WD[0].index.size)))
wind_mean[1][0] = 100*(1-(nans_per_level[1]/float(WD[1].index.size)))
wind_std[0][0] = 100*(1-(nans_per_level[0]/float(WD[0].index.size)))
wind_std[1][0] = 100*(1-(nans_per_level[1]/float(WD[1].index.size)))

out = 'mean-wind'

if out == 'mean-comp':
    fig = plt.figure(figsize=(9,8))
elif out == 'shear':
    fig = plt.figure(figsize=(7,8))
elif out == 'mean-wind':
    fig = plt.figure(figsize=(7, 8))
elif out == 'shear-mod':
    fig = plt.figure(figsize=(6, 8))

gs = gridspec.GridSpec(2, 3,
                       width_ratios=[2, 2, 1],
                       )

axes = list()
axes.append(plt.subplot(gs[0]))
axes.append(plt.subplot(gs[1]))
axes.append(plt.subplot(gs[2]))
axes.append(plt.subplot(gs[3]))
axes.append(plt.subplot(gs[4]))
axes.append(plt.subplot(gs[5]))
axes = np.array(axes)
axes = axes.reshape((2,3))

lw = 3

if out in ['mean-comp','shear']:
    for row in range(2):
        for col,comp,cl in zip(range(3), [90, 180, 0], colors):

            ax = axes[row,col]
            x = wind_mean[row][comp]
            std = wind_std[row][comp]

            if out == 'mean-comp':
                if col < 2:
                    ' plot mean '
                    ax.plot(x, y, color=cl, lw=lw)
                    ' plot std_dev'
                    ax.fill_betweenx(y,x-std,
                                     x2=x+std,
                                     where=x-std<x+std,
                                     color=cl,
                                     alpha=0.2)

                    ax.set_xlim([-10, 20])
                    if row == 1 and (col == 0 or col == 1):
                        ax.set_xlabel('$[m\,s^{-1}]$')
                    anotU,anotV = ['U','V']
                else:
                    x = wind_flow_mean[row][comp]
                    ax.plot(x, y, color=cl, lw=lw)
                    ax.set_xlim([0, 100])
                    labels=['0','','40','','80','']
                    ax.set_xticklabels(labels)
                anrows = [0, 0, 0, 1]
                ancols = [0, 1, 2, 2]
            elif out == 'shear':
                dx = x[1:]-x[:-1]
                dxdz = dx/dz
                ynew = np.linspace(ydz.min(),int(ydz.max()),500)
                s = Spline(ydz, dxdz)
                xnew = s(ynew)
                ax.plot(xnew*1e3, ynew, color=cl, lw=lw)
                # ax.plot(dxdz*1e3, ydz, color=cl, lw=lw)
                ax.set_xlim([-5, 20])
                if row == 1 and (col == 0 or col == 1):
                    ax.set_xlabel('$[x1e^{-3}s^{-1}]$')
                anotU, anotV = ['dU/dZ', 'dV/dZ']
                anrows = [0, 0, 0, 1]
                ancols = [0, 1, 1, 1]

            if row == 0:
                ax.set_xticklabels('')

            if col > 0:
                ax.set_yticklabels('')

            if row == 0 and col == 0:
                ax.set_ylabel('Altitude [m] MSL')

            if row == 1 and col == 2:
                ax.set_xlabel('[%]')

            ax.grid(True)
            ax.set_ylim([0, 3000])

            if col < 2:
                ax.vlines(0,0,3000,color=(0.4,0.4,0.4),
                          lw=lw, linestyle='--')

            if out == 'shear':
                if (row == 0 or row == 1) and col == 2:
                    ax.remove()

    panel_name_loc = 0.8

elif out == 'mean-wind':

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    axs = [ax1,ax2,ax3,ax4]

    axes[0,2].remove()
    axes[1,2].remove()

    # wspd_all, wdir_all = wind_spd[0], wind_dir[0]
    # wspd_rai, wdir_rai = wind_spd[1], wind_dir[1]


    # u_all = wind_mean[0][90]
    # v_all = wind_mean[0][180]
    # u_rai = wind_mean[1][90]
    # v_rai = wind_mean[1][180]
    #
    # spd_all = np.sqrt(u_all**2+v_all**2)
    # dir_all = 270-np.arctan2(v_all, u_all)*180/np.pi
    # dir_all[dir_all > 360] -= 360
    #
    # spd_rai = np.sqrt(u_rai**2+v_rai**2)
    # dir_rai = 270-np.arctan2(v_rai, u_rai)*180/np.pi
    # dir_rai[dir_rai > 360] -= 360
    #
    #
    # wspd_all, wdir_all = wind_spd[0], wind_dir[0]
    # wspd_rai, wdir_rai = wind_spd[1], wind_dir[1]
    #
    # " plot mean values "
    # ax1.plot(wspd_all,y,color=colors[0],lw=lw)
    # ax2.plot(dir_all,y,color=colors[1],lw=lw)
    # ax3.plot(wspd_rai,y,color=colors[0],lw=lw)
    # ax4.plot(wdir_rai,y,color=colors[1],lw=lw)

    WD_stack_all = np.array(WD[0].tolist())
    WS_stack_all = np.array(WS[0].tolist())
    WD_stack_rai = np.array(WD[1].tolist())
    WS_stack_rai = np.array(WS[1].tolist())

    df1 = pd.DataFrame(data=WS_stack_all)
    df2 = pd.DataFrame(data=WD_stack_all)
    df3 = pd.DataFrame(data=WS_stack_rai)
    df4 = pd.DataFrame(data=WD_stack_rai)

    df1.boxplot(vert=False,grid=False,whis=0,
                showfliers=False,ax=ax1)
    df2.boxplot(vert=False,grid=False,whis=0,
                showfliers=False,ax=ax2)
    df3.boxplot(vert=False,grid=False,whis=0,
                showfliers=False,ax=ax3)
    df4.boxplot(vert=False,grid=False,whis=0,
                showfliers=False,ax=ax4)

    f = interp1d(hgt.tolist(), range(1,41))


    ax1.set_xlim([0,25])
    ax3.set_xlim([0,25])

    ax2.set_xticks(range(90,360,60))
    ax4.set_xticks(range(90,360,60))

    ax2.set_xlim([90,330])
    ax4.set_xlim([90,330])

    anotU, anotV = ['speed', 'direction']
    anrows = [0, 0, 0, 1]
    ancols = [0, 1, 1, 1]

    for ax in axs:
        ax.grid(True)
        newticks = [f(160)]+f(np.arange(570, 3500, 500)).tolist()
        ax.set_yticks(newticks)
        ax.set_yticklabels(np.arange(0, 3500, 500),fontsize=15)
        ax.set_ylim([0, f(3070)])

    ax1.set_xticklabels('')
    ax2.set_xticklabels('')
    ax2.set_yticklabels('')
    ax4.set_yticklabels('')

    ax3.set_xlabel('$[m\,s^{-1}]$')
    ax4.set_xlabel('$[degrees]$')
    ax1.set_ylabel('Altitude [m] MSL')

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

    shear = np.sqrt(dudz_new**2 + dvdz_new**2)
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

    shear = np.sqrt(dudz_new**2 + dvdz_new**2)
    axs[3].plot(shear * 1e3, ynew, color='k', lw=lw)
    axs[3].set_xlim([0, 20])


    axs[1].remove()
    axs[2].remove()
    axs[4].remove()
    axs[5].remove()

    axs[0].set_xticklabels('')

    axs[3].set_xlabel('$[x1e^{-3}s^{-1}]$')
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
for row, col, n in zip(anrows,ancols, range(4)):
    ax = axes[row, col]
    ax.text(locx[n],
            locy[n],
            anot[n],
            fontsize=14,
            ha='center',
            va='center',
            weight='bold',
            rotation=rot[n],
            transform=ax.transAxes)

" panel name"
if out == 'mean-comp':
    rows = [0, 0, 0, 1, 1, 1]
    cols = [0, 1, 2, 0, 1, 2]
elif out == 'shear':
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
elif out == 'mean-wind':
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
elif out == 'shear-mod':
    rows = [0, 1]
    cols = [0, 0]
pname = iter(list(string.ascii_lowercase)[:len(rows)])
for p,r,c, in zip(pname,rows,cols):
    ax = axes[r,c]
    transf = ax.transAxes
    ax.text(panel_name_loc,0.93,'({})'.format(p),
            fontsize=15,
            weight='bold',
            transform=transf)

if out == 'mean-comp':
    tx = '13-season mean and std_dev wind component profile'
    plt.subplots_adjust(hspace=0.1, wspace=0.15,
                        left=0.1, right=0.95)
elif out == 'shear':
    tx = '13-season mean vertical wind-shear profile'
    plt.subplots_adjust(hspace=0.1, wspace=0.15,
                        right=1.15)
elif out == 'mean-wind':
    tx = '13-season wind distribution profile'
    plt.subplots_adjust(hspace=0.1, wspace=0.15,
                        right=1.15)
elif out == 'shear-mod':
    tx = '13-season shear magnitude profile'
    plt.subplots_adjust(hspace=0.1, wspace=0.15,
                        left=0.15, right=2.1)

plt.suptitle(tx,fontsize=15,weight='bold',y=0.98)

# plt.show()

fname = ('/Users/raulvalenzuela/Documents/'
         'windprof_components_{}.png'.format(out))
plt.savefig(fname, dpi=300, format='png',papertype='letter',
           bbox_inches='tight')
