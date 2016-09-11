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

from matplotlib import rcParams
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from rv_utilities import discrete_cmap,pandas2stack

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
        ' at least one non-missin)'
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
wind_flow_mean = [dict(), dict()]
nans_per_level = [np.array([]).astype(int),
                  np.array([]).astype(int)]

' for each all/rainy category '
for n in range(2):

    print n
    WD_sin = WD[n].apply(lambda x: sin(x))
    WD_cos = WD[n].apply(lambda x: cos(x))

    U_df = -1*WS[n].multiply(WD_sin)
    V_df = -1*WS[n].multiply(WD_cos)

    wind_flow_180 = -(U_df*sin(180)+V_df*cos(180))
    wind_flow_90 = U_df*sin(90)+V_df*cos(90)

    stack_180 = np.array(wind_flow_180.tolist())
    stack_90 = np.array(wind_flow_90.tolist())

    ' mean value per level '
    wind_flow_mean[n][180] = np.nanmean(stack_180,axis=0)
    wind_flow_mean[n][90] = np.nanmean(stack_90,axis=0)

    nans_per_level[n] = np.append(nans_per_level[n],
                                  np.isnan(stack_180).sum(axis=0))

cmap = discrete_cmap(7, base_cmap='Set1')
colors = [cmap(0), cmap(1), (0, 0, 0)]

dz = np.array([160]+[92]*(max_hgt_gate-1))

y = np.array([0])
y = np.append(y, hgt[:max_hgt_gate])
ydz = y[:-1]+(y[1:]-y[:-1])/2.


" percentage of obs for each category "
wind_flow_mean[0][0] = 100*(1-(nans_per_level[0]/float(WD[0].index.size)))
wind_flow_mean[1][0] = 100*(1-(nans_per_level[1]/float(WD[1].index.size)))


# fig, axes = plt.subplots(2, 2, figsize=(8,8), sharey=True)
fig = plt.figure()
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[2,2,1],
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
out = 'shear'

for row in range(2):
    for col,comp,cl in zip(range(3), [90, 180, 0], colors):

        ax = axes[row,col]
        x = wind_flow_mean[row][comp]

        if out == 'mean':
            if col <2:

                f = interp1d(y, x)
                ynew = np.linspace(0, int(y.max()), 100)
                xnew = f(ynew)
                # ax.plot(xnew, ynew, color=cl, lw=lw)
                ax.plot(x, y, color=cl, lw=lw)
                ax.set_xlim([-2, 14])
                if row == 1 and (col == 0 or col == 1):
                    ax.set_xlabel('$[m\,s^{-1}]$')
                anotU,anotV = ['U','V']
            else:
                x = wind_flow_mean[row][comp]
                ax.plot(x, y, color=cl, lw=lw)
                ax.set_xlim([0, 100])
                labels=['0','','40','','80','']
                ax.set_xticklabels(labels)
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

    ''' fill jet '''
#    cond1 = np.where(xnew<=xnew[0])[0]
#    cond2 = np.where(xnew>xnew[0])[0]
#    if cond1.size>0 and cond2.size>0:
#        jet = xnew[cond1]
#        axes[0].fill_betweenx(ynew,xnew,x2=jet.max(),
#                              where=xnew<jet.max(),
#                              color=color,
#                              alpha=0.5)


''' axis annotation '''
locx = [0.5, 0.5, 1.1, 1.1]
locy = [1.05, 1.05, 0.5, 0.5]
anot = [anotU,
        anotV,
        'All (n={:d})'.format(WD[0].index.size),
        'czd-rain (n={:d})'.format(WD[1].index.size)]
rot = [0, 0, -90, -90]
for row, col, n in zip([0, 0, 0, 1], [0, 1, 2, 2], range(4)):
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

if out == 'mean':
    tx = '13-season mean wind profile'
    gs.update(hspace=0.1, wspace=0.15)
else:
    tx = '13-season mean vertical wind-shear profile'
    gs.update(hspace=0.1, wspace=0.15, right=1.2)
plt.suptitle(tx,fontsize=15,weight='bold',y=1.0)

# fig.set_size_inches(5,10)
# fig.canvas.draw()

# plt.show()


# # #fname='/home/raul/Desktop/fig_windrose_layer_0-500.png'
# fname = ('/Users/raulvalenzuela/Documents/'
#          'windprof_components_{}.png'.format(out))
# plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
