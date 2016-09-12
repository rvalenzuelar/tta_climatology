"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse_data
import seaborn as sns
from matplotlib import rcParams
from rv_utilities import discrete_cmap

# if seaborn-style plot shows up need
# to use:
# %matplotlib inline

sns.reset_orig()

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


try:
    WS
except NameError:

    WD = pd.Series()
    WS = pd.Series()

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
        first = max(first_bby ,first_czd ,first_wpr)
        last  = min(last_bby ,last_czd ,last_wpr)

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
        precip = pd.concat([bby.precip ,czd.precip] ,axis=1)
        precip_nans = precip.apply(lambda x: x.isnull().any(),
                                   axis=1 ,reduce=True)
        precip_nans. name ='precip_nan'
        tx = 'year:{}, any_precip_nan:{:4d}'
        print(tx.format(year ,precip_nans.sum()))

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
        WD = WD.append(wdr_rain)
        WS = WS.append(wsp_rain)

WD_sin = WD.apply(lambda x: sin(x))
WD_cos = WD.apply(lambda x: cos(x))

U_df = -1 * WS.multiply(WD_sin)
V_df = -1 * WS.multiply(WD_cos)

wind_flow_180 = -(U_df * sin(180) + V_df * cos(180))
wind_flow_90 = U_df * sin(90) + V_df * cos(90)

" mean lowest 500 m"
mean_V = wind_flow_180.apply(lambda x: np.nanmean(x[:5]))
mean_U = wind_flow_90.apply(lambda x: np.nanmean(x[:5]))

wd_layer = 270-(np.arctan2(mean_V, mean_U)*180/np.pi)
wd_layer[wd_layer > 360] -= 360


thres = [112, 126, 140, 154, 168]
cmap = discrete_cmap(7, base_cmap='OrRd')
colors1 = [cmap(r+2) for r in range(len(thres))]
cmap = sns.color_palette("GnBu_d",6)
cmap.reverse()
colors2 = cmap
lw = 3

fig, ax = plt.subplots(1, 2, sharey=True,sharex=True)

for th,cl1,cl2 in zip(thres, colors1, colors2):

    wd_thr = wd_layer[wd_layer < th]

    U_thr = U_df[wd_thr.index]
    V_thr = V_df[wd_thr.index]

    U_thr_mean = np.nanmean(np.array(U_thr.tolist()),axis=0)
    V_thr_mean = np.nanmean(np.array(V_thr.tolist()),axis=0)

    y = [0]
    y = np.append(y, hgt)

    if th == 140:
        mk = 'o'
    else:
        mk = None
    ax[0].plot(U_thr_mean, y, color=cl1, lw=lw,
               label='$<$'+str(th)+'$^{\circ}$',
               marker=mk)
    ax[1].plot(V_thr_mean, y, color=cl2, lw=lw,
               label='$<$' + str(th) + '$^{\circ}$',
                marker=mk)

ax[0].vlines(0, 0, 3000, color=(0.4, 0.4, 0.4),
             lw=lw, linestyle='--')
ax[1].vlines(0, 0, 3000, color=(0.4, 0.4, 0.4),
             lw=lw, linestyle='--')

ax[0].set_ylim([0, 3000])

ax[0].set_ylabel('Altitude [m] MSL')
ax[0].legend(loc=0, fontsize=12, numpoints=1)
ax[0].text(0.5,1.03,'U-comp',
           fontsize=14,
           ha='center',
           va='center',
           weight='bold',
           transform=ax[0].transAxes)
ax[0].set_xlabel('$[m\,s^{-1}]$')
ax[1].legend(loc=0, fontsize=12, numpoints=1)
ax[1].text(0.5,1.03,'V-comp',
           fontsize=14,
           ha='center',
           va='center',
           weight='bold',
           transform=ax[1].transAxes)
tx = 'czd-rain (n={:d})'.format(WD.index.size)
ax[1].text(1.05, 0.5, tx,
           fontsize=14,
           ha='center',
           va='center',
           weight='bold',
           transform=ax[1].transAxes,
           rotation=-90)
ax[1].set_xlabel('$[m\,s^{-1}]$')

ax[0].grid(True)
ax[1].grid(True)

ax[0].set_xlim([-10,15])

plt.subplots_adjust(top=0.85, bottom=0.1)

tx = '13-season mean wind component profile\nper wind direction threshold'
plt.suptitle(tx,fontsize=15,weight='bold',y=1.0)


# #fname='/home/raul/Desktop/fig_windrose_layer_0-500.png'
fname = ('/Users/raulvalenzuela/Documents/'
         'windprof_tta_sensit.png')
plt.savefig(fname, dpi=300, format='png',papertype='letter',
           bbox_inches='tight')
