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
from tta_analysis2 import bootstrap

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
    precip_good = pd.DataFrame()

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
        precip = pd.concat([bby.precip, czd.precip] ,axis=1)
        precip.columns=['bby', 'czd']
        precip_nans = precip.apply(lambda x: x.isnull().any(),
                                   axis=1 ,reduce=True)
        precip_nans.name ='precip_nan'
        tx = 'year:{}, any_precip_nan:{:4d}'
        print(tx.format(year ,precip_nans.sum()))
        precip_good = precip_good.append(precip[~precip_nans])

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


    thres = np.array(range(85, 275, 10))
    cmap = discrete_cmap(7, base_cmap='OrRd')
    colors1 = [cmap(r+2) for r in range(len(thres))]
    cmap = sns.color_palette("GnBu_d", 6)
    cmap.reverse()
    colors2 = cmap
    lw = 3


    bby_rainr = list()
    czd_rainr = list()
    ratio = list()
    bby_CI_bot = list()
    bby_CI_top = list()
    czd_CI_bot = list()
    czd_CI_top = list()


    for th in thres:

        ' sensitivity here '
        wd_thr = wd_layer[(wd_layer >= th) & (wd_layer < th+10)]
        rainr = precip_good.loc[wd_thr.index]
        brr = rainr['bby'].sum() / rainr.index.size
        crr = rainr['czd'].sum() / rainr.index.size
        bby_rainr.append(brr)
        czd_rainr.append(crr)
        ratio.append(crr/brr)

        nsamples = 50000
        alpha = 0.05
        bby_CI = bootstrap(rainr['bby'].values,
                           nsamples,
                           np.mean,
                           alpha)
        czd_CI = bootstrap(rainr['czd'].values,
                           nsamples,
                           np.mean,
                           alpha)

        bby_CI_bot.append(bby_CI[0])
        bby_CI_top.append(bby_CI[1])
        czd_CI_bot.append(czd_CI[0])
        czd_CI_top.append(czd_CI[1])

    bby_CI_bot = np.array(bby_CI_bot)
    bby_CI_top = np.array(bby_CI_top)
    czd_CI_bot = np.array(czd_CI_bot)
    czd_CI_top = np.array(czd_CI_top)

    ratio_bot = czd_CI_bot/bby_CI_bot
    ratio_top = czd_CI_top/bby_CI_top

fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)

ax[0].scatter(thres+5, bby_rainr, color='g')
ax[0].errorbar(thres+5, bby_rainr,
               yerr=[bby_rainr-bby_CI_bot,
                     bby_CI_top-bby_rainr],
               linestyle='none',
               color='g')

ax[0].scatter(thres+5, czd_rainr, color='r')
ax[0].errorbar(thres+5, czd_rainr,
               yerr=[czd_rainr-czd_CI_bot,
                     czd_CI_top-czd_rainr],
               linestyle='none',
               color='r')

ax[0].grid(True)


ax[1].scatter(thres+5, ratio, color='b')
ax[1].errorbar(thres+5, ratio,
               yerr=[ratio-ratio_bot,
                     ratio_top-ratio],
               linestyle='none',
               color='b')

ax[1].grid(True)

plt.show()
