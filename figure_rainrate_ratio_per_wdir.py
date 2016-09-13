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
from tta_analysis2 import bootstrap, bootstrap_ratio
from curve_fitting import curv_fit

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

    bby_rainr = list()
    czd_rainr = list()
    ratio = list()
    bby_CI_bot = list()
    bby_CI_top = list()
    czd_CI_bot = list()
    czd_CI_top = list()
    rto_CI_bot = list()
    rto_CI_top = list()

    del_th = 30
    ini_th = 90
    end_th = 270
    thres = np.array(range(ini_th-del_th/2, end_th+del_th/2, del_th))

    for th in thres:

        " sensitivity here "
        wd_thr = wd_layer[(wd_layer >= th) & (wd_layer < th+del_th)]
        rainr = precip_good.loc[wd_thr.index]
        brr = rainr['bby'].sum() / rainr.index.size
        crr = rainr['czd'].sum() / rainr.index.size
        bby_rainr.append(brr)
        czd_rainr.append(crr)
        ratio.append(crr/brr)

        " confidence interval for mean value "
        nsamples = 5000
        alpha = 0.05
        bby_CI = bootstrap(rainr['bby'].values,
                           nsamples,
                           np.mean,
                           alpha)
        czd_CI = bootstrap(rainr['czd'].values,
                           nsamples,
                           np.mean,
                           alpha)
        rto_CI = bootstrap_ratio(rainr['czd'].values,
                                 rainr['bby'].values,
                                 nsamples,
                                 alpha)

        bby_CI_bot.append(bby_CI[0])
        bby_CI_top.append(bby_CI[1])
        czd_CI_bot.append(czd_CI[0])
        czd_CI_top.append(czd_CI[1])
        rto_CI_bot.append(rto_CI[0])
        rto_CI_top.append(rto_CI[1])


    bby_CI_bot = np.array(bby_CI_bot)
    bby_CI_top = np.array(bby_CI_top)
    czd_CI_bot = np.array(czd_CI_bot)
    czd_CI_top = np.array(czd_CI_top)
    rto_CI_bot = np.array(rto_CI_bot)
    rto_CI_top = np.array(rto_CI_top)

    fit_czd = curv_fit(x=thres + del_th/2,
                       y=czd_rainr,
                       model='gaussian')

    fit_bby = curv_fit(x=thres + del_th/2,
                       y=bby_rainr,
                       model='gaussian')

    fit_rto = curv_fit(x=thres + del_th/2,
                       y=ratio,
                       model='4PL')


cmap = discrete_cmap(7, base_cmap='Set1')
cl_bby = cmap(2)
cl_czd = cmap(1)
cl_rto = cmap(0)
mk_size = 40
scale = 1.4
fig, ax = plt.subplots(2, 1,
                       figsize=(6*scale, 8*scale),
                       sharey=True, sharex=True)

' ------ add mean values and CI ------ '
ax[0].errorbar(thres+del_th/2, bby_rainr,
               yerr=[bby_rainr-bby_CI_bot,
                     bby_CI_top-bby_rainr],
               linestyle='none',
               color=cl_bby,
               fmt='o',
               label='BBY rain rate (95% CI)',
               lw=2)

ax[0].errorbar(thres+del_th/2, czd_rainr,
               yerr=[czd_rainr-czd_CI_bot,
                     czd_CI_top-czd_rainr],
               linestyle='none',
               color=cl_czd,
               fmt='o',
               label='CZD rain rate (95% CI)',
               lw=2)

ax[1].errorbar(thres+del_th/2, ratio,
               yerr=[ratio-rto_CI_bot,
                     rto_CI_top-ratio],
               linestyle='none',
               color=cl_rto,
               fmt='o',
               label='CZD/BBY ratio (95% CI)',
               lw=2)

' ------ annotate model parameters fitted  ------ '
mu_czd = fit_czd.params['center'].value
si_czd = fit_czd.params['sigma'].value
am_czd = fit_czd.params['amplitude'].value
r2 = fit_czd.R_sq
tx = '$A$:{0:2.1f}\n$\mu$:{1:2.1f}\n$\sigma$:{2:2.1f}\n$r^{3}$:{4:2.2f}'
tx_czd = tx.format(am_czd, mu_czd, si_czd, '2', r2)

mu_bby = fit_bby.params['center'].value
si_bby = fit_bby.params['sigma'].value
am_bby = fit_bby.params['amplitude'].value
r2 = fit_bby.R_sq
tx_bby = tx.format(am_bby, mu_bby, si_bby, '2', r2)

la = fit_rto.params['la'].value
gr = fit_rto.params['gr'].value
ce = fit_rto.params['ce'].value
ua = fit_rto.params['ua'].value
tx = 'bot_asym:    {0:2.1f}\ngrowth_rate:{1:2.1f}\n'
tx += 'center:         {2:2.1f}\nupp_asym:   {3:2.1f}\n'
tx += '$r^{5}$:{4:2.2f}'
tx_rto = tx.format(la, gr, ce, ua, fit_rto.R_sq, '2')

grp = zip([tx_bby, tx_czd, tx_rto],
          [ax[0], ax[0], ax[1]],
          [cl_bby, cl_czd, cl_rto],
          [(245, 0.5), (195, 3.4), (145, 1.6)],
          [(240, 2.0), (210, 4.0), (180, 0.5)],
          )
for tx, a, cl, xy1, xy2 in grp:
    a.annotate(tx,
               xy=xy1,
               xytext=xy2,
               xycoords='data',
               textcoords='data',
               zorder=10000,
               color=cl,
               weight='bold',
               fontsize=14,
               arrowprops=dict(arrowstyle='-|>',
                               ec='k',
                               fc='k',
                               )
               )


' ------ add fitted models ------ '
lw = 2

xnew = np.array(range(90, 271, 1))

ynew_bby = fit_bby.eval(x=xnew)
ynew_czd = fit_czd.eval(x=xnew)
ynew_rto = fit_rto.eval(x=xnew)
gaus_tx = 'Gaussian fit'
logi_tx = 'Logistic fit'

ax[0].plot(xnew,ynew_czd,lw=lw,
           color=cl_czd,
           label='CZD '+gaus_tx)
ax[0].plot(xnew,ynew_bby,lw=lw,
           color=cl_bby,
           label='BBY '+gaus_tx)
ax[1].plot(xnew,ynew_rto,lw=lw,
           color=cl_rto,
           label=logi_tx)

' ------ general figure setup ------ '
ax[0].set_xticks(range(90, 280, 30))
ax[0].set_xlim([88, 272])
ax[0].set_ylabel('rain rate $[mm h^{-1}]$')
ax[0].set_ylim([0, 6])
ax[0].grid(True)
ha, la = ax[0].get_legend_handles_labels()
leg = ax[0].legend([ha[3], ha[0], ha[2], ha[1]],
                   [la[3], la[0], la[2], la[1]],
                   scatterpoints=1,
                   numpoints=1,
                   loc=2)
leg.get_frame().set_visible(False)

ax[1].set_ylim([0, 6])
ax[1].set_ylabel('ratio')
ax[1].set_xlabel('wind direction')
ax[1].grid(True)
handles, labels = ax[1].get_legend_handles_labels()
leg = ax[1].legend(handles[::-1],
                   labels[::-1],
                   scatterpoints=1,
                   numpoints=1,
                   loc=2)
leg.get_frame().set_visible(False)

k = 'Surf-500m'
tx = '13-season relationship between CZD, BBY rain\n'
tx += 'and wind direction over BBY in the layer-mean {}\n'.format(k)
tx += '(wind direction bins of '+str(del_th)+'$^{\circ}$)'
plt.suptitle(tx, fontsize=15, weight='bold', y=0.99)

plt.subplots_adjust(hspace=0.05)

# plt.show()

place = '/home/raul/Desktop/'
fname='relationship_rain_wd_bin{}.png'.format(del_th)
# fname='/Users/raulv/Desktop/relationship_rain_wd.png'
plt.savefig(place+fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')