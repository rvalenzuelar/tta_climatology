"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu


"""

import matplotlib.pyplot as plt
import numpy as np
import tta_analysis3 as tta
from matplotlib import rcParams
from rv_utilities import discrete_cmap
from curve_fitting import curv_fit

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['mathtext.default'] = 'sf'


# years = [1998]
years = [1998] + range(2001, 2013)

try:
    wd_layer
except NameError:
    out = tta.preprocess(years=years, layer=[0, 500])
    precip_good = out['precip_good']
    precip_good = precip_good[precip_good.czd > 0.25]
    wd_layer = out['wd_layer']
    wd_layer = wd_layer[precip_good.index]

    bby_rainr = list()
    czd_rainr = list()
    ratio = list()
    bby_CI_bot = list()
    bby_CI_top = list()
    czd_CI_bot = list()
    czd_CI_top = list()
    rto_CI_bot = list()
    rto_CI_top = list()

    " construct threshold array "
    del_th = 10
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
        bby_CI = tta.bootstrap(rainr['bby'].values,
                           nsamples,
                           np.mean,
                           alpha)
        czd_CI = tta.bootstrap(rainr['czd'].values,
                           nsamples,
                           np.mean,
                           alpha)
        rto_CI = tta.bootstrap_ratio(rainr['czd'].values,
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

" start figure "
cmap = discrete_cmap(7, base_cmap='Set1')
cl_bby = cmap(2)
cl_czd = cmap(1)
cl_rto = cmap(0)
mk_size = 40
scale = 1.4
fig, ax = plt.subplots(2, 1,
                       figsize=(6*scale, 6*scale),
                       sharey=True, sharex=True)

' ------ add mean values and CI ------ '
ax[0].errorbar(thres+del_th/2, bby_rainr,
               yerr=[bby_rainr-bby_CI_bot,
                     bby_CI_top-bby_rainr],
               linestyle='none',
               color=cl_bby,
               fmt='o',
               label='BBY rain rate (95% CI)',
               lw=2,
               markeredgecolor='k')

ax[0].errorbar(thres+del_th/2, czd_rainr,
               yerr=[czd_rainr-czd_CI_bot,
                     czd_CI_top-czd_rainr],
               linestyle='none',
               color=cl_czd,
               fmt='o',
               label='CZD rain rate (95% CI)',
               lw=2,
               markeredgecolor='k')

ax[1].errorbar(thres+del_th/2, ratio,
               yerr=[ratio-rto_CI_bot,
                     rto_CI_top-ratio],
               linestyle='none',
               color=cl_rto,
               fmt='o',
               label='CZD/BBY ratio (95% CI)',
               lw=2,
               markeredgecolor='k')

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
tx = '$\mathit{{u}}$: {3:2.1f}\n' \
     '$\mathit{{l}}$: {0:2.1f}\n' \
     '$\mathit{{g}}$: {1:2.1f}\n' \
     '$\mathit{{c}}$: {2:2.1f}\n' \
     '$r^{{2}}$:{4:2.2f}'
tx_rto = tx.format(la, gr, ce, ua, fit_rto.R_sq)

grp = zip([tx_bby, tx_czd, tx_rto],
          [ax[0], ax[0], ax[1]],
          [cl_bby, cl_czd, cl_rto],
          [(245, 0.5), (195, 3.4), (145, 1.6)],
          [(240, 2.0), (210, 4.0), (180, 0.5)],
          ['k','k',(0,0,0,0)]
          )
for tx, a, cl, xy1, xy2, acl in grp:
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
                               ec=acl,
                               fc=acl,
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

' ------ add thres lines ------ '
lw=3
ls='--'
# ax[1].vlines(120,0,7,lw=lw,alpha=0.5,linestyle=ls)
# ax[1].vlines(130,0,7,lw=lw,alpha=0.5,linestyle=ls)
ax[1].vlines(140,0,7,lw=lw,alpha=0.5,linestyle=ls)
# ax[1].vlines(150,0,7,lw=lw,alpha=0.5,linestyle=ls)
# ax[1].vlines(160,0,7,lw=lw,alpha=0.5,linestyle=ls)

# ax[1].text(120,6,'120',rotation=90,ha='right')
# ax[1].text(130,6,'130',rotation=90,ha='right')
ax[1].text(140,6,'140',rotation=90,ha='right')
# ax[1].text(150,6,'150',rotation=90,ha='right')
# ax[1].text(160,6,'160',rotation=90,ha='right')

' ------ general figure setup ------ '
ax[0].text(0.93,0.9,'(a)',
           fontsize=15,
           weight='bold',
           transform=ax[0].transAxes)
ax[0].set_xticks(range(90, 280, 30))
ax[0].set_xlim([88, 272])
ax[0].set_ylabel('rain rate $[mm h^{-1}]$')
ax[0].set_ylim([0, 6])
# ax[0].grid(True)
ha, la = ax[0].get_legend_handles_labels()
leg = ax[0].legend([ha[3], ha[0], ha[2], ha[1]],
                   [la[3], la[0], la[2], la[1]],
                   scatterpoints=1,
                   numpoints=1,
                   loc=2)
leg.get_frame().set_visible(False)

ax[1].text(0.93,0.9,'(b)',
           fontsize=15,
           weight='bold',
           transform=ax[1].transAxes)
ax[1].set_ylim([0, 7])
ax[1].set_ylabel('ratio')
ax[1].set_xlabel('$\overline{WDIR}_{500}$')
# ax[1].grid(True)
handles, labels = ax[1].get_legend_handles_labels()
leg = ax[1].legend(handles[::-1],
                   labels[::-1],
                   scatterpoints=1,
                   numpoints=1,
                   loc=(0.45, 0.75))
leg.get_frame().set_visible(False)

k = 'Surf-500m'
tx = '13-season relationship between CZD, BBY rain\n'
tx += 'and wind direction over BBY in the layer-mean {}\n'.format(k)
tx += '(CZD$\geq$0.25mm; wind direction bins of {}$^\circ$)'.format(
    str(del_th))
plt.suptitle(tx, fontsize=15, weight='bold', y=1.0)



plt.subplots_adjust(hspace=0.08)

# plt.show()

# place = '/home/raul/Desktop/'
place = '/Users/raulvalenzuela/Documents/'
fname='relationship_rain_wd_bin{}.png'.format(del_th)
plt.savefig(place+fname, dpi=300, format='png', papertype='letter',
            bbox_inches='tight')