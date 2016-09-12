"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu



"""

import parse_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from tta_analysis2 import tta_analysis
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['mathtext.default'] = 'sf'

def sin(arg):
    return np.sin(np.radians(arg))

def cos(arg):
    return np.cos(np.radians(arg))


" creates one param per nhour "
thres = [112, 126, 140, 154, 168]
params = [{'wdir_thres': tr,
           'wdir_layer': [0, 500],
           'rain_czd':   0.25,
           'nhours':     1
           } for tr in thres]


# years = [1998]
years = [1998]+range(2001, 2013)

U = pd.Series(name='U-comp')
V = pd.Series(name='V-comp')

try:
    results

except NameError:

    results = collections.OrderedDict()
    for tr in thres:
        results[tr] = {'U': U, 'V': V}

    for year in years:
        for p in params:
            tta = tta_analysis(year=year)

            tta.start_df_layer(**p)
            tta_dates = tta.tta_dates

            " parse surface and profile obs "
            bby = parse_data.surface('bby', year=year)
            wpr = parse_data.windprof(year=year)

            wpr_tta = wpr.dframe.loc[tta_dates]
            wdr_tta = wpr_tta['wdir']
            wsp_tta = wpr_tta['wspd']

            bby_tta = bby.dframe.loc[tta_dates]

            " append surface values to windprof "
            surf_wsp = iter(bby_tta.wspd.values.tolist())
            surf_wdr = iter(bby_tta.wdir.values.tolist())

            wsp_tta = wsp_tta.map(lambda x: [surf_wsp.next()] + x)
            wdr_tta = wdr_tta.map(lambda x: [surf_wdr.next()] + x)

            " compute components "
            wd_sin = wdr_tta.map(lambda x: sin(x))
            wd_cos = wdr_tta.map(lambda x: cos(x))
            u = -1 * wsp_tta.multiply(wd_sin)
            v = -1 * wsp_tta.multiply(wd_cos)

            " append to big dataframe "
            thres = p['wdir_thres']
            results[thres]['U'] = results[thres]['U'].append(u)
            results[thres]['V'] = results[thres]['V'].append(v)

    print('Done')

fig, ax = plt.subplots(1, 2, sharey=True,sharex=True)
y = [0]
y = np.append(y, wpr.hgt)

cmap = discrete_cmap(7, base_cmap='Set1')
colors = [cmap(r) for r in range(len(thres))]
lw = 3

for thres,cl in zip(results.keys(),colors):

    U_stack = np.array(results[thres]['U'].tolist())
    V_stack = np.array(results[thres]['V'].tolist())

    U_mean = np.nanmean(U_stack, axis=0)
    V_mean = np.nanmean(V_stack, axis=0)

    ax[0].plot(U_mean, y, color=cl, lw=lw,
               label=str(thres)+'$^{\circ}$')
    ax[1].plot(V_mean, y, color=cl, lw=lw)

ax[0].set_ylabel('Altitude [m] MSL')
ax[0].legend(loc=0, fontsize=15)
ax[0].text(0,4100,'U-comp', ha='center', fontsize=15)
ax[0].set_xlabel('$[m\,s^{-1}]$')

ax[1].text(6,4100,'V-comp', ha='center', fontsize=15)
ax[1].set_xlabel('$[m\,s^{-1}]$')

ax[0].grid(True)
ax[1].grid(True)

ax[0].set_xlim([-10,15])

plt.subplots_adjust(bottom=0.15)

plt.show()
