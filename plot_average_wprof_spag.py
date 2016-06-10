'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''

import parse_data as pr
import matplotlib.pyplot as plt
import numpy as np
from tta_analysis import tta_analysis

sp_tta = []
wd_tta = []
sp_notta = []
wd_notta = []

# for y in [1998]+range(2001, 2013):
for y in [1998]:

    print 'Analyzing year {}'.format(y)
    tta = tta_analysis(y)
    tta.start(wdir_surf=125,wdir_wprof=170,
              rain_bby=0.25,nhours=5)

    wspd_tta = tta.wprof_ws[tta.bool].T
    wdir_tta = tta.wprof_wd[tta.bool].T
    a, b, _, _, _, _ = pr.average_wind(wdir_tta, wspd_tta)
    sp_tta.append(b)
    wd_tta.append(a)

    wspd_notta = tta.wprof_ws[~tta.bool].T
    wdir_notta = tta.wprof_wd[~tta.bool].T
    a, b, _, _, _, _ = pr.average_wind(wdir_notta, wspd_notta)
    sp_notta.append(b)
    wd_notta.append(a)

sptta_mean = np.array(sp_tta).mean(axis=0)
spnotta_mean = np.array(sp_notta).mean(axis=0)
wdtta_mean = np.array(wd_tta).mean(axis=0)
wdnotta_mean = np.array(wd_notta).mean(axis=0)

y = tta.wprof_hgt
blue1 = (0, 0, 1, 0.3)
blue2 = (0, 0.5, 1, 0.3)
green2a = (0, 1, 0, 0.3)
green1a = (0.3, 0.6, 0, 0.3)
green1 = green1a[:3]
green2 = green2a[:3]

fig, ax = plt.subplots(1, 2, sharey=True)
ax = ax.flatten()

for st, wt, snt, wnt in zip(sp_tta, wd_tta, sp_notta, wd_notta):
    ax[0].plot(st, y, color=blue1)
    ax[0].plot(snt, y, color=blue2)
    ax[1].plot(wt, y, color=green1a)
    ax[1].plot(wnt, y, color=green2a)

lw = 5
ax[0].plot(sptta_mean, y, color=(0, 0, 1), linewidth=lw, label='TTA')
ax[0].plot(spnotta_mean, y, color=(0, 0.5, 1), linewidth=lw, label='NO-TTA')
ax[0].set_xlabel('wind speed [ms-1]')
ax[0].set_ylabel('height [m MSL]')
hs, labs = ax[0].get_legend_handles_labels()
ax[0].legend(hs, labs, loc=4)
ax[1].plot(wdtta_mean, y, color=green1, linewidth=lw, label='TTA')
ax[1].plot(wdnotta_mean, y, color=green2, linewidth=lw, label='NO-TTA')
ax[1].set_xlim([0,360])
ax[1].set_xlabel('wind direction [deg]')
hs, labs = ax[1].get_legend_handles_labels()
ax[1].legend(hs, labs, loc=0)

plt.show(block=False)
