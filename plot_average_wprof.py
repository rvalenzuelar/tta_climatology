
import parse_data as pr
import matplotlib.pyplot as plt
import numpy as np
from tta_analysis import tta_analysis

sp_tta = []
wd_tta = []
sp_notta = []
wd_notta = []

for y in [1998]+range(2001, 2013):

    tta = tta_analysis(y)
    tta.start()

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


sp_tta_min = np.min(np.squeeze(np.array(sp_tta)), axis=0)
sp_notta_min = np.min(np.squeeze(np.array(sp_notta)), axis=0)

fig, ax = plt.subplots(2, 2, figsize=(8, 11), sharey=True)
ax = ax.flatten()

y = tta.wprof_hgt
blue_alpha = (0, 0, 1, 0.3)
green_alpha = (0, 0.6, 0, 0.3)

sptta_mean = np.array(sp_tta).mean(axis=0)
wdtta_mean = np.array(wd_tta).mean(axis=0)
spnotta_mean = np.array(sp_notta).mean(axis=0)
wdnotta_mean = np.array(wd_notta).mean(axis=0)

utta_mean = -sptta_mean*np.sin(np.radians(wdtta_mean))
unotta_mean = -spnotta_mean*np.sin(np.radians(wdnotta_mean))
vtta_mean = -sptta_mean*np.cos(np.radians(wdtta_mean))
vnotta_mean = -spnotta_mean*np.cos(np.radians(wdnotta_mean))

lw = 3
ax[0].plot(sptta_mean, y, label='TTA', linewidth=lw)
ax[0].plot(spnotta_mean, y, label='NO-TTA', linewidth=lw)
hs, labs = ax[0].get_legend_handles_labels()
ax[0].set_xlabel('speed [ms-1]')
ax[0].set_ylabel('height [m MSL]')
ax[0].legend(hs, labs, loc=4)
ax[0].text(0.05, 0.95, 'Wind speed', weight='bold',
           transform=ax[0].transAxes)

ax[1].plot(wdtta_mean, y,  linewidth=lw)
ax[1].plot(wdnotta_mean, y, linewidth=lw)
ax[1].set_xlabel('direction [deg]')
ax[1].text(0.05, 0.95, 'Wind direction', weight='bold',
           transform=ax[1].transAxes)

ax[2].plot(utta_mean, y, linewidth=lw)
ax[2].plot(unotta_mean, y, linewidth=lw)
ax[2].set_xlabel('speed [ms-1]')
ax[2].set_ylabel('height [m MSL]')
ax[2].text(0.05, 0.95, 'U-comp', weight='bold',
           transform=ax[2].transAxes)

ax[3].plot(vtta_mean, y, linewidth=lw)
ax[3].plot(vnotta_mean, y, linewidth=lw)
ax[3].set_xlabel('speed [ms-1]')
ax[3].text(0.05, 0.95, 'V-comp', weight='bold',
           transform=ax[3].transAxes)

plt.show(block=False)
