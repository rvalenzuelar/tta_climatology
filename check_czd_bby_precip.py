import numpy as np
import parse_data
import matplotlib.pyplot as plt

fig, ax = plt.subplots(7, 2, figsize=(10, 11))
ax = ax.flatten()

for n, y in enumerate([0, 1998] + range(2001, 2013)):
    if n == 0:
        ax[n].axis('off')
    else:
        print y
        bby = parse_data.bby(y, hourly=True)
        czd = parse_data.czd(y, hourly=True)
        bby.precip.plot(ax=ax[n])
        czd.precip.plot(ax=ax[n])
        ax[n].set_xticklabels('')
        ax[n].text(0.8, 1.0, str(y), transform=ax[n].transAxes)

fig.suptitle('BBY Precipitation [mm]')
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
plt.show(block=False)
