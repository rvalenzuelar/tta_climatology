import parse_data
import matplotlib.pyplot as plt


fig, ax = plt.subplots(7, 2, figsize=(10, 11))
ax = ax.flatten()

for n, y in enumerate([0, 1998] + range(2001, 2013)):
    if n == 0:
        ax[n].axis('off')
    else:
        print y
        wprof = parse_data.windprof(y)
        ax[n].imshow(wprof.ws, aspect='auto', origin='lower',
                     interpolation='none')
        ax[n].set_xticklabels('')
        ax[n].text(0.8, 1.0, str(y), transform=ax[n].transAxes)

fig.suptitle('BBY Windprof wspd [mm]')
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
plt.show(block=False)
