'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''
import parse_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.dates as mdates
from rv_utilities import pandas2stack

fig, ax = plt.subplots(7, 2, figsize=(10, 11))
ax = ax.flatten()

target='wdir'

for n, y in enumerate([0, 1998] + range(2001, 2013)):
    if n == 0:
        ax[n].axis('off')
    else:
        print y
        if n != 1:
            ax[n].set_yticklabels('')
        if n == 12:
            ax[n].set_xlabel(r'$ Time \rightarrow$')
        
        # parse windprof dataframe with wspd and wdir
        wprof = parse_data.windprof(y)
        wp = np.squeeze(pandas2stack(wprof.dframe[target]))

        # plot array
        # ax[n].imshow(wp, aspect='auto', origin='lower',
        #              interpolation='none')

        X,Y = wprof.time,wprof.hgt
        wp_ma = ma.masked_where(np.isnan(wp),wp)
        ax[n].pcolormesh(X,Y,wp_ma)
        ax[n].xaxis.set_major_locator(mdates.MonthLocator())
        ax[n].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        txt = 'Season: {}/{}'
        ax[n].text(0.05, 0.8, txt.format(str(y-1),str(y)),
                    weight='bold', transform=ax[n].transAxes)

fig.suptitle('BBY Windprof '+target)
fig.subplots_adjust(bottom=0.05, top=0.95,
                    left=0.05, right=0.95,
                    hspace=0.2)
plt.show(block=False)
