'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''

import parse_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import matplotlib.gridspec as gsp
import matplotlib.dates as mdates
from rv_utilities import pandas2stack

target='wdir'

fig = plt.figure(figsize=(10,5))

gs = gsp.GridSpec(1, 2,
                  width_ratios=[2,1]
                  )

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

y=2003

wprof = parse_data.windprof(y)
wp = np.squeeze(pandas2stack(wprof.dframe[target]))
wp_ma = ma.masked_where(np.isnan(wp),wp)
X,Y=wprof.time,wprof.hgt
ax1.pcolormesh(X,Y,wp_ma,vmin=0,vmax=360)
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax1.set_xlabel(r'$ Time \rightarrow$')
ax1.set_ylabel('height gate')

for prof in range(wp.shape[1]):
    ax2.plot(wp[:,prof],range(wp.shape[0]),
             color='r',alpha=0.05)
ax2.set_yticklabels('')
ax2.set_xlabel(target)

ax1.set_title('BBY Windprof wdir')
plt.tight_layout()
plt.show(block=False)