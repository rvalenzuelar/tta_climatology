# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 16:18:24 2016

@author: raulv
"""

import matplotlib.pyplot as plt
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['axes.labelpad'] = 0.1

y1 = [1.52,1.51,1.50,1.52,1.68,2.02,2.40,2.67,2.83]
y2 = [1.46,1.48,1.46,1.45,1.62,1.98,2.38,2.69,2.83]
y3 = [1.42,1.47,1.42,1.40,1.57,1.95,2.34,2.64,2.83]
y4 = [1.33,1.42,1.40,1.35,1.44,1.85,2.26,2.65,2.93]

x = range(100,190,10)

fig,ax = plt.subplots(figsize=(10,8))

cmap = discrete_cmap(7, base_cmap='Set1')
lw = 2

ax.plot(x,y1,'o-',lw=lw,color=cmap(0),label='1h')
ax.plot(x,y2,'o-',lw=lw,color=cmap(1),label='2h')
ax.plot(x,y3,'o-',lw=lw,color=cmap(2),label='4h')
ax.plot(x,y4,'o-',lw=lw,color=cmap(3),label='8h')

ax.set_xlabel('wind direction limit')
ax.set_ylabel('rainfall ratio (CZD/BBY)')

plt.legend(loc=0,numpoints=1)

#plt.show()

#fname='/home/raul/Desktop/hist2d_surf_160-1541.png'
fname='/Users/raulv/Desktop/rainfall_ratio_surf_wd.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

