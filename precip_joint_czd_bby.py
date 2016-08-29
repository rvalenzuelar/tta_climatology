# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:06:41 2016

@author: raulv
"""
import matplotlib.pyplot as plt
import numpy as np
import parse_data
from matplotlib.colors import LogNorm
from rv_utilities import discrete_cmap

import matplotlib as mpl
#inline_rc = dict(mpl.rcParams)
#mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['mathtext.default'] = 'sf'

years = [1998]+range(2001,2013)
xlist = list()
ylist = list()

for year in years:

    czd=parse_data.surface('czd', year=year)
    bby=parse_data.surface('bby', year=year)

    first_bby = bby.dframe.index[0]
    first_czd = czd.dframe.index[0]

    last_bby = bby.dframe.index[-1]
    last_czd = czd.dframe.index[-1]
    
    first = max(first_bby,first_czd)   
    last  = min(last_bby,last_czd)
    x = bby.dframe.loc[first:last].precip.values.astype(float)
    y = czd.dframe.loc[first:last].precip.values.astype(float)

    ''' remove nans '''
    isnan = np.isnan(x) | np.isnan(y)
    x = x[~isnan]
    y = y[~isnan]

    ''' filter '''
##    fltr = (x>0) & (y>0)
#    fltr =  (y>0)
#    x = x[fltr]    
#    y = y[fltr]    
    
    
    xlist.extend(x)
    ylist.extend(y)


gauge_res = 0.254
nbins = 100
H,xed,yed = np.histogram2d(xlist,ylist,
                           bins=np.arange(-gauge_res/2.,
                                          gauge_res*0.5*nbins*2,
                                          gauge_res),
                           normed=False)

Hm = np.ma.masked_where(H<=2,H)

fig,ax = plt.subplots(figsize=(10,8))

im = ax.pcolormesh(xed,yed,Hm,
               norm=LogNorm(),
               cmap='inferno',
               vmax = 5e4
               )
cmap = discrete_cmap(7, base_cmap='Set1')
color = cmap(1)
ax.plot([-gauge_res/2.,6],[2,-gauge_res/2.],
        color=color,lw=3,zorder=1000)
ax.set_xlabel('rain czd $[mm\,h^{-1}]$')
ax.set_ylabel('rain bby $[mm\,h^{-1}]$')
ax.set_xlim([-gauge_res/2.,10+gauge_res])
ax.set_ylim([-gauge_res/2.,10+gauge_res])
plt.colorbar(im,label='count')
plt.grid()

plt.show()

##fname='/home/raul/Desktop/hist2d_surf_160-896.png'
#fname='/Users/raulv/Desktop/hist2d_precip.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
