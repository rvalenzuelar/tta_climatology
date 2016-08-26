# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:08:35 2016

@author: raulv
"""

import parse_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from rv_utilities import add_floating_colorbar

# if seaborn-style plot shows up need 
# to use:
# %matplotlib inline
import matplotlib as mpl
inline_rc = dict(mpl.rcParams)
mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15



#target_hgts = (0,1,2,3,4,5,6,7,8)
target_hgts = (9,10,11,12,13,14,15,16,17)


#years = [2003]
years = [1998]+range(2001,2013)

try:
    wdsrf
except NameError:
    ws = {th:list() for th in target_hgts}
    wd = {th:list() for th in target_hgts}
    wdsrf = list()    
    for year in years:
        print(year)
        wpr = parse_data.windprof(year=year)
        wspd = wpr.dframe.wspd
        wdir = wpr.dframe.wdir
        hgt = wpr.hgt
        
        czd = parse_data.surface('czd', year=year)
        rain_czd = czd.dframe.precip > 0
        rain_dates = rain_czd.loc[rain_czd.values].index
    
        bby = parse_data.surface('bby', year=year)
        wdr_surf = bby.dframe.wdir
        
        wdsrf.extend(wdr_surf.loc[rain_dates].values.astype(float))
        
        wspd_rain = wspd.loc[rain_dates]
        wdir_rain = wdir.loc[rain_dates]
    
        for h in target_hgts:
            for s,d in zip(wspd_rain, wdir_rain):
                ws[h].append(s[h])
                wd[h].append(d[h])

scale = 1.5
fig,axes = plt.subplots(3,3,figsize=(11*scale,11*scale),
                        sharex=True,sharey=True)
axes=axes.flatten()
lw = 2
normed=True
x = np.array(wdsrf)
for h,ax in zip(target_hgts,axes):
    
    y = np.array(wd[h])
    nans = (np.isnan(x) | np.isnan(y))    
    H,xed,yed = np.histogram2d(x[~nans], y[~nans],
                                   bins=range(0,360,10),
                                   normed=normed)    
    
    if normed is True:
        H *= 1e5
    
    ''' make grid '''
    X,Y = np.meshgrid(xed[:-1],yed[:-1])    
    
    v = np.arange(4,34,4)
    im = ax.contourf(X,Y,H,
                     v,
                     cmap=cm.get_cmap('plasma'))

    ax.text(60,300,'{:2.0f}m'.format(hgt[h]),fontsize=15,
            weight='bold')

axes[3].set_ylabel('wdir-surface')
#axes[4].text(310,-60,'wdir-above-ground',fontsize=15)
axes[7].set_xlabel('wdir-above-ground')

''' ranges '''
ax.set_xticks(range(0,360,60))
ax.set_yticks(range(0,360,60))
ax.set_xlim([0,360])
ax.set_ylim([0,360])
                     
add_floating_colorbar(fig=fig,im=im,
                      position=[0.25,0.05,0.5,0.8],
                      loc='bottom',
                      label='Normalized frequency [%]')
                      
plt.subplots_adjust(hspace=0.05,wspace=0.05)                      
plt.show()

#fname='/home/raul/Desktop/all_season_windrose_perhgt.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
