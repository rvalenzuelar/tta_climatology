# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:08:35 2016

@author: raulv
"""

import parse_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rv_utilities import add_floating_colorbar,discrete_cmap


import matplotlib as mpl
#inline_rc = dict(mpl.rcParams)
#mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15



target_hgts = tuple(range(16))
#target_hgts = (9,10,11,12,13,14,15,16,17)

#target_hgts = (0,1,2,3,4,5,6,7,8,9)
#target_hgts = (0,9,10,11,12,13,14,15,16,17)

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

sns.set_style("whitegrid")
scale = 1.1
fig,axes = plt.subplots(4,4,figsize=(11*scale,11*scale),
                        sharex=True,sharey=True)
axes=axes.flatten()
cmap = discrete_cmap(7, base_cmap='Set1')
color = cmap(1)
fsize = 15
lw = 2
normed = True
lim_surf = 130
lim_160m = 170
x = np.array(wdsrf)
#x = np.array(wd[0])
first = True
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
    
    ''' contourf '''
    v = np.arange(4,34,4)
    im = ax.contourf(X,Y,H,v,
                     cmap=cm.get_cmap('viridis'))
    ''' 1:1 line '''
    ax.plot([0,360],[0,360],'--',color=(0.5,0.5,0.5))
    
    ''' hlines and vlines '''    
#    ax.hlines(lim_surf,0,360,color=color)
#    ax.vlines(lim_160m,0,360,color=color)
#
#    if first:    
#        ax.text(0,lim_surf,str(lim_surf),
#                fontsize=fsize,color=color,
#                weight='bold')
#        ax.text(lim_160m,0,str(lim_160m),
#                fontsize=fsize,color=color,
#                weight='bold') 
#        first = False
        
    ''' altitude text '''
    ax.text(60,300,'{:2.0f}m'.format(hgt[h]),fontsize=15,
            weight='bold')


axes[0].text(5,30,'1:1',fontsize=15,rotation=45,
            color=(0.5,0.5,0.5),va='bottom')
axes[8].text(-90,370,'wdir-surface',fontsize=15,
             ha='right',rotation=90)
#axes[3].set_ylabel('wdir-160-m')
#axes[4].text(310,-60,'wdir-above-ground',fontsize=15)
axes[13].text(180,-90,'wdir-above-ground',fontsize=15,)

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

axes[-1].remove()

#plt.show()

#fname='/home/raul/Desktop/hist2d_surf_160-1541.png'
fname='/Users/raulv/Desktop/hist2d_surf_160-1541.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
