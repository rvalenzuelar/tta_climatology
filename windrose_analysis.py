# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:02:21 2016

@author: raul
"""

import parse_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from rv_windrose import WindroseAxes

# if seaborn-style plot shows up need 
# to use:
# %matplotlib inline
import matplotlib as mpl
inline_rc = dict(mpl.rcParams)
mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['ytick.color'] = (0.8,0.8,0.8)

''' get percentile '''
def get_wdir_perc(axes,perc):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    tcsum=tsum.cumsum()
    return wdir[np.where(tcsum<=perc)[0][-1]]


target_hgts = (0,1,2,3,4,5)

ws = {th:list() for th in target_hgts}
wd = {th:list() for th in target_hgts}

#years = [2003]
years = [1998]+range(2001,2013)


for year in years:
    wpr = parse_data.windprof(year=year)
    wspd = wpr.dframe.wspd
    wdir = wpr.dframe.wdir
    hgt = wpr.hgt
    
    czd = parse_data.surface('czd', year=year)
    rain_czd = czd.dframe.precip > 0
    rain_dates = rain_czd.loc[rain_czd.values].index
    
    wspd_rain = wspd.loc[rain_dates]
    wdir_rain = wdir.loc[rain_dates]
    
    for h in target_hgts:
        for s,d in zip(wspd_rain, wdir_rain):
            ws[h].append(s[h])
            wd[h].append(d[h])

scale = 1.1
axes = WindroseAxes.from_ax(subplots=(3,2),
                            figsize=(9*scale,12*scale))
lw = 2
zorder = 10000
for h,ax in zip(target_hgts,axes):
    ax.contourf(wd[h], ws[h],
                bins=range(0,24,3),  # speed bins
                nsector=36,
                cmap=cm.plasma,
                normed=True)
    ax.text(-0.1,0.9,'Hgt:{:2.0f}m'.format(hgt[h]),
            weight='bold',fontsize=15,
            transform=ax.transAxes)

    ''' percentile values '''
    # add 5deg to get mid of bin
    med_wdir = get_wdir_perc(ax,50) + 5
    above_text = '{}\n(50%)'.format(int(med_wdir))    
    theta = np.array([-med_wdir+90,-med_wdir+90])*np.pi/180.
    ax.plot(theta,[0,10], color='r', lw=lw, zorder=zorder)
    ax.text(0.58,-0.12,above_text,color='r',weight='bold',
                 transform=ax.transAxes)

    ''' adjust frequency axis '''
    for ax in axes:
        ax.set_radii_angle(angle=45)
        ytks = ax.get_yticks()
        newlabs = ['']*len(ytks)   
        newlabs = [str(int(t)) for t in ytks[1::2]]
        newtcks = ytks[1::2]
        ax.set_yticks(newtcks)
        ax.set_yticklabels(newlabs)
    
''' add legend '''
axes[4].legend(loc=(0.2,-0.4),
               ncol=4)

plt.show()

#fname='/home/raul/Desktop/all_season_windrose_perhgt.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
