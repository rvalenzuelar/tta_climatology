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
# sns.reset_defaults()
# %matplotlib inline


mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['ytick.color'] = (0.8,0.8,0.8)
mpl.rcParams['mathtext.default'] = 'sf'

''' get percentile '''
def get_wdir_perc(axes,perc):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    tcsum=tsum.cumsum()
    # add 5deg to get mid of bin
    stat = wdir[np.where(tcsum<=perc)[0][-1]]+5
    return stat

def get_wdir_mode(axes):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    # add 5deg to get mid of bin
    stat = wdir[np.where(tsum==tsum.max())[0][0]]+5
    return stat


target_hgts = tuple(range(-1,15))
#target_hgts = (0,2,4,6,8,10)

#years = [1998]
years = [1998]+range(2001,2013)

try:
    wdsrf
except NameError:
    ws = {th:list() for th in target_hgts}
    wd = {th:list() for th in target_hgts}
    wdsrf = list()
    
    for year in years:
          
        wpr  = parse_data.windprof(year=year)
        wspd = wpr.dframe.wspd
        wdir = wpr.dframe.wdir
        hgt  = wpr.hgt
        
        czd = parse_data.surface('czd', year=year)
        rain_czd = czd.dframe.precip > 0
#        rain_dates = rain_czd.loc[rain_czd.values].index
        rain_dates = None
    
        bby = parse_data.surface('bby', year=year)
        
        if rain_dates is None:
            wd[-1].extend(bby.dframe.wdir.values.astype(float))
            ws[-1].extend(bby.dframe.wspd.values.astype(float))                
            wspd_rain = wspd
            wdir_rain = wdir            
        else:
            wd[-1].extend(bby.dframe.wdir.loc[rain_dates].values.astype(float))
            ws[-1].extend(bby.dframe.wspd.loc[rain_dates].values.astype(float))               
            wspd_rain = wspd.loc[rain_dates]
            wdir_rain = wdir.loc[rain_dates]
        
        for h in target_hgts[1:]:
            for s,d in zip(wspd_rain, wdir_rain):
                ws[h].append(s[h])
                wd[h].append(d[h])

scale = 1.3
axes = WindroseAxes.from_ax(subplots=(4,4),
                            space=(0.05,0.1),
                            figsize=(8*scale,9*scale))
lw = 2
zorder = 10000
hgt = np.append(hgt,[43])
first = True
for h,ax in zip(target_hgts,axes):
    
    ax.contourf(wd[h], ws[h],
                bins    = range(0,24,3),  # speed bins
                nsector = 36,
                cmap    = cm.viridis,
                normed  = True)

    if h == -1:
        txt='Surface'
    else:
        txt='{:2.0f}m'.format(hgt[h])
    ax.text(0.5,1.05,txt,
            weight='bold',fontsize=15,
            ha='center',
            transform=ax.transAxes)

    if h == 0:
        xtcklab = ax.get_xticklabels()
        ax.set_xticklabels(xtcklab,
                           fontsize=10,
                           position=(0,-0.2),
                           weight='bold',
                           color=(0.5,0.5,0.5,0.8))
    else:
        ax.set_xticklabels('')

    ''' add stat line '''
    max_ylim = 15
    stat_wdir = get_wdir_mode(ax)
    theta = np.array([-stat_wdir+90,-stat_wdir+90])*np.pi/180.
    ax.plot(theta,[0,max_ylim], color='r',
            lw=lw, zorder=zorder)
#    stat_wdir = get_wdir_perc(ax,50)
#    theta = np.array([-stat_wdir+90,-stat_wdir+90])*np.pi/180.
#    ax.plot(theta,[0,10], color='k', linestyle='--',
#            lw=lw, zorder=zorder)    


    ''' adjust frequency axis '''
    ax.set_ylim([0,max_ylim])
    ax.set_radii_angle(angle=45)
    ytks = ax.get_yticks()
    newlabs = ['']*len(ytks)   
    newlabs = [str(int(t)) for t in ytks[1::2]]
    newlabs[-1] = newlabs[-1]+'%'
    newtcks = ytks[1::2]
    if first:
        ax.set_yticks(newtcks)
        ax.set_yticklabels(newlabs,
                           fontsize=12,
                           weight='bold',
                           color=(0.5,0.5,0.5,0.8))
        first = False
    else:
        ax.set_yticks(newtcks)
        ax.set_yticklabels('')            
        
    
''' add legend '''
axes[12].legend(loc=(0.4,-0.5), ncol=4)
axes[13].text(1.0,-0.18,'wind speed [$m\,s^{-1}$]',
              fontsize=15,ha='center',
              transform=axes[13].transAxes)

tx  = 'Wind roses at BBY for all hours'
#tx  = 'Wind roses at BBY for hours with rain CZD '
#tx += '$\geq$ 0.25 mm'
plt.suptitle(tx,fontsize=15, weight='bold',y=0.95)

#plt.show()

fname='/home/raul/Desktop/windrose_perhgt_0-1449m_all.png'
#fname='/Users/raulv/Desktop/windrose_perhgt_srf-1500.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')