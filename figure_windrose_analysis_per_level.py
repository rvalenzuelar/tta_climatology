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
from matplotlib import rcParams

# if seaborn-style plot shows up need 
# to use:
# sns.reset_defaults()
# %matplotlib inline


rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['ytick.color'] = (0.8, 0.8, 0.8)
rcParams['mathtext.default'] = 'sf'

''' get percentile '''
def get_wdir_perc(axes,perc):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    tcsum=tsum.cumsum()
    # add 5deg to get mid of bin
    stat = wdir[np.where(tcsum <= perc)[0][-1]]+5
    return stat

def get_wdir_mode(axes):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    # add 5deg to get mid of bin
    stat = wdir[np.where(tsum.max() == tsum)[0][0]] + 5
    return stat


# target_hgts = tuple(range(-1, 11))
target_hgts = (-1, 0, 2, 4, 6, 8, 10, 13, 15, 20, 26, 31)

#years = [1998]
years = [1998]+range(2001,2013)

try:
    wdsrf
except NameError:
    ws = {th:list() for th in target_hgts}
    wd = {th:list() for th in target_hgts}
    wdsrf = list()
    
    select_rain = 'all'
    
    for year in years:
          
        wpr  = parse_data.windprof(year=year)
        wspd = wpr.dframe.wspd
        wdir = wpr.dframe.wdir
        hgt  = wpr.hgt
        
        czd = parse_data.surface('czd', year=year)
        bby = parse_data.surface('bby', year=year)        

        if select_rain == 'all':
            select = None
        elif select_rain == 'czd':
            rain_czd = czd.dframe.precip > 0
            select = rain_czd[rain_czd].index
        elif select_rain == 'bby':
            rain_bby = bby.dframe.precip > 0
            select = rain_bby[rain_bby].index
        elif select_rain == 'norain':
            norain_czd = czd.dframe.precip == 0 
            norain_bby = bby.dframe.precip == 0 
            norain = norain_czd & norain_bby
            select = norain[norain].index
        
        if select is None:
            wd[-1].extend(bby.dframe.wdir.values.astype(float))
            ws[-1].extend(bby.dframe.wspd.values.astype(float))                
            wspd_rain = wspd
            wdir_rain = wdir            
        else:
            wd[-1].extend(bby.dframe.wdir.loc[select].values.astype(float))
            ws[-1].extend(bby.dframe.wspd.loc[select].values.astype(float))               
            wspd_rain = wspd.loc[select]
            wdir_rain = wdir.loc[select]
        
        for h in target_hgts[1:]:
            for s,d in zip(wspd_rain, wdir_rain):
                ws[h].append(s[h])
                wd[h].append(d[h])

scale = 1.0
axes = WindroseAxes.from_ax(subplots=(3, 4),
                            space=(0.05, 0.1),
                            figsize=(8*scale, 7*scale))
lw = 2
zorder = 10000
hgt = np.append(hgt, (43,))
first = True
panel = ['a','b','c','d','e','f','g','h','i','j','k','l']
for h,ax,pan in zip(target_hgts,axes,panel):
    
    ax.contourf(wd[h], ws[h],
                bins=range(0,30,3),  # speed bins
                nsector=36,
                cmap=cm.viridis,
                normed=True)

    if h == -1:
        txt = 'Surface'
    else:
        txt = '{:2.0f}m'.format(hgt[h])
    ax.text(0.5, 1.05, '('+pan+') '+txt,
            weight='bold',
            fontsize=15,
            ha='center',
            transform=ax.transAxes)

    # ax.text(0.05, 0.05, '('+pan+')',
    #         weight='bold',
    #         fontsize=15,
    #         ha='center',
    #         transform=ax.transAxes)

    if h == 0:
        xtcklab = ax.get_xticklabels()
        ax.set_xticklabels(xtcklab,
                           fontsize=10,
                           position=(0, -0.2),
                           weight='bold',
                           color=(0.5, 0.5, 0.5, 0.8))

    else:
        ax.set_xticklabels('')

    " adjust frequency axis "
    max_ylim = 15    
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
        
    ''' add stat line '''
#    stat_wdir = get_wdir_mode(ax)
#    theta = np.array([-stat_wdir+90,-stat_wdir+90])*np.pi/180.
#    ax.plot(theta,[0,max_ylim], color='r',
#            lw=lw, zorder=zorder)
#    stat_wdir = get_wdir_perc(ax,50)
#    theta = np.array([-stat_wdir+90,-stat_wdir+90])*np.pi/180.
#    ax.plot(theta,[0,10], color='k', linestyle='--',
#            lw=lw, zorder=zorder)    

    
''' add legend '''
axes[8].legend(bbox_to_anchor=(0.5, -0.9, 0.1, 0.1),
               # loc=(-0.1,-0.6),
               ncol=3,
               fontsize=12)
axes[9].text(1.1,-0.2,'wind speed [$m\,s^{-1}$]',
              fontsize=12,ha='center',
              transform=axes[9].transAxes)

if select_rain == 'czd':
    select_rain += '-rain'
elif select_rain == 'bby':
    select_rain += '-rain'
elif select_rain == 'all':
    select_rain = 'winter-season'
    
tx = 'Wind roses at BBY for {} hours'.format(select_rain)
#tx  = 'Wind roses at BBY for hours with rain CZD '
#tx += '$\geq$ 0.25 mm'
plt.suptitle(tx,fontsize=15, weight='bold',y=0.99)


# plt.show()

# place = '/home/raul/Desktop/'
place = '/Users/raulvalenzuela/Documents/'
fname = 'windrose_perhgt_0-3013m_{}.png'.format(select_rain)
plt.savefig(place+fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
