# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:02:21 2016

@author: raul
"""

import parse_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
mpl.rcParams['mathtext.default'] = 'sf'

''' get percentile '''
def get_wdir_perc(axes,perc):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    tcsum=tsum.cumsum()
    return wdir[np.where(tcsum<=perc)[0][-1]]


#years = [1998]
years = [1998]+range(2001,2013)

bot_layer = 0
top_layer = 500  # [m]

WD = [pd.DataFrame(),pd.DataFrame()]
WS = [pd.DataFrame(),pd.DataFrame()]

try:
    ws
except NameError:
    
    for year in years:

        wpr = parse_data.windprof(year=year)
        bby = parse_data.surface('bby', year=year)
        czd = parse_data.surface('czd', year=year)
        
        hgt = wpr.hgt        
        
        ''' reduce to common time period '''
        first_bby = bby.dframe.index[0]
        first_czd = czd.dframe.index[0]
        first_wpr = wpr.dframe.index[0]
    
        last_bby = bby.dframe.index[-1]
        last_czd = czd.dframe.index[-1]
        last_wpr = wpr.dframe.index[-1]
        
        first = max(first_bby,first_czd,first_wpr)   
        last  = min(last_bby,last_czd,last_wpr)

        wpr = wpr.dframe.loc[first:last]
        bby = bby.dframe.loc[first:last]        
        czd = czd.dframe.loc[first:last]
        
        rain_czd = czd.precip > 0
        rain_dates = rain_czd.loc[rain_czd.values].index

        ''' init dataframes for year '''        
        wd = pd.DataFrame()          
        ws = pd.DataFrame()          
        wd_rain = pd.DataFrame(index=rain_dates)          
        ws_rain = pd.DataFrame(index=rain_dates)  
        
        bby_wd = bby.wdir
        bby_ws = bby.wspd
        
        ''' creates surface column '''
        wd['surf'] = bby_wd
        ws['surf'] = bby_ws            
        wd_rain['surf'] = bby_wd.loc[rain_dates]
        ws_rain['surf'] = bby_ws.loc[rain_dates]


        wspd = wpr.wspd
        wdir = wpr.wdir
        wspd_rain = wspd.loc[rain_dates]
        wdir_rain = wdir.loc[rain_dates]
        
        if top_layer >=160:
            idx = np.where((hgt>=bot_layer) & 
                           (hgt<top_layer))[0]

            ''' creates columns for each level '''
            for h in hgt[idx]:
                col = '{:2.0f}'.format(h)
                wd[col]=np.zeros(wd.index.size)        
                ws[col]=np.zeros(ws.index.size)        
                wd_rain[col]=np.zeros(wd_rain.index.size)        
                ws_rain[col]=np.zeros(ws_rain.index.size)  
            
            for n, [s,d] in enumerate(zip(wspd,wdir)):
                d = np.array(d)
                s = np.array(s)
                wd.iloc[n,1:] = d[idx]
                ws.iloc[n,1:] = s[idx]

            for n, [sr,dr] in enumerate(zip(wspd_rain, wdir_rain)):
                dr = np.array(dr)
                sr = np.array(sr)
                wd_rain.iloc[n,1:] = dr[idx]
                ws_rain.iloc[n,1:] = sr[idx]

        WD[0] = WD[0].append(wd)
        WS[0] = WS[0].append(ws)
        WD[1] = WD[1].append(wd_rain)
        WS[1] = WS[1].append(ws_rain)

    ''' average layer per hour '''
    WD_sine = WD[0].applymap(lambda x: np.sin(np.radians(x)))
    WD_cosn = WD[0].applymap(lambda x: np.cos(np.radians(x)))
    
    U_df = -1*WS[0].multiply(WD_sine)
    V_df = -1*WS[0].multiply(WD_cosn)
        
    U_mean = U_df.mean(axis=1)
    V_mean = V_df.mean(axis=1)
    
    U_sq = U_mean.apply(lambda x: x**2) 
    V_sq = V_mean.apply(lambda x: x**2) 
    
    WS_mean0 = np.sqrt(U_sq+V_sq)
    WD_mean0 = 270 - (np.arctan2(V_mean,U_mean)*180/np.pi)
    WD_mean0[WD_mean0>360] = WD_mean0[WD_mean0>360]-360
    
    ''' average layer per hour '''
    WD_sine = WD[1].applymap(lambda x: np.sin(np.radians(x)))
    WD_cosn = WD[1].applymap(lambda x: np.cos(np.radians(x)))
    
    U_df = -1*WS[1].multiply(WD_sine)
    V_df = -1*WS[1].multiply(WD_cosn)
        
    U_mean = U_df.mean(axis=1)
    V_mean = V_df.mean(axis=1)
    
    U_sq = U_mean.apply(lambda x: x**2) 
    V_sq = V_mean.apply(lambda x: x**2) 
    
    WS_mean1 = np.sqrt(U_sq+V_sq)
    WD_mean1 = 270 - (np.arctan2(V_mean,U_mean)*180/np.pi)
    WD_mean1[WD_mean1>360] = WD_mean1[WD_mean1>360]-360



scale = 1.0
axes = WindroseAxes.from_ax(subplots=(2,1),
                            space=[0.05,0.05],
                            figsize=(8*scale,9*scale))
#axes = [axes]
Wd_mean = [WD_mean0,WD_mean1]
Ws_mean = [WS_mean0,WS_mean1]
lw = 2
zorder = 10000
for ax,WD_mean,WS_mean in zip(axes,Wd_mean,Ws_mean):
    
    wd = WD_mean.values    
    ws = WS_mean.values    
    
    ax.contourf(wd, ws,
                bins    = range(0,24,3),  # speed bins
                nsector = 36,
                cmap    = cm.viridis,
                normed  = True)

    xtcklab = ax.get_xticklabels()
    ax.set_xticklabels(xtcklab,
                       fontsize=15,
                       position=(0,-0.18),
                       weight='bold',
                       color=(0.5,0.5,0.5,0.8))

    ''' percentile values '''
#    # add 5deg to get mid of bin
#    med_wdir = get_wdir_perc(ax,50) + 5
#    above_text = '{}\n(50%)'.format(int(med_wdir))    
#    theta = np.array([-med_wdir+90,-med_wdir+90])*np.pi/180.
#    ax.plot(theta,[0,10], color='r', lw=lw, zorder=zorder)
#    ax.text(0.58,-0.12,above_text,color='r',weight='bold',
#                 transform=ax.transAxes)

    ''' adjust frequency axis '''
    ax.set_ylim([0,12])
    for ax in axes:
        ax.set_radii_angle(angle=45)
        ytks = ax.get_yticks()
        newlabs = [str(int(t)) for t in ytks]
        newlabs[-1] = newlabs[-1]+'%'
        ax.set_yticks(ytks)
        ax.set_yticklabels(newlabs,
                           weight='bold',
                           color=(0.5,0.5,0.5,0.8))

axes[1].set_xticklabels('')
axes[1].set_yticklabels('')

txt = '13-season layer-mean wind'
txt +=' {:d}-{:d}m'.format(bot_layer,top_layer)
axes[0].text(0.5,1.05,txt,
            weight='bold',fontsize=15,
            ha='center',
            transform=axes[0].transAxes)       

txt = 'All profiles '
axes[0].text(1.1,0.5,txt,
            weight='bold',fontsize=16,
            rotation=-90,
            va='center',
            transform=axes[0].transAxes) 

txt = 'Rain CZD $\geq$ 0.25 mm'
axes[1].text(1.1,0.5,txt,
            weight='bold',fontsize=15,
            rotation=-90,
            va='center',
            transform=axes[1].transAxes) 
    
''' add legend '''
axes[1].legend(loc=(-0.2,-0.4), ncol=3)
axes[1].text(0.5,-0.15,'wind speed [$m\,s^{-1}$]',
              fontsize=15,ha='center',
              transform=axes[1].transAxes)

#plt.show()


fname='/home/raul/Desktop/fig_windrose_layer_0-500.png'
#fname='/Users/raulv/Desktop/windrose_layer_0-500.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
