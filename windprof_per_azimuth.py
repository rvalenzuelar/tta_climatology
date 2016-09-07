# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:54:04 2016

@author: raul
"""

import parse_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from rv_windrose import WindroseAxes
from rv_utilities import discrete_cmap


# if seaborn-style plot shows up need 
# to use:
# %matplotlib inline
import matplotlib as mpl
inline_rc = dict(mpl.rcParams)
mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['mathtext.default'] = 'sf'


def sin(arg):
    return np.sin(np.radians(arg))
    
def cos(arg):
    return np.cos(np.radians(arg))


#years = [1998]
years = [1998]+range(2001,2013)


#max_hgt_gate = 15  # 1450 [m]
#max_hgt_gate = 21  # 2000 [m]
max_hgt_gate = 40  # 3750 [m] max top

try:
    WS
except NameError:

    WD = [pd.DataFrame(),pd.DataFrame()]
    WS = [pd.DataFrame(),pd.DataFrame()]
    
#    WD = pd.DataFrame()
#    WS = pd.DataFrame()
    
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
        

        ''' creates columns for each level '''
        for h in hgt[:max_hgt_gate]:
            col = '{:2.0f}'.format(h)
            
            ''' weird pandas bug doesnt allow 
                2001 as column name '''
            if col =='2001': 
                col='2000'
            
            wd[col]=np.zeros(wd.index.size)        
            ws[col]=np.zeros(ws.index.size)
            wd_rain[col]=np.zeros(rain_dates.size)        
            ws_rain[col]=np.zeros(rain_dates.size)  
        
        for n, [s,d] in enumerate(zip(wspd,wdir)):
            d = np.array(d)
            s = np.array(s)
            wd.iloc[n,1:] = d
            ws.iloc[n,1:] = s

        for n, [sr,dr] in enumerate(zip(wspd_rain, wdir_rain)):
            wd_rain.iloc[n,1:] = np.array(dr[:max_hgt_gate])
            ws_rain.iloc[n,1:] = np.array(sr[:max_hgt_gate])

        WD[0] = WD[0].append(wd)
        WS[0] = WS[0].append(ws)
        WD[1] = WD[1].append(wd_rain)
        WS[1] = WS[1].append(ws_rain)

#        WD = WD.append(wd_rain)
#        WS = WS.append(ws_rain)

#        WD = WD.append(wd)
#        WS = WS.append(ws)        


''' component analysis '''
wind_flow_mean = [list(),list()]        
#flow_dir = range(90,190,10)
#flow_dir = range(180,280,10) 
flow_dir = [90,140,180]

for n in range(2):
    
    WD_sin = WD[n].applymap(lambda x: sin(x))
    WD_cos = WD[n].applymap(lambda x: cos(x))
    
    U_df = -1*WS[n].multiply(WD_sin)
    V_df = -1*WS[n].multiply(WD_cos)
    
    for fdir in flow_dir:
        if fdir >= 180:
            wind_flow = -(U_df*sin(fdir)+V_df*cos(fdir))
        else:
            wind_flow = U_df*sin(fdir)+V_df*cos(fdir)            
        wind_flow_mean[n].append(wind_flow.mean())

    
cmap = discrete_cmap(len(flow_dir), base_cmap='Set2')
colors = [cmap(n) for n in range(len(flow_dir))]

dz = np.array([160]+[92]*(max_hgt_gate-1))

y = np.array([0])
y = np.append(y,hgt[:max_hgt_gate])
ydz = y[:-1]+(y[1:]-y[:-1])/2.

fig,axes = plt.subplots(2,1,figsize=(8,12),sharey=True)
lw=3
for wf,wd,color in zip(wind_flow_mean[0],flow_dir,colors):
    x = wf.values
    
    f = interp1d(y,x)
    ynew = np.linspace(0,int(y.max()),100)
    xnew = f(ynew)    
    
    axes[0].plot(xnew,ynew,label=str(wd),
            color=color,lw=lw)

    axes[0].text(x[-1], hgt[max_hgt_gate-1]+30, str(wd),
                fontsize=14,
                ha='center',
                weight='bold')

    ''' fill jet '''
#    cond1 = np.where(xnew<=xnew[0])[0]
#    cond2 = np.where(xnew>xnew[0])[0]
#    if cond1.size>0 and cond2.size>0:        
#        jet = xnew[cond1]    
#        axes[0].fill_betweenx(ynew,xnew,x2=jet.max(),
#                              where=xnew<jet.max(),
#                              color=color,
#                              alpha=0.5)

    
    dx = x[1:]-x[:-1]
    dxdz = dx/dz
    
    axes[1].plot(dxdz,ydz,label=str(wd),color=color,lw=lw)   

#    from scipy.interpolate import spline
#    ynew = np.linspace(ydz.min(),int(ydz.max()),100)
#    dxdz_smooth = spline(dxdz,ydz,ynew)
#    axes[1].plot(dxdz_smooth,ynew,label=str(wd),color=color,lw=lw)   
    
    axes[0].set_ylim([0,int(hgt[max_hgt_gate-1])+200])        
    
axes[0].set_xlim([-12,14])
#axes[0].set_xlim([-20,5])

axes[0].grid()
axes[1].grid()

tx = '13-season wind profile per direction component'
plt.suptitle(tx,fontsize=15,weight='bold',y=0.95)

plt.show()


#fname='/home/raul/Desktop/fig_windrose_layer_0-500.png'
##fname='/Users/raulv/Desktop/windrose_layer_0-500.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
