# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:14:22 2016

@author: raul
"""
import parse_data
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from rv_windrose import WindroseAxes
from rv_utilities import pandas2stack, discrete_cmap

sns.set_style('whitegrid')

# if seaborn-style plot shows up need 
# to use:
# sns.reset_defaults()
# %matplotlib inline

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['mathtext.default'] = 'sf'

cmap = discrete_cmap(7, base_cmap='Set1')

#years = [1998]
years = [1998]+range(2001,2013)

try:
    WS
except NameError:
#    ws = {th:list() for th in target_hgts}
#    wd = {th:list() for th in target_hgts}
#    wdsrf = list()
    
    WS = pd.DataFrame()
    WD = pd.DataFrame()
    
    for year in years:
          
        czd = parse_data.surface('czd', year=year)        
        bby = parse_data.surface('bby', year=year)
        wpr = parse_data.windprof(year=year)
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

        wspd = wpr.dframe.loc[first:last].wspd
        wdir = wpr.dframe.loc[first:last].wdir
        czd = czd.dframe.loc[first:last]
        bby = bby.dframe.loc[first:last]
        
        ''' select rainy days '''
        rain_czd = czd.precip > 0
        rain_dates = rain_czd.loc[rain_czd.values].index
#        rain_dates = None
            
        if rain_dates is None:
            wd = pd.DataFrame(index=wspd.index,columns=range(16))
            ws = pd.DataFrame(index=wspd.index,columns=range(16))
            
            wd.iloc[:,0] = bby.wdir
            ws.iloc[:,0] = bby.wspd
            
            wdir = pandas2stack(wdir).T
            wspd = pandas2stack(wspd).T
            wd.iloc[:,1:]=np.squeeze(wdir[:,:15])
            ws.iloc[:,1:]=np.squeeze(wspd[:,:15])
            
        else:
            
            wd = pd.DataFrame(index=wspd.loc[rain_dates].index,
                              columns=range(16))
            ws = pd.DataFrame(index=wspd.loc[rain_dates].index,
                              columns=range(16))
            
            wd.iloc[:,0] = bby.wdir.loc[rain_dates]
            ws.iloc[:,0] = bby.wspd.loc[rain_dates]
            
            wdir = pandas2stack(wdir.loc[rain_dates]).T
            wspd = pandas2stack(wspd.loc[rain_dates]).T
            wd.iloc[:,1:]=np.squeeze(wdir[:,:15])
            ws.iloc[:,1:]=np.squeeze(wspd[:,:15])

        WS = WS.append(ws)
        WD = WD.append(wd)
        
WD_sine = WD.applymap(lambda x: np.sin(np.radians(x)))
WD_cosn = WD.applymap(lambda x: np.cos(np.radians(x)))

U = -1*WS.multiply(WD_sine)
V = -1*WS.multiply(WD_cosn)

U_mean = U.mean()
V_mean = V.mean()

U_std = U.std()
V_std = V.std()

dU=U.diff(axis=1)
dV=V.diff(axis=1)

dz = np.array([np.nan,160]+[92]*14)

dUdz = dU.apply(lambda x:x/dz,axis=1)
dVdz = dV.apply(lambda x:x/dz,axis=1)

dUdz_mean = dUdz.mean().values*1e3
dVdz_mean = dVdz.mean().values*1e3

dUdz_std = dUdz.std().values*1e3
dVdz_std = dVdz.std().values*1e3

scale = 1.3
fig,axes = plt.subplots(2,2,sharey=True,
                        figsize=(8*scale,8*scale))
axes = axes.flatten()

y=np.array([0])
y=np.append(y,hgt[:15])
ydz = np.append([0], y[:-1]+(y[1:]-y[:-1])/2.)

pos = (0.9,0.9)

nclr = 0

axes[0].plot(U_mean,y,color=cmap(nclr),label='Mean')
axes[0].plot(U_mean-U_std,y,'--',color=cmap(nclr),label='Std Dev')
axes[0].plot(U_mean+U_std,y,'--',color=cmap(nclr))
axes[0].set_xlim([-10,15])
axes[0].set_xlabel('$[m\,s^{-1}]$')
axes[0].set_ylabel('Altitude MSL [m]')
axes[0].text(pos[0],pos[1],r'$U$',
            fontsize=20,
            transform=axes[0].transAxes)
axes[0].legend(bbox_to_anchor=[0.8,1.1],ncol=2,fontsize=12)

axes[1].plot(dUdz_mean,ydz,color=cmap(nclr))
axes[1].plot(dUdz_mean+dUdz_std,ydz,'--',color=cmap(nclr))
axes[1].plot(dUdz_mean-dUdz_std,ydz,'--',color=cmap(nclr))
axes[1].set_xlim([-20,25])
axes[1].text(pos[0],pos[1],r'$\frac{dU}{dZ}$',
            fontsize=25,
            transform=axes[1].transAxes)
axes[1].set_xlabel('$[x1e^{-3}\,s^{-1}]$')

axes[2].plot(V_mean,y,color=cmap(nclr+1),label='Mean')
axes[2].plot(V_mean-V_std,y,'--',color=cmap(nclr+1),label='Std Dev')
axes[2].plot(V_mean+V_std,y,'--',color=cmap(nclr+1))
axes[2].set_xlim([-8,20])
axes[2].set_xlabel('$[m\,s^{-1}]$')
axes[2].text(pos[0],pos[1],r'$V$',
            fontsize=20,
            transform=axes[2].transAxes)
#axes[2].legend(loc=4,fontsize=12)
axes[2].legend(bbox_to_anchor=[0.8,1.1],ncol=2,fontsize=12)

axes[3].plot(dVdz_mean,ydz,color=cmap(nclr+1))
axes[3].plot(dVdz_mean+dVdz_std,ydz,'--',color=cmap(nclr+1))
axes[3].plot(dVdz_mean-dVdz_std,ydz,'--',color=cmap(nclr+1))
axes[3].set_xlim([-30,40])
axes[3].text(pos[0],pos[1],r'$\frac{dV}{dZ}$',
            fontsize=25,
            transform=axes[3].transAxes)
axes[3].set_xlabel('$[x1e^{-3}\,s^{-1}]$')

plt.subplots_adjust(hspace=0.3)

if rain_dates is None:
    tx = '13-season wind-component profile at BBY'
else:
    tx  = '13-season wind-component profile at BBY with rain CZD '
    tx += '$\geq $ 0.25  mm'
plt.suptitle(tx, fontsize=15,weight='bold',y=0.95)

plt.show()

#fname='/home/raul/Desktop/wind-comp_profile_rczd.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')