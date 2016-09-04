# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 12:55:47 2016

@author: raulv
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

shear = np.sqrt(dUdz_mean**2+dVdz_mean**2)

scale = 1.3
fig,axes = plt.subplots(figsize=(8*scale,8*scale))
axes = [axes]

y=np.array([0])
y=np.append(y,hgt[:15])
ydz = np.append([0], y[:-1]+(y[1:]-y[:-1])/2.)

pos = (0.9,0.9)

nclr = 0

axes[0].plot(shear,y,color=cmap(nclr),label='Mean')
#axes[0].plot(U_mean-U_std,y,'--',color=cmap(nclr),label='Std Dev')
#axes[0].plot(U_mean+U_std,y,'--',color=cmap(nclr))




#plt.subplots_adjust(hspace=0.3)

if rain_dates is None:
    tx = '13-season wind-component profile at BBY'
else:
    tx  = '13-season wind-component profile at BBY with rain CZD '
    tx += '$\geq $ 0.25  mm'
plt.suptitle(tx, fontsize=15,weight='bold',y=0.95)

plt.show()