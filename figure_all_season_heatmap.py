# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:38:30 2016

@author: raul
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:58:08 2016

@author: raul
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tta_analysis import tta_analysis 
from rv_utilities import add_floating_colorbar
from matplotlib import rcParams
from rv_utilities import discrete_cmap

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['ytick.color'] = (0.8,0.8,0.8)

try:
    wdsrf_all
except NameError:           
    wdsrf_all      = list()
    wdsrf_rain_czd = list()
    wdsrf_rain_bby = list()
    
    wd160_all      = list()
    wd160_rain_czd = list()
    wd160_rain_bby = list()
    
#    wssrf_all      = list()
#    wssrf_rain_czd = list()
#    wssrf_rain_bby = list()
#     
#    ws160_all      = list()
#    ws160_rain_czd = list()
#    ws160_rain_bby = list()
    
    years = [1998]+range(2001,2013)
#    years = [1998]
    
    for y in years:
    
        print(y)
        
        tta = tta_analysis(y)
        
        ''' parameters do not really matter for 
        wdir statistics because we are using the wdir
        column only '''
        tta.start_df(wdir_surf=150,wdir_wprof=150,
                     rain_bby=None,rain_czd=None,nhours=1)      
    
        ''' wdsrf is wdir at surface and wdwpr is first
            gate of wind profiler '''
        wdsrf_all.extend(tta.df.wdsrf.values.astype(float))
        wdsrf_rain_czd.extend(tta.df.wdsrf[tta.df.rczd>0].values.astype(float))
        wdsrf_rain_bby.extend(tta.df.wdsrf[tta.df.rbby>0].values.astype(float))
     
        wd160_all.extend(tta.df.wdwpr.values.astype(float))
        wd160_rain_czd.extend(tta.df.wdwpr[tta.df.rczd>0].values.astype(float))
        wd160_rain_bby.extend(tta.df.wdwpr[tta.df.rbby>0].values.astype(float))
    
#        wssrf_all.extend(tta.df.wssrf.values.astype(float))
#        wssrf_rain_czd.extend(tta.df.wssrf[tta.df.rczd>0].values.astype(float))
#        wssrf_rain_bby.extend(tta.df.wssrf[tta.df.rbby>0].values.astype(float))
#     
#        ws160_all.extend(tta.df.wswpr.values.astype(float))
#        ws160_rain_czd.extend(tta.df.wswpr[tta.df.rczd>0].values.astype(float))
#        ws160_rain_bby.extend(tta.df.wswpr[tta.df.rbby>0].values.astype(float))

''' convert lists to numpy array '''
wdsrf_all      = np.array(wdsrf_all)
wdsrf_rain_czd = np.array(wdsrf_rain_czd)
wdsrf_rain_bby = np.array(wdsrf_rain_bby)
wd160_all      = np.array(wd160_all)
wd160_rain_czd = np.array(wd160_rain_czd)
wd160_rain_bby = np.array(wd160_rain_bby)

''' get nans '''
nans_all      = (np.isnan(wdsrf_all)      | np.isnan(wd160_all))
nans_rain_czd = (np.isnan(wdsrf_rain_czd) | np.isnan(wd160_rain_czd))
nans_rain_bby = (np.isnan(wdsrf_rain_bby) | np.isnan(wd160_rain_bby))

''' make 2d histograms
    The bi-dimensional histogram of samples x and y.
    Values in x are histogrammed along the first
    dimension and values in y are histogrammed along
    the second dimension.
'''

normed = True
x,y = [wdsrf_all[~nans_all],
       wd160_all[~nans_all]]
H_all,xed,yed = np.histogram2d(x, y,
                               bins=range(0,360,10),
                               normed=normed)

x,y = [wdsrf_rain_czd[~nans_rain_czd],
       wd160_rain_czd[~nans_rain_czd]]
H_rain_czd,xed,yed = np.histogram2d(x, y,
                                    bins=range(0,360,10),
                                    normed=normed)

x,y = [wdsrf_rain_bby[~nans_rain_bby],
       wd160_rain_bby[~nans_rain_bby]]
H_rain_bby,xed,yed = np.histogram2d(x, y,
                                    bins=range(0,360,10),
                                    normed=normed)

if normed is True:
    H_all      *= 1e5
    H_rain_czd *= 1e5
    H_rain_bby *= 1e5

''' make grid '''
X,Y = np.meshgrid(xed[:-1],yed[:-1])

''' make plot '''
sns.set_style("whitegrid")

#fig,ax = plt.subplots(1,2,figsize=(10,5),sharey=True,sharex=True)
#
#cmap = cm.get_cmap('plasma')
#v = np.arange(5,65,5)
#
#im1 = ax[0].contourf(X,Y,H_all,v,cmap=cmap)
#im2 = ax[1].contourf(X,Y,H_rain_czd,v,cmap=cmap)
##im3 = ax[2].contourf(X,Y,H_rain_bby,v,cmap=cmap)
#
#add_floating_colorbar(fig=fig,im=im2,
#                      position=[0.25,-0.05,0.5,0.8],
#                      loc='bottom',
#                      label='Normalized frequency [%]')
#
#
#ax[1].hlines(130,0,360,color='k')
#ax[1].vlines(170,0,360,color='k')
#
#ax[0].set_xticks(range(0,360,60))
#ax[0].set_yticks(range(0,360,60))
#ax[0].set_xlim([0,360])
#ax[0].set_ylim([0,360])
#
#''' some labels '''
#va = 'bottom'
#ha = 'center'
#ax[0].text(180,360,'All',va=va,ha=ha,fontsize=15)
#ax[0].text(340,-45,'wdir 160-m',fontsize=15)
#ax[1].text(180,360,'Rain at CZD',va=va,ha=ha,fontsize=15)
##ax[2].text(180,360,'Rain at BBY',va=va,ha=ha,fontsize=15)
#
#ax[0].set_ylabel('wdir surface')


fig,ax = plt.subplots(1,1,figsize=(6,6),sharey=True,sharex=True)

v = np.arange(4,34,4)
im = ax.contourf(X,Y,H_rain_czd,v,
                 cmap=cm.get_cmap('plasma'))

''' add colorbar '''
add_floating_colorbar(fig=fig,im=im,
                      position=[0.25,-0.05,0.5,0.8],
                      loc='bottom',
                      label='Normalized frequency [%]')

''' add lines '''
cmap = discrete_cmap(7, base_cmap='Set1')
lim_surf = 130
lim_160m = 170
fsize = 15
color = cmap(1)
ax.hlines(lim_surf,0,360,color=color)
ax.text(0,lim_surf,str(lim_surf),
        fontsize=fsize,color=color,
        weight='bold')
ax.vlines(lim_160m,0,360,color=color)
ax.text(lim_160m,0,str(lim_160m),
        fontsize=fsize,color=color,
        weight='bold')

''' ranges '''
ax.set_xticks(range(0,360,60))
ax.set_yticks(range(0,360,60))
ax.set_xlim([0,360])
ax.set_ylim([0,360])

''' labels '''
va = 'bottom'
ha = 'center'
ax.text(180,360,'Rain at CZD',va=va,ha=ha,fontsize=15)
ax.set_ylabel('wdir surface')
ax.set_xlabel('wdir 160-m')


plt.show()

#template = '/home/raul/Desktop/fig_hist2d_{}-{}.png'
#fname=template.format(str(lim_surf).zfill(3),str(lim_160m))
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')

    