# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:59:32 2016

@author: raulv
"""

import numpy as np
import matplotlib.pyplot as plt
from tta_analysis import tta_analysis 
#from rv_utilities import add_colorbar
from rv_utilities import add_floating_colorbar
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15


bins = range(0,370,10)
first = True
freq_srf_all_array = np.zeros((1,36))
freq_srf_rain_czd_array = np.zeros((1,36))
freq_srf_rain_bby_array = np.zeros((1,36))

freq_160_all_array = np.zeros((1,36))
freq_160_rain_czd_array = np.zeros((1,36))
freq_160_rain_bby_array = np.zeros((1,36))

#years = [1998]+range(2001,2013)
years = [1998]

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
    wdsrf_all      = tta.df.wdsrf.values.astype(float)
    wdsrf_rain_czd = tta.df.wdsrf[tta.df.rczd>0].values.astype(float)
    wdsrf_rain_bby = tta.df.wdsrf[tta.df.rbby>0].values.astype(float)
 
    wd160_all      = tta.df.wdwpr.values.astype(float)
    wd160_rain_czd = tta.df.wdwpr[tta.df.rczd>0].values.astype(float)
    wd160_rain_bby = tta.df.wdwpr[tta.df.rbby>0].values.astype(float)


    ''' select non-nan values '''
    wdsrf_all      = wdsrf_all[~np.isnan(wdsrf_all)]
    wdsrf_rain_czd = wdsrf_rain_czd[~np.isnan(wdsrf_rain_czd)]
    wdsrf_rain_bby = wdsrf_rain_bby[~np.isnan(wdsrf_rain_bby)]

    wd160_all      = wd160_all[~np.isnan(wd160_all)]
    wd160_rain_czd = wd160_rain_czd[~np.isnan(wd160_rain_czd)]
    wd160_rain_bby = wd160_rain_bby[~np.isnan(wd160_rain_bby)]

    
    ''' surface histograms '''
    freq_srf_all,_    = np.histogram(wdsrf_all,bins=bins)
    freq_srf_all_norm = freq_srf_all/float(freq_srf_all.sum())
    freq_srf_all_norm = np.expand_dims(freq_srf_all_norm,axis=0)

    freq_srf_rain_czd,_    = np.histogram(wdsrf_rain_czd,bins=bins)
    freq_srf_rain_czd_norm = freq_srf_rain_czd/float(freq_srf_rain_czd.sum())
    freq_srf_rain_czd_norm = np.expand_dims(freq_srf_rain_czd_norm,axis=0)

    freq_srf_rain_bby,_    = np.histogram(wdsrf_rain_bby,bins=bins)
    freq_srf_rain_bby_norm = freq_srf_rain_bby/float(freq_srf_rain_bby.sum())
    freq_srf_rain_bby_norm = np.expand_dims(freq_srf_rain_bby_norm,axis=0)

    ''' 160m histograms '''
    freq_160_all,_    = np.histogram(wd160_all,bins=bins)
    freq_160_all_norm = freq_160_all/float(freq_160_all.sum())
    freq_160_all_norm = np.expand_dims(freq_160_all_norm,axis=0)

    freq_160_rain_czd,_    = np.histogram(wd160_rain_czd,bins=bins)
    freq_160_rain_czd_norm = freq_160_rain_czd/float(freq_160_rain_czd.sum())
    freq_160_rain_czd_norm = np.expand_dims(freq_160_rain_czd_norm,axis=0)

    freq_160_rain_bby,_    = np.histogram(wd160_rain_bby,bins=bins)
    freq_160_rain_bby_norm = freq_160_rain_bby/float(freq_160_rain_bby.sum())
    freq_160_rain_bby_norm = np.expand_dims(freq_160_rain_bby_norm,axis=0)

   
    if first is True:
        freq_srf_all_array      = freq_srf_all_norm
        freq_srf_rain_czd_array = freq_srf_rain_czd_norm
        freq_srf_rain_bby_array = freq_srf_rain_bby_norm

        freq_160_all_array      = freq_160_all_norm
        freq_160_rain_czd_array = freq_160_rain_czd_norm
        freq_160_rain_bby_array = freq_160_rain_bby_norm

        first = False
    else:
        freq_srf_all_array      = np.vstack((freq_srf_all_array     ,freq_srf_all_norm))
        freq_srf_rain_czd_array = np.vstack((freq_srf_rain_czd_array,freq_srf_rain_czd_norm))
        freq_srf_rain_bby_array = np.vstack((freq_srf_rain_bby_array,freq_srf_rain_bby_norm))

        freq_160_all_array      = np.vstack((freq_160_all_array     ,freq_160_all_norm))
        freq_160_rain_czd_array = np.vstack((freq_160_rain_czd_array,freq_160_rain_czd_norm))
        freq_160_rain_bby_array = np.vstack((freq_160_rain_bby_array,freq_160_rain_bby_norm))  


       
''' start plot ''' 
fig,ax = plt.subplots(2,3,figsize=(15,10),sharey=True)
ax=ax.flatten()
#x = np.array(bins[:-1])
x = np.array(np.arange(5,365,10))
y = np.array(range(len(years)))
X,Y = np.meshgrid(x,y)
countv = np.arange(0.02,0.22,0.02)
cmap='plasma'
im = ax[0].contourf(X,Y,freq_160_all_array,countv,cmap=cmap)  
im = ax[1].contourf(X,Y,freq_160_rain_czd_array,countv,cmap=cmap)
im = ax[2].contourf(X,Y,freq_160_rain_bby_array,countv,cmap=cmap)
im = ax[3].contourf(X,Y,freq_srf_all_array,countv,cmap=cmap)  
im = ax[4].contourf(X,Y,freq_srf_rain_czd_array,countv,cmap=cmap)
im = ax[5].contourf(X,Y,freq_srf_rain_bby_array,countv,cmap=cmap)    

ax[0].set_yticks(range(0,13))
labs = [str(yl) for yl in years]
ax[0].set_yticklabels(labs)
for i in range(6):
    ax[i].set_xticks(range(0,390,60))
    ax[i].set_xlim([0,360])
    
''' vertical lines '''
lcolor = (0.8,0.8,0.8)
lw = 3
for i,pos in zip([1,2,4,5],[170,170,125,125]):
    ax[i].vlines(pos,0,12,linestyle='--',color=lcolor,lw=lw)
    
''' colorbar '''
add_floating_colorbar(fig=fig,
                      im=im,
                      #         [left,bot,width,hgt]
                      position=[0.35,0.05,0.3,0.3],
                      ticklabels=['2%']+['']*4+['12%']+['']*3+['20%'],
                      loc='bottom',
                      label='Normalized frequency'
                      )

''' label annotations '''
vpos = 1.04
top_labels = ['All','Rain at CZD','Rain at BBY']
for i, lab in zip(range(6),top_labels*2):
    ax[i].text(0.5,vpos,lab,ha='center',va='center',
               fontsize=16,weight='bold',
               transform=ax[i].transAxes)
         
side_labels = ('160-m','Surface')
for i, lab in zip([2,5],side_labels):
    ax[i].text(1.05,0.5,lab,ha='center',va='center',
               rotation=-90,fontsize=16,weight='bold',
               transform=ax[i].transAxes)   

''' axes labels '''
ax[4].set_xlabel('Wind direction [deg]')
ax[0].set_ylabel('Winter season [year]')

plt.subplots_adjust(hspace=0.2,
                    wspace=0.1)
plt.show()
#fname='/home/raul/Desktop/wdir_seasonal_histogram.png'
#fname='/Users/raulv/Dropbox/AuthoredPapers/Long-term statistics/figures_v02/wdir_seasonal_histogram.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')
    