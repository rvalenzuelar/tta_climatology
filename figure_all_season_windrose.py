# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:58:08 2016

@author: raul
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tta_analysis import tta_analysis 
from rv_windrose import WindroseAxes
#from rv_utilities import add_floating_colorbar
from matplotlib import rcParams


rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['ytick.color'] = (0.8,0.8,0.8)
rcParams['figure.facecolor']='w'

try:
    wdsrf_all
except NameError:           
    wdsrf_all      = list()
    wdsrf_rain_czd = list()
    wdsrf_rain_bby = list()
    wdsrf_both = list()
    
    wd160_all      = list()
    wd160_rain_czd = list()
    wd160_rain_bby = list()
    wd160_both = list()
    
    wssrf_all      = list()
    wssrf_rain_czd = list()
    wssrf_rain_bby = list()
    wssrf_both = list()
     
    ws160_all      = list()
    ws160_rain_czd = list()
    ws160_rain_bby = list()
    ws160_both = list()
    
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
            
        rain_czd = tta.df.rczd>0
        rain_bby = tta.df.rbby>0
        both     = (tta.df.rczd>0) & (tta.df.rbby>0)

        wdsrf = tta.df.wdsrf      
        wssrf = tta.df.wssrf      
        wdwpr = tta.df.wdwpr
        wswpr = tta.df.wswpr
        
        wdsrf_all.extend(wdsrf.values.astype(float))
        wdsrf_rain_czd.extend(wdsrf[rain_czd].values.astype(float))
        wdsrf_rain_bby.extend(wdsrf[rain_bby].values.astype(float))
        wdsrf_both.extend(wdsrf[both].values.astype(float))
     
        wd160_all.extend(wdwpr.values.astype(float))
        wd160_rain_czd.extend(wdwpr[rain_czd].values.astype(float))
        wd160_rain_bby.extend(wdwpr[rain_bby].values.astype(float))
        wd160_both.extend(wdwpr[both].values.astype(float))
        
        wssrf_all.extend(wssrf.values.astype(float))
        wssrf_rain_czd.extend(wssrf[rain_czd].values.astype(float))
        wssrf_rain_bby.extend(wssrf[rain_bby].values.astype(float))
        wssrf_both.extend(wssrf[both].values.astype(float))
     
        ws160_all.extend(wswpr.values.astype(float))
        ws160_rain_czd.extend(wswpr[rain_czd].values.astype(float))
        ws160_rain_bby.extend(wswpr[rain_bby].values.astype(float))
        ws160_both.extend(wswpr[both].values.astype(float))


wd_tuple = (
            wd160_all     , 
            wd160_rain_czd, 
#            wd160_rain_bby,
#            wd160_both,
            wdsrf_all, 
            wdsrf_rain_czd, 
#            wdsrf_rain_bby,
#            wdsrf_both
            )

ws_tuple = (
            ws160_all     , 
            ws160_rain_czd, 
#            ws160_rain_bby,
#            ws160_both,
            wssrf_all, 
            wssrf_rain_czd, 
#            wssrf_rain_bby, 
#            wssrf_both, 
            ) 

''' start plot ''' 

scale=0.7
axes = WindroseAxes.from_ax(subplots=(2,len(wd_tuple)/2),
                            figsize=(12*scale,10*scale))

for ax,wd,ws in zip(axes,wd_tuple,ws_tuple):        
    ax.contourf(wd, ws,
                bins=range(0,24,3),  # speed bins
                nsector=36,
                cmap=cm.plasma,
                normed=True)

''' add legend '''
xpos={2:-0.2, 3:-0.7, 4:0.6}
axes[2].legend(loc=(xpos[len(wd_tuple)/2],-0.5),
               ncol=4)


''' label annotations '''
vpos = 1.2
top_labels = ['All','Rain at CZD','Rain at BBY']
for i, lab in zip([0,1],top_labels*2):
    axes[i].text(0.5,vpos,lab,ha='center',va='center',
               fontsize=16,weight='bold',
               transform=axes[i].transAxes)

hpos = 1.2
side_labels = ('160-m','Surface')
for i, lab in zip([1,3],side_labels):
    axes[i].text(hpos,0.5,lab,ha='center',va='center',
               rotation=-90,fontsize=16,weight='bold',
               transform=axes[i].transAxes) 

panel_labels = ('(a)','(b)','(c)','(d)')
for i, lab in zip([0,1,2,3],panel_labels):
    axes[i].text(0.0,1.0,lab,ha='center',va='center',
               fontsize=15,weight='bold',
               transform=axes[i].transAxes) 

''' add azimuthal lines '''
lw = 2
zorder = 10000
th = 170    
theta=np.array([-th+90,-th+90])*np.pi/180.
axes[1].plot(theta,[0,8], color='r', lw=lw, zorder=zorder)
axes[1].text(0.58,-0.12,'170\n(50%)',color='r',weight='bold',
             transform=axes[1].transAxes)
th = 130
theta=np.array([-th+90,-th+90])*np.pi/180.
axes[3].plot(theta,[0,8], color='r', lw=lw, zorder=zorder)
axes[3].text(0.87,0.05,'130\n(33%)',color='r',weight='bold',
             transform=axes[3].transAxes)

for ax in axes:
    ax.set_radii_angle(angle=45)
    ytks = ax.get_yticks()
    newlabs = ['']*len(ytks)   
    newlabs = [str(int(t)) for t in ytks[1::2]]
    newtcks = ytks[1::2]
    ax.set_yticks(newtcks)
    ax.set_yticklabels(newlabs)
    
#plt.show()

fname='/home/raul/Desktop/fig_all_season_windrose.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')


''' percentile analysis '''
def get_wdir_perc(axes,perc):
    info=axes._info
    table=info['table']
    wdir=info['dir']
    tsum=table.sum(axis=0)
    tcsum=tsum.cumsum()
    return wdir[np.where(tcsum<=perc)[0][-1]]
    
get_wdir_perc(axes[1],50)
get_wdir_perc(axes[3],33)
    