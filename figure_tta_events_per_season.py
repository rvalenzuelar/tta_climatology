# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:19:45 2016

@author: raul
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tta_analysis import tta_analysis
from datetime import timedelta
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15

sns.set_style("whitegrid")

years = tuple([1998]+range(2001,2013))

params = (
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=None,nhours=1),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=2),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=4),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=8)
         )

' defines jump between ttas '          
h = timedelta(hours=1)

try:
    n_ttas
except NameError:    
    n_ttas = {
              'nhours1':{'w_events':[],'w_hours':[],
                         'w_czd':[],'w_bby':[],
                         'n_events':[],'n_hours':[],
                         'n_czd':[],'n_bby':[]},
              }    
    for p in params:
        print p    
        n_events = list()
        n_hours = list()
        r_czd = list()
        r_bby = list()
        for y in years:
            tta = tta_analysis(y)
            tta.start_df(**p)
            s = pd.Series(tta.tta_dates)
            if s.size == 0:
                ev = 0
                ho = 0
                cz = 0
                bb = 0
            else:
                sdiff = s-s.shift()
                ev = np.sum( sdiff>h ) + 1
                ho = tta.tta_hours
                cz = tta.tta_rainfall_czd
                bb = tta.tta_rainfall_bby
            av_time = ho/float(ev)
            template = 'year:{:4d}, n_ttas:{:3d}, n_hours:{:4d}, ave_time:{:2.1f}, r_czd:{:3d}, r_bby:{:3d},'
            print template.format(y,ev,ho,av_time,cz,bb)        
            n_events.append(ev)
            n_hours.append(ho)
            r_czd.append(cz)
            r_bby.append(bb)

        nh = str(p['nhours'])
        if p['rain_czd'] is None:
            prefix = 'n_'
        else:
            prefix = 'w_'
            
        n_ttas['nhours'+nh][prefix + 'events'] = np.array(n_events)
        n_ttas['nhours'+nh][prefix + 'hours']  = np.array(n_hours)
        n_ttas['nhours'+nh][prefix + 'czd']    = np.array(r_czd)
        n_ttas['nhours'+nh][prefix + 'bby']    = np.array(r_bby)


nttas = n_ttas['nhours1']


labs = [str(int(np.mod(y,100.))).zfill(2) for y in years]
scale = 1.2
x = np.array(years)
width = 0.35
titles = (
          'Total number of TTA hours',
          'Number of TTA events (1 or more hours)',
          )
ys = [
      [nttas['n_hours'],nttas['w_hours']],
      [nttas['n_events'],nttas['w_events']],
      ]
xlabs = ('','winter season [year]')
ylabs = (
         'Hours',
         'Events',
         )
panels = ('(a)','(b)')
panel_pos = [1300,180]
nttas['w_hours'].mean()

avr = '$\overline{X}$ = '
std = '$\sigma$  = '

cmap = discrete_cmap(7, base_cmap='Set1')

fig,axes = plt.subplots(2,1,figsize=(8*scale,10*scale))
axes = axes.flatten()

grp = zip(axes,ys,titles,xlabs,ylabs,panels,panel_pos)

c = (cmap(1),cmap(0))

for ax,y,t,xlab,ylab,p,pos in grp:
    mean_n = y[0].mean()
    mean_w = y[1].mean()
    avr_n = avr+'{:2.0f}\n'.format(mean_n)
    avr_w = avr+'{:2.0f}\n'.format(mean_w)

    stdd_n = y[0].std()
    stdd_w = y[1].std()
    std_n = std+'{:2.0f}'.format(stdd_n)
    std_w = std+'{:2.0f}'.format(stdd_w)
    
    ax.bar(x,y[0],width,color=c[0],label='All obs')
    ax.bar(x + width,y[1],width,color=c[1],label='Rain at CZD')
    ax.plot([1997,2014],[mean_n]*2,color=c[0])
    ax.plot([1997,2014],[mean_w]*2,color=c[1])
    ax.text(1998,pos,p,weight='bold',fontsize=18, va='bottom')
    ax.text(1999,mean_n,avr_n,color=c[0],fontsize=15, va='bottom')
    ax.text(1999,mean_n,std_n,color=c[0],fontsize=15, va='bottom')
    ax.text(1999,mean_w,avr_w,color=c[1],fontsize=15, va='bottom')
    ax.text(1999,mean_w,std_w,color=c[1],fontsize=15, va='bottom')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labs)
    ax.xaxis.grid(False)
    ax.set_title(t,fontsize=15,weight='bold')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim([1997.8,2013])
    ax.legend(loc=[0.1,0.8],fontsize=15)
ax.legend_.remove()
plt.show()

#fname='/home/raul/Desktop/fig_events_per_season.png'
#plt.savefig(fname, dpi=150, format='png',papertype='letter',
#            bbox_inches='tight')



