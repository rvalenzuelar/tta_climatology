# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:52:30 2016

@author: raul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tta_analysis import tta_analysis
from tta_time_histogram import tta_time_histogram

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

wrain = list()
nrain = list()
max_hours = 15 
for p in params:
    print p    
    for y in years:
        tta = tta_analysis(y)
        tta.start_df(**p)
        s = pd.Series(tta.tta_dates)
        out = tta_time_histogram(s,max_hours=max_hours)
        print out 
        if p['rain_czd'] is None:
            nrain.append(out)
        else:
            wrain.append(out)


''' initialize total dictionary '''
totals_wrain = {}
totals_nrain = {}
for t in range(1,max_hours+1):
    totals_wrain[t] = 0
    totals_nrain[t] = 0
totals_wrain['more'] = 0
totals_nrain['more'] = 0
for lst in wrain:
    for key,value in lst.iteritems():
        totals_wrain[key]+=value
for lst in nrain:
    for key,value in lst.iteritems():
        totals_nrain[key]+=value


sns.set_style("whitegrid")
labs=[str(k) for k in totals_wrain.keys()]
fig,ax = plt.subplots()
width = 0.3
x = np.arange(len(labs))
ax.bar(x,totals_nrain.values(),width,color='b',label='All obs')
ax.bar(x+width,totals_wrain.values(),width,color='r',label='Rain at CZD')
ax.set_xticks(x + width)
ax.set_xticklabels(labs)
ax.xaxis.grid(False)
ax.set_xlabel('TTA duration [h]')
ax.set_ylabel('Count')
ax.set_ylim([0,550])
plt.legend(loc=0,fontsize=15)

fname='/home/raul/Desktop/fig_tta_time_histogram.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')