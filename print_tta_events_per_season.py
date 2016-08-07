# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:19:45 2016

@author: raul
"""

import numpy as np
import pandas as pd
from tta_analysis import tta_analysis
from datetime import timedelta

years = tuple([1998]+range(2001,2013))

params = (
          dict(wdir_surf=125,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=125,wdir_wprof=170,
               rain_bby=None,rain_czd=None,nhours=1),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=2),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=4),
#          dict(wdir_surf=125,wdir_wprof=170,
#               rain_bby=None,rain_czd=None,nhours=8)
         )

n_ttas = {
          'nhours1':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},
#          'nhours2':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
#                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},
#          'nhours4':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
#                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},
#          'nhours8':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
#                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},

          }

' defines jump between ttas '          
h = timedelta(hours=1)

for p in params:
    print p    
    n_events = 0
    n_hours = 0
    r_czd = 0
    r_bby = 0
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
        n_events += ev
        n_hours += ho
        r_czd += cz
        r_bby += bb


    nh = str(p['nhours'])
    if p['rain_czd'] is None:
        prefix = 'n_'
    else:
        prefix = 'w_'
        
    n_ttas['nhours'+nh][prefix + 'events'] = n_events
    n_ttas['nhours'+nh][prefix + 'hours']  = n_hours
    n_ttas['nhours'+nh][prefix + 'czd']    = r_czd
    n_ttas['nhours'+nh][prefix + 'bby']    = r_bby


print n_ttas



