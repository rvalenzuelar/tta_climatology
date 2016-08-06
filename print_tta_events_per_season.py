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
          'nhours2':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},
          'nhours4':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},
          'nhours8':{'w_events':[],'w_hours':[],'w_czd':[],'w_bby':[],
                     'n_events':[],'n_hours':[],'n_czd':[],'n_bby':[]},

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
            n_events += 0
            n_hours += 0
            r_czd += 0
            r_bby += 0
        else:
            sdiff = s-s.shift()
            n_events += np.sum( sdiff>h ) + 1
            n_hours += tta.tta_hours
            r_czd += tta.tta_rainfall_czd
            r_bby += tta.tta_rainfall_bby
        template = 'year:{},n_ttas:{},n_hours:{}'
        print template.format(y,n_events,n_hours)
    nh = str(p['nhours'])
    if p['rain_czd'] is None:
        n_ttas['nhours'+nh]['n_events'] = n_events
        n_ttas['nhours'+nh]['n_hours'] = n_hours
        n_ttas['nhours'+nh]['n_czd'] = r_czd
        n_ttas['nhours'+nh]['n_bby'] = r_bby
    else:
        n_ttas['nhours'+nh]['w_events'] = n_events
        n_ttas['nhours'+nh]['w_hours'] = n_hours
        n_ttas['nhours'+nh]['w_czd'] = r_czd
        n_ttas['nhours'+nh]['w_bby'] = r_bby

print n_ttas



