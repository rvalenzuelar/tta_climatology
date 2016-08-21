# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:09:25 2016

@author: raul
"""
from tta_analysis import tta_analysis

#years=[1998]
years = [1998]+range(2001,2013)
template = '{:^6} - {:^20} - {:^20} - {:^4}'
print(template.format('Season','Beg','End','Hours'))
thours = 0
for y in years:
    tta = tta_analysis(y)
    tta.start_df(wdir_surf  = 130,
                 wdir_wprof = 170,
                 rain_bby   = None,
                 rain_czd   = 0.25,
                 nhours     = 1)
    year = tta.include_dates[-1].year
    beg = tta.include_dates[0].strftime('%H%M UTC %d %b %Y')
    end = tta.include_dates[-1].strftime('%H%M UTC %d %b %Y')
    hours = tta.count_hrs_include
    print(template.format(str(year),beg,end,str(hours)))
    thours += hours
print('Total hours: {}'.format(thours))
