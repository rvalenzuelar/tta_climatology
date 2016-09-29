# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:09:25 2016

@author: raul
"""
import tta_analysis3 as tta


# years = [1998]
years = [1998]+range(2001,2013)
template = '{:^6} - {:^20} - {:^20} - {:^11} - {:^11}'
print(template.format('Season','Beg','End','all-hours', 'rainy-hours'))
all_thours = 0
rai_thours = 0

for y in years:

    out = tta.start(years=[y], layer=[0,500],verbose=False)
    precip_all = out['precip_good']
    precip_rainy = precip_all[precip_all.czd >0.25]

    year = precip_all.index[-1].year
    beg = precip_all.index[0].strftime('%H%M UTC %d %b %Y')
    end = precip_all.index[-1].strftime('%H%M UTC %d %b %Y')
    all_hours = precip_all.index.size
    rai_hours = precip_rainy.index.size
    print(template.format(str(year),
                          beg,
                          end,
                          str(all_hours),
                          str(rai_hours)))
    all_thours += all_hours
    rai_thours += rai_hours

print('Total all_hours: {}'.format(all_thours))
print('Total rainy_hours: {}'.format(rai_thours))
