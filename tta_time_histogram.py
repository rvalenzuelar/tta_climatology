# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:50:32 2016

@author: raul
"""

def tta_time_histogram(dates,max_hours=None):
    ''' dates is a pandas series 
        array containing tta dates '''
    from datetime import timedelta
    idx = dates.index.get_values()
    h = timedelta(hours=1)
    ''' initialize total dictionary '''
    totals = {}
    for t in range(1,max_hours+1):
        totals[t] = 0
    totals['more'] = 0
    nh = 1
    d0 = dates[idx[0]]
    last_date = dates[idx[-1]]
    for d in dates[1:]:
        if d == d0+h:
            nh += 1
            d0 = d
            if d == last_date:
                if nh>max_hours:
                    totals['more'] += 1
                else:
                    totals[nh] += 1
        else:
            if nh>max_hours:
                totals['more'] += 1
                d0 = d
                nh = 1                 
            else:
                if d == last_date:
                    totals[1] += 2
                else:
                    totals[nh] += 1
                    d0 = d
                    nh = 1 
    return totals
    