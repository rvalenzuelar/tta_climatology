# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:55:04 2016

@author: raulv
"""

#import numpy as np
from tta_analysis2 import tta_analysis

years = [1998]+range(2001,2013)

params = [
          dict(wdir_surf=100,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=110,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=120,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=130,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=140,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=150,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=160,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=170,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=180,wdir_wprof=None,rain_czd=0.25,nhours=1),
         ]


try:
    results
except NameError:

    results = {nparam:{
                      'bby':{'total':(),'hours':()},
                      'czd':{'total':(),'hours':()}
                      } for nparam in range(len(params))
              }

    for n,p in enumerate(params):
        print(p)
        pbby = 0
        pczd = 0
        hbby = 0
        hczd = 0
        for y in years:
            print(y)
            tta=tta_analysis(y)     
            tta.start_df(**p)
            pbby += tta.df[tta.df.tta].rbby.sum()
            pczd += tta.df[tta.df.tta].rczd.sum()
            hbby += tta.df[tta.df.tta].rbby.count()
            hczd += tta.df[tta.df.tta].rczd.count()
            
        print[n,pbby,pczd,hbby,hczd]
        results[n]['bby']['total'] = pbby            
        results[n]['czd']['total'] = pczd
        results[n]['bby']['hours'] = hbby            
        results[n]['czd']['hours'] = hczd
        
first = True
for nparam,result in results.iteritems():

    
    if first:
        cols = ['mnrain','mnhours','Wd_Surf','Wd_160m |',
                'CZD-total','CZD-rr |',
                'BBY-total','BBY-rr |',
                'Total-ratio',
                'RR-ratio',
                'Hours',
                ]
        header = '{:>7} '*len(cols)
        print(header.format(*cols))
        first = False

    
    col =  '{:>7} {:7d} {:7d} {:>7} |'
    col += '{:9.0f} {:7.2f} |' 
    col += '{:9.0f} {:7.2f} |'
    col += '{:11.2f} {:8.2f} {:6.0f}'

    p = params[nparam]
    bby_total = result['bby']['total']
    bby_hours = result['bby']['hours']
    czd_total = result['czd']['total']
    czd_hours = result['czd']['hours']
    
    czd_rr = czd_total/float(czd_hours)
    bby_rr = bby_total/float(bby_hours)
    
    print(col.format(p['rain_czd'],
                     p['nhours'],
                     p['wdir_surf'],
                     p['wdir_wprof'],
                     czd_total,
                     czd_rr,
                     bby_total,
                     bby_rr,
                     czd_total/bby_total,
                     czd_rr/bby_rr,
                     bby_hours
                    )
        )



