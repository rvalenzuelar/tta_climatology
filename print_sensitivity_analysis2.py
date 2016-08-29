# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:59:03 2016

@author: raul
"""

import numpy as np
from tta_analysis2 import tta_analysis

years = [1998]+range(2001,2013)

params = [
#          dict(wdir_surf=100,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=110,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=120,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=130,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=140,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=150,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=160,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=170,wdir_wprof=None,rain_czd=0.25,nhours=1),
#          dict(wdir_surf=180,wdir_wprof=None,rain_czd=0.25,nhours=1),

#          dict(wdir_surf=100,wdir_wprof=None,rain_czd=0.25,nhours=2),          
#          dict(wdir_surf=110,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=120,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=130,wdir_wprof=None,rain_czd=0.25,nhours=2),         
#          dict(wdir_surf=140,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=150,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=160,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=170,wdir_wprof=None,rain_czd=0.25,nhours=2),
#          dict(wdir_surf=180,wdir_wprof=None,rain_czd=0.25,nhours=2),

#          dict(wdir_surf=100,wdir_wprof=None,rain_czd=0.25,nhours=4),                 
#          dict(wdir_surf=110,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=120,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=130,wdir_wprof=None,rain_czd=0.25,nhours=4),         
#          dict(wdir_surf=140,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=150,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=160,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=170,wdir_wprof=None,rain_czd=0.25,nhours=4),
#          dict(wdir_surf=180,wdir_wprof=None,rain_czd=0.25,nhours=4),

          dict(wdir_surf=100,wdir_wprof=None,rain_czd=0.25,nhours=8), 
          dict(wdir_surf=110,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=120,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=130,wdir_wprof=None,rain_czd=0.25,nhours=8),         
          dict(wdir_surf=140,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=150,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=160,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=170,wdir_wprof=None,rain_czd=0.25,nhours=8),
          dict(wdir_surf=180,wdir_wprof=None,rain_czd=0.25,nhours=8),
         ]

try:
    results
except NameError:
    results = {nparam:list() for nparam in range(len(params))}
    first = True
    for y in years:
        print y
        tta=tta_analysis(y)
        if first:    
            for n,p in enumerate(params):
                tta.start_df(**p)
                results[n] = tta.print_stats(return_results=True,
                                             skip_print=True)
            first = False
        else:
            for n,p in enumerate(params):
                tta.start_df(**p)
                r = tta.print_stats(return_results=True,
                                    skip_print=True)        
                
                results[n] = np.vstack((results[n],r))
        

first = True
for nparam,result in results.iteritems():

    tta_hours = result[:,8].sum()
    bby_tta = result[:,1].sum()/tta_hours
    czd_tta = result[:,4].sum()/tta_hours
    tta_ratio = czd_tta/bby_tta
    
    notta_hours = result[:,9].sum()
    bby_notta = result[:,2].sum()/notta_hours
    czd_notta = result[:,5].sum()/notta_hours
    notta_ratio = czd_notta/bby_notta
    
    if first:
        cols = ['mnrain','mnhours','Wd_Surf','Wd_160m',
                'TTczd','TTbby','ratio','hours',
                'NTczd','NTbby','ratio','hours']
        header = '{:>7} '*len(cols)
        print(header.format(*cols))
        first = False

    
    col =  '{:7.2f} {:7d} {:7d} {:>7} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f}'

    p = params[nparam]

    print(col.format(p['rain_czd'],
                     p['nhours'],
                     p['wdir_surf'],
                     p['wdir_wprof'],
                     czd_tta,bby_tta,tta_ratio,tta_hours,
                     czd_notta,bby_notta,notta_ratio,notta_hours))











