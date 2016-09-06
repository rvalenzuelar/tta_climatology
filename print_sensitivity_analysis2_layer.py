# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:25:52 2016

@author: raul
"""

import numpy as np
from tta_analysis2 import tta_analysis

years = [1998]+range(2001,2013)

''' creates one param per wd sector '''               
params_wsec = [{ 'wdir_thres':  '[{},{}['.format(a,a+10),
                'wdir_layer': [0,500],
                'rain_czd':   0.25,
                'nhours':     1
              } for a in range(85,275,10)]

''' creates one param per nhour '''
params_nh = [{ 'wdir_thres': 150,
                'wdir_layer': [0,500],
                'rain_czd':   0.25,
                'nhours':     a
              } for a in [1,2,4,8]]

''' creates one param per wdir_thres '''
params_wth = [{ 'wdir_thres': a,
                'wdir_layer': [0,500],
                'rain_czd':   0.25,
                'nhours':     2
              } for a in [140,160]]

params = params_nh + params_wth

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
                tta.start_df_layer(**p)
                results[n] = tta.print_stats(return_results=True,
                                             skip_print=True)
            first = False
        else:
            for n,p in enumerate(params):
                tta.start_df_layer(**p)
                r = tta.print_stats(return_results=True,
                                    skip_print=True)        
                
                results[n] = np.vstack((results[n],r))
        

first = True
ratio = list()
rr_czd = list()
rr_bby = list()

layer = params[0]['wdir_layer']
    
for nparam,result in results.iteritems():

    tta_hours = result[:,8].sum()
    bby_tta = np.nansum(result[:,1])/tta_hours
    czd_tta = np.nansum(result[:,4])/tta_hours
    tta_ratio = czd_tta/bby_tta
    
    notta_hours = result[:,9].sum()
    bby_notta = np.nansum(result[:,2])/notta_hours
    czd_notta = np.nansum(result[:,5])/notta_hours
    notta_ratio = czd_notta/bby_notta
    
    if first:
        print('layer {}-{}m'.format(layer[0],layer[1]))
        cols = ['mnrain','mnhours','Wd_Thres  ',
                'TTczd','TTbby','ratio','hours',
                'NTczd','NTbby','ratio','hours']
        header = '{:>7} '*len(cols)
        print(header.format(*cols))
        first = False

    
    col =  '{:7.2f} {:7d} {:9} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f}'

    p = params[nparam]

    print(col.format(p['rain_czd'],
                     p['nhours'],
                     p['wdir_thres'],
#                     p['wdir_wprof'],
                     czd_tta,bby_tta,tta_ratio,tta_hours,
                     czd_notta,bby_notta,notta_ratio,
                     notta_hours))

    ratio.append(tta_ratio)
    rr_czd.append(czd_tta)
    rr_bby.append(bby_tta)









