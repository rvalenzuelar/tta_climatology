# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:59:03 2016

@author: raul
"""

import numpy as np
from tta_analysis2 import tta_analysis

years = [1998]+range(2001,2013)

#params = [{ 'wdir_surf':  a,
#            'wdir_wprof': None,
#            'rain_czd':   0.25,
#            'nhours':     1
#           } for a in range(100,190,10)]
                
params = [{ 'wdir_surf':  '[{},{}['.format(a,a+10),
            'wdir_wprof': None,
            'rain_czd':   0.25,
            'nhours':     1
           } for a in range(85,275,10)]

#params = [{ 'wdir_surf':  None,
#            'wdir_wprof': '[{},{}['.format(a,a+10),
#            'wprof_gate': 0,
#            'rain_czd':   0.25,
#            'nhours':     1
#           } for a in range(85,275,10)]

#params = [{ 'wdir_surf':  None,
#            'wdir_wprof': '[{},{}['.format(a,a+10),
#            'wprof_gate': 1,
#            'rain_czd':   0.25,
#            'nhours':     1
#           } for a in range(85,275,10)]
           
#params = [{ 'wdir_surf':  None,
#            'wdir_wprof': '[{},{}['.format(a,a+10),
#            'wprof_gate': 4,
#            'rain_czd':   0.25,
#            'nhours':     1
#           } for a in range(85,275,10)]                   

#params = [{ 'wdir_surf':  None,
#            'wdir_wprof': '[{},{}['.format(a,a+10),
#            'wprof_gate': 10,
#            'rain_czd':   0.25,
#            'nhours':     1
#           } for a in range(85,275,10)]
 

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
ratio = list()
rr_czd = list()
rr_bby = list()
try:
    ngate = params[0]['wprof_gate']
except KeyError:
    ngate = ''
    
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
        cols = ['mnrain','mnhours','Wd_Surf  ',
                'Wd_wprof{}'.format(ngate),
                'TTczd','TTbby','ratio','hours',
                'NTczd','NTbby','ratio','hours']
        header = '{:>7} '*len(cols)
        print(header.format(*cols))
        first = False

    
    col =  '{:7.2f} {:7d} {:9} {:>7} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f} '
    col += '{:7.2f} {:7.2f} {:7.2f} {:7.0f}'

    p = params[nparam]

    print(col.format(p['rain_czd'],
                     p['nhours'],
                     p['wdir_surf'],
                     p['wdir_wprof'],
                     czd_tta,bby_tta,tta_ratio,tta_hours,
                     czd_notta,bby_notta,notta_ratio,notta_hours))

    ratio.append(tta_ratio)
    rr_czd.append(czd_tta)
    rr_bby.append(bby_tta)









