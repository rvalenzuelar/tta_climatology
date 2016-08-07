# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:47:31 2016

@author: raulv
"""

from tta_analysis import tta_analysis 

for y in [1998]+range(2001,2013):

    
    tta = tta_analysis(y)
    
    ''' parameters do not really matter for 
    wdir statistics because we are using the wdir
    column only '''
    tta.start_df(wdir_surf=150,wdir_wprof=150,
                 rain_bby=None,rain_czd=None,nhours=1)    

    ''' wdsrf is wdir at bby '''
#    wdir_rain = tta.df.wdsrf[tta.df.rbby>0]
    wdir_rain = tta.df.wdsrf[tta.df.rczd>0]

    median = wdir_rain.median()
    mean = wdir_rain.mean()
    mode = wdir_rain.mode().values
    modeMin = mode[0]
    modeMax = mode[-1]    
    
    template = 'year:{}, median:{:5.1f}, modeMin:{:5.1f}, '
    template += 'modeMax:{:5.1f}, mean:{:5.1f}'
    print template.format(y,median,modeMin,modeMax,mean)

