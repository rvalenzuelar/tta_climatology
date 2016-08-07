# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:59:32 2016

@author: raulv
"""

import numpy as np
import matplotlib.pyplot as plt
from tta_analysis import tta_analysis 
from rv_utilities import add_colorbar

bins = range(0,370,10)
first = True
freq_array = np.zeros((1,36))

for y in [1998]+range(2001,2013):

    print(y)
    
    tta = tta_analysis(y)
    
    ''' parameters do not really matter for 
    wdir statistics because we are using the wdir
    column only '''
    tta.start_df(wdir_surf=150,wdir_wprof=150,
                 rain_bby=None,rain_czd=None,nhours=1)      

    ''' wdsrf is wdir at bby '''
#    wdsrf = tta.df.wdsrf
    wdsrf = tta.df.wdsrf[tta.df.rczd>0]
    
    wdsrf = wdsrf.values.astype(float)
    wdsrf = wdsrf[~np.isnan(wdsrf)]

    ''' histogram '''
    freq,_ = np.histogram(wdsrf,bins=bins)
    freq_norm = freq/float(freq.sum())
    freq_norm = np.expand_dims(freq_norm,axis=0)
   
    if first is True:
        freq_array = freq_norm
        first = False
    else:
        freq_array = np.vstack((freq_array,freq_norm))
        
        
fig,ax = plt.subplots()
x = np.array(bins[:-1])
y = np.array([1998]+range(2001,2013))
im = ax.pcolormesh(x,y,freq_array)  
add_colorbar(ax,im)
plt.show()
    