# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 12:06:26 2016

@author: raulv

Reads matfiles from surface obs (NOAA/ESRL/PSD)
and save pandas dataframes in HDF5 files.

"""

import os
import pandas as pd
from rv_utilities import datenum_to_datetime
from parse_data import quality_control, get_statistical
from scipy.io import loadmat

base_dir = os.environ['TTA_PATH']
bbypath = os.environ['TTA_PATH']+'/SURFACE/climatology/BBY{}_Sfcmet'

years = [1998]+range(2001,2013)

for year in years:
    print(year)
    fname = bbypath.format(str(year)[-2:])
    mat = loadmat(fname)['Sfcmet_bby']
    
    date, wspd, wdir, precip = [], [], [], []
    
    hourly = True
    
    for n in range(mat.size):
        dt = datenum_to_datetime(mat['dayt'][0][n][0][0])
    
        ''' converts to Timestamp '''
        date.append(pd.to_datetime(dt))
    
        wspd.append(mat['wspd'][0][n][0][0])
        wdir.append(mat['wdir'][0][n][0][0])
        precip.append(mat['precip'][0][n][0][0])
        
    d = {'wspd': wspd, 'wdir': wdir, 'precip': precip}
    dframe = pd.DataFrame(data=d, index=date)
    
    if year == 2001:
        '2001 has weird values beginning the mat file'
        dframe = dframe.ix[67:]
        
    dframe = quality_control(dframe)
    
    if hourly:
       dframe = get_statistical(dframe, minutes=60)
    
        
    store = pd.HDFStore(fname+'.h5')
    store.put('dframe',dframe)
    store.close()