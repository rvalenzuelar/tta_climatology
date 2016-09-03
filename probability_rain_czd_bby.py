# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:03:52 2016

@author: raul
"""

import matplotlib.pyplot as plt
import numpy as np
import parse_data
from matplotlib.colors import LogNorm
from rv_utilities import discrete_cmap

import matplotlib as mpl
#inline_rc = dict(mpl.rcParams)
#mpl.rcParams.update(inline_rc)

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['mathtext.default'] = 'sf'

years = [1998]+range(2001,2013)
xlist = list()
ylist = list()

for year in years:

    czd=parse_data.surface('czd', year=year)
    bby=parse_data.surface('bby', year=year)

    first_bby = bby.dframe.index[0]
    first_czd = czd.dframe.index[0]

    last_bby = bby.dframe.index[-1]
    last_czd = czd.dframe.index[-1]
    
    ''' start and end are the same '''
    first = max(first_bby,first_czd)   
    last  = min(last_bby,last_czd)
    x = bby.dframe.loc[first:last].precip.values.astype(float)
    y = czd.dframe.loc[first:last].precip.values.astype(float)

    ''' remove nans '''
    isnan = np.isnan(x) | np.isnan(y)
    x = x[~isnan]
    y = y[~isnan]

    ''' filters '''
    bothzero = (x==0) & (y==0)
    x = x[~bothzero]    
    y = y[~bothzero]

    czdzero = (y==0)
    x = x[~czdzero]    
    y = y[~czdzero]

#    bbyzero = (x==0)
#    x = x[~bbyzero]    
#    y = y[~bbyzero]
    
    
    xlist.extend(x)
    ylist.extend(y)

bby = np.array(xlist)
czd = np.array(ylist)

diff  = czd-bby
ratio = czd/bby

#hist, bin_edges = np.histogram(a, density=True)
bin_res = 0.254
n_bins = 11  # around the zero bin
min_bin = -bin_res*n_bins-(bin_res/2.)
max_bin = bin_res*(n_bins+1)+(bin_res/2.)
bins=np.arange(min_bin,max_bin,bin_res)

#out = plt.hist(diff,bins=bins,normed=False)

fig,axes = plt.subplots(2,1,figsize=(8,10))

freq,edges = np.histogram(diff,bins=bins)
axes[0].bar(edges[:-1]+0.01,freq/float(freq.sum()),
            width=bin_res)
axes[0].set_ylabel('Probability')
axes[0].set_xlabel('CZD-BBY')

''' finit values (bby>0) '''
ratio = ratio[~np.isinf(ratio)]
binw = 1
freq,edges = np.histogram(ratio,
                          bins=np.arange(0,11,binw),
                          )
axes[1].bar(edges[:-1]+0.01,freq/float(freq.sum()),
            width=binw)
axes[1].set_ylabel('Probability')
axes[1].set_xlabel('CZD/BBY')

tx  = 'Rain rates difference and ratio for hours \n'
tx += 'with CZD >0.25mm'
plt.suptitle(tx,ha='center',weight='bold',fontsize=15,y=0.95)

#plt.show()

fname='/home/raul/Desktop/rain_czd_bby_probability.png'
#fname='/Users/raulv/Desktop/rain_czd_bby_probability.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')
