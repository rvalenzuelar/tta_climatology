# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:05:31 2016

@author: raulv
"""

import matplotlib.pyplot as plt
import collections
import ast
import numpy as np
from rv_utilities import discrete_cmap
from matplotlib import rcParams
from curve_fitting import curv_fit

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['axes.labelpad'] = 0.1
rcParams['mathtext.default'] = 'sf'

results = collections.OrderedDict()
       

txt = open('ratio_rr_persector_results.txt','read')
lines = txt.readlines()
join  = ''.join(lines).replace('\n', '')
dictline = join.split(';')
for dl in dictline:
    line = ast.literal_eval(dl)
    results[line.keys()[0]] = line.values()[0]
txt.close()

x = range(90, 280, 10)
k = 'Surf-500m'

out_rto = curv_fit(x=x, y=results[k]['ratio'],
                   model='4PL')

out_czd = curv_fit(x=x, y=results[k]['TTczd'],
                   model='gaussian')
                
out_bby = curv_fit(x=x, y=results[k]['TTbby'],
                   model='gaussian')
                
la = out_rto.params['la'].value
gr = out_rto.params['gr'].value
ce = out_rto.params['ce'].value
ua = out_rto.params['ua'].value
tx = 'bot_asym:    {:2.1f}\ngrowth_rate:{:2.1f}\n'
tx += 'center:         {:2.1f}\nupp_asym:   {:2.1f}'
tx_rto = tx.format(la,gr,ce,ua)

mu_czd = out_czd.params['center'].value
si_czd = out_czd.params['sigma'].value
tx_czd = '$\mu$:{:2.1f}\n$\sigma$:{:2.1f}'.format(mu_czd,si_czd)

mu_bby = out_bby.params['center'].value
si_bby = out_bby.params['sigma'].value
tx_bby = '$\mu$:{:2.1f}\n$\sigma$:{:2.1f}'.format(mu_bby,si_bby)

xnew = np.array(range(90,280,1))

scale = 1.4
fig,axes = plt.subplots(2,1,
                        figsize=(6*scale, 8*scale),
                        sharex=True
                        )
axes = axes.flatten()


cmap = discrete_cmap(7, base_cmap='Set1')
lw = 2

ynew_bby = out_bby.eval(x=xnew)
ynew_czd = out_czd.eval(x=xnew)
ynew_rto = out_rto.eval(x=xnew)

rsq_bby = out_bby.R_sq
rsq_czd = out_czd.R_sq
rsq_rto = out_rto.R_sq

axes[0].plot(x,results[k]['TTczd'],'o',
             color=cmap(1),label='CZD rain')
axes[0].plot(x,results[k]['TTbby'],'o',
             color=cmap(2),label='BBY rain')
axes[0].plot(xnew,ynew_czd,lw=lw,
             color=cmap(1),
             label='Gaussian fit (R-sq: {:2.2f})'.format(rsq_czd))
axes[0].plot(xnew,ynew_bby,lw=lw,
             color=cmap(2),
             label='Gaussian fit (R-sq: {:2.2f})'.format(rsq_bby))
axes[0].legend(numpoints=1,loc=2)

axes[0].text(200,5,tx_czd,fontsize=15,color=cmap(1))
axes[0].text(240,5,tx_bby,fontsize=15,color=cmap(2))

axes[1].plot(x,results[k]['ratio'],'o',
        lw=lw,color=cmap(0),label='CZD/BBY ratio')
axes[1].plot(xnew,ynew_rto,lw=lw,
             color=cmap(0),
             label='Logistic fit (R-sq: {:2.2f})'.format(rsq_rto))
axes[1].legend(numpoints=1,loc=2)
axes[1].text(200,5.8,tx_rto,fontsize=15,color=cmap(0),va='top')


axes[0].set_xticks(range(90,280,30))
axes[0].set_xlim([88,272])
axes[0].set_ylabel('rain rate $[mm h^{-1}]$')
axes[0].set_ylim([0,6])
axes[0].grid()

axes[1].set_ylim([0,6])
axes[1].set_ylabel('ratio')
axes[1].set_xlabel('wind direction')
axes[1].grid()

tx  = '13-season relationship between CZD, BBY rain\n'
tx += 'and wind direction over BBY in the layer-mean {}'.format(k)
plt.suptitle(tx,fontsize=15,weight='bold',y=0.96)

plt.subplots_adjust(hspace=0.05)

plt.show()

# #fname='/home/raul/Desktop/relationship_rain_wd.png'
# fname='/Users/raulv/Desktop/relationship_rain_wd.png'
# plt.savefig(fname, dpi=300, format='png',papertype='letter',
#             bbox_inches='tight')
