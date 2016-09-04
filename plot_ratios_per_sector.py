# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:18:51 2016

@author: raul
"""
import matplotlib.pyplot as plt
import collections
import ast
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['axes.labelpad'] = 0.1

results = collections.OrderedDict()
       

txt = open('ratio_rr_persector_results.txt','read')
lines = txt.readlines()
join  = ''.join(lines).replace('\n','')
dictline = join.split(';')
for dl in dictline:
    line = ast.literal_eval(dl)
    results[line.keys()[0]]=line.values()[0]

         
x = range(90,280,10)       

scale = 1.4
fig,axes = plt.subplots(3,2,
                        figsize=(8*scale,8*scale),
                        sharex=True,sharey=True)
axes=axes.flatten()
#axes[-1].remove()


cmap = discrete_cmap(7, base_cmap='Set1')
lw = 2

for ax,k in zip(axes,results.keys()):
    ax.plot(x,results[k]['ratio'],'o-',
            lw=lw,color=cmap(0),label='Ratio')
    ax.plot(x,results[k]['TTczd'],'o-',
            lw=lw,color=cmap(1),label='RR-czd')
    ax.plot(x,results[k]['TTbby'],'o-',
            lw=lw,color=cmap(2),label='RR-bby')
    ax.text(120,5.5,k,weight='bold',fontsize=15)

    ax.set_xticks(range(90,280,30))
    ax.set_ylim([0,6])
    ax.set_xlim([88,272])
    ax.grid()

    if k == '1081m':
        ax.annotate('22.5',
                xy         = (99, 6),
                xytext     = (120,4.5),
                xycoords   = 'data',
                textcoords = 'data',
                zorder     = 10000,
                color      = cmap(0),
                weight     = 'bold',
                fontsize   = 15,
                arrowprops=dict(arrowstyle = '-|>',
                                ec         = 'k',
                                fc         = 'k',
                                )
                )          
    
    if k == '252m':
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        ax.set_ylabel('rainfall ratio (CZD/BBY) and rain rate [mm h-1]')
    
    if k == 'Surf':
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)        
        ax.legend(loc=0,numpoints=1)
        
    if k == '0-500m':
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
    
    
    
plt.subplots_adjust(wspace=0.05,hspace=0.05)
tl = '13-season rainfall ratio and rain rate per wind direction sector\n'
tl += '(bins of 10 degrees)'
plt.suptitle(tl, fontsize=15,weight='bold',va='top',y=0.95)
plt.xlabel('wind direction (bin center)',x=0.)

plt.show()

#fname='/home/raul/Desktop/fig_rainfallratio_persector.png'
##fname='/Users/raulv/Desktop/fig_windprof_panels.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')



