# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:18:51 2016

@author: raul
"""
import matplotlib.pyplot as plt
import collections
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['axes.labelpad'] = 0.1

results = collections.OrderedDict()

results['Surf'] = {
                    'TTczd':[1.38,1.30,1.46,1.78,2.44,3.36,3.98,4.62,
                             4.73,5.23,4.25,3.43,3.14,2.43,1.75,1.81,
                             1.29,1.38,1.08],
                    'TTbby':[0.94,0.81,1.08,1.22,1.25,1.08,1.04,1.15,
                             1.13,1.41,1.22,1.17,1.26,0.87,0.54,0.57,
                             0.43,0.59,0.31],
                    'ratio':[1.48,1.61,1.35,1.46,1.95,3.13,3.82,4.02,
                             4.18,3.70,3.49,2.93,2.50,2.80,3.27,3.18,
                             3.02,2.32,3.47]        
                   }
results['160m']= {
                'TTczd':[0.91,1.3,1.48,1.63,1.74,1.94,3.0,3.6,
                       3.92,3.75,3.45,2.74,3.26,2.88,1.93,
                       1.41,1.29,1.23,1.17],
                'TTbby':[0.52,0.86,0.98,1.24,1.19,1.02,1.08,
                       1.00,0.95,1.15,0.79,1.03,1.02,0.74,
                       0.58,0.51,0.48,0.34,0.36],
                'ratio':[1.75,1.51,1.50,1.32,1.47,1.90,2.77,
                       3.61,4.10,3.28,4.35,2.67,3.20,3.87,
                       3.31,2.80,2.69,3.63,3.23]
                   }
results['252m']={
                 'TTczd':[1.08,0.86,1.46,1.64,1.62,2.09,2.73,
                          3.23,3.88,3.73,3.42,2.89,2.98,2.66,
                          2.42,1.53,1.28,1.32,1.22],
                 'TTbby':[0.68,0.54,0.98,1.21,1.09,1.35,1.10,
                          1.01,0.91,1.10,0.82,0.87,1.02,0.72,
                          0.68,0.60,0.45,0.43,0.38],
                 'ratio':[1.58,1.60,1.49,1.35,1.49,1.55,2.48,
                          3.21,4.25,3.41,4.17,3.30,2.91,3.67,
                          3.55,2.54,2.85,3.03,3.22]                   
                }
results['528m']={
                'TTczd':[1.03,1.14,0.77,1.12,1.53,1.98,2.46,2.88,
                         3.09,3.39,3.69,3.29,2.99,2.56,2.34,2.16,
                         1.61,1.51,1.24],
                'TTbby':[0.79,0.64,0.60,0.68,0.96,1.39,1.48,1.32,
                         0.88,0.96,0.99,1.03,0.72,0.71,0.82,0.68,
                         0.71,0.54,0.35],
                'ratio':[1.30,1.80,1.30,1.65,1.60,1.43,1.67,2.18,
                         3.51,3.55,3.74,3.19,4.17,3.58,2.87,3.18,
                         2.28,2.81,3.57]        
        
                }
results['1081m']={
                'TTczd':[1.27,1.91,0.78,0.85,1.44,1.04,1.86,2.37,
                         2.69,2.74,3.25,3.52,3.23,2.97,2.93,2.11,
                         2.19,1.81,1.41],
                'TTbby':[0.30,0.08,0.60,0.48,0.74,0.64,1.36,1.39,
                         1.30,1.03,0.95,0.95,0.95,1.01,0.88,0.76,
                         0.75,0.58,0.44],
                'ratio':[4.23,22.50,1.29,1.76,1.94,1.64,1.37,1.71,
                         2.06,2.67,3.42,3.69,3.42,2.95,3.31,2.79,
                         2.92,3.14,3.18],        
                }

results['Surf-500m']={
                'TTczd':[0.88,0.85,1.47,1.57,1.48,1.87,2.66,
                         3.27,3.71,3.83,3.43,2.97,3.07,2.67,
                         2.29,1.55,1.24,1.23,1.07],
                'TTbby':[0.77,0.56,0.87,1.18,1.15,1.22,1.13,
                         1.04,0.81,1.12,0.91,0.87,0.94,0.84,
                         0.77,0.54,0.45,0.35,0.39],
                'ratio':[1.14,1.52,1.68,1.33,1.29,1.53,2.35,
                         3.13,4.56,3.41,3.79,3.42,3.25,3.17,
                         2.96,2.89,2.76,3.53,2.72]                         
                }
         
         
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

#plt.show()

fname='/home/raul/Desktop/fig_rainfallratio_persector.png'
#fname='/Users/raulv/Desktop/fig_windprof_panels.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')



