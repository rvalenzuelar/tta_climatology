# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:38:40 2016

@author: raul
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wprof_cfad import cfad
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from rv_utilities import add_floating_colorbar

from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['axes.labelpad'] = 0.1

params = dict(
              wdsurf = '[120,150]',
              wdwpro = None,
              rainbb = None,
              raincz = 0.25,
              nhours = 2
              )

try:
    outwr
except NameError:    
    outwr = cfad(year=[1998]+range(2001,2013),
                 **params)
      
        
''' creates plot with seaborn style '''
with sns.axes_style("white"):
    sns.set_style('ticks',
              {'xtick.direction': u'in',
               'ytick.direction': u'in'}
              )    
    
    scale=1.3
    fig = plt.figure(figsize=(8*scale,6*scale))
    

    gsA = gridspec.GridSpec(1, 1,
                            hspace=0.2)

    gsAA = gssp(2, 1,
                subplot_spec=gsA[0],
                hspace=0.25)

    gs00 = gssp(1, 3,
                subplot_spec=gsAA[0],
                wspace=0.15
                )
    
    gs01 = gssp(1, 3,
                subplot_spec=gsAA[1],
                wspace=0.15
                )


    ax00 = plt.subplot(gs00[0],gid='(a)')
    ax01 = plt.subplot(gs00[1],gid='(b)')
    ax02 = plt.subplot(gs00[2],gid='(c)')
    ax03 = plt.subplot(gs01[0],gid='(d)')
    ax04 = plt.subplot(gs01[1],gid='(e)')
    ax05 = plt.subplot(gs01[2],gid='(f)')


plot = outwr.plot('u',axes=[ax00,ax01,ax02],
               add_median=True,add_title=False,
               add_cbar=False, subax_label=True, show=False,
                top_altitude=2000)

outwr.plot('v',axes=[ax03,ax04,ax05],
               add_median=True,add_title=False,
               add_cbar=False,show=False,subax_label=False,
                top_altitude=2000)


add_floating_colorbar(fig=fig,im=plot['im'],
                      loc='bottom',
                      position=[0.25,0.06,0.5,0.3],
                      label='Normalized frequency [%]')

tx = ('Rain at CZD')
ax02.text(1.05,0.0,tx,ha='center',va='center',
        fontsize=15,weight='bold',rotation=-90,
        transform=ax02.transAxes)

ax01.set_yticklabels('')
ax02.set_yticklabels('')
ax04.set_yticklabels('')
ax05.set_yticklabels('')


axes = [ax00,ax01,ax02,ax03,ax04,ax05,
        ]

for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=18,weight='bold',
            transform=ax.transAxes)

#plt.show()

sr = '120-150'
wp = params['wdwpro']
nh = params['nhours']
name = 'cfad_wprof_rczd_{}-{}-{}.png'.format(sr,wp,nh)
fname='/home/raul/Desktop/'+name
#fname='/Users/raulv/Desktop/'+name
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')

