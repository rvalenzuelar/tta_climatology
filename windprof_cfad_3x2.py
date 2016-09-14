# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:26:14 2016

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

#params = [dict(
#              wdsurf = '[175,185[',
#              wdwpro = None,
#              rainbb = None,
#              raincz = 0.25,
#              nhours = 1
#              )
#          ]

params = [{ 'wdsurf':  '[{},{}['.format(a,a+10),
            'wdwpro': None,
            'raincz':   0.25,
            'nhours':     1
           } for a in range(185,275,10)]

for par in params:

#    try:
#        outwr
#    except NameError:    
#        outwr = cfad(year=[1998]+range(2001,2013),
#                     **par)
          
    outwr = cfad(year=[1998]+range(2001,2013),
                 **par)
            
    ''' creates plot with seaborn style '''
    with sns.axes_style("white"):
        sns.set_style('ticks',
                  {'xtick.direction': u'in',
                   'ytick.direction': u'in'}
                  )    
        
        scale=1.3
        fig = plt.figure(figsize=(6*scale,8*scale))
        
    
        gsA = gridspec.GridSpec(1, 1,
                               )
    
        gsAA = gssp(1, 2,
                    subplot_spec=gsA[0],
                    wspace=0.05
                    )
    
        gs00 = gssp(3, 1,
                    subplot_spec=gsAA[0],
                    hspace=0.1,
                    )
        
        gs01 = gssp(3, 1,
                    subplot_spec=gsAA[1],
                    hspace=0.1,
                    )
    
    
        ax00 = plt.subplot(gs00[0],gid='(a)')
        ax01 = plt.subplot(gs01[0],gid='(b)')
        ax02 = plt.subplot(gs00[1],gid='(c)')
        ax03 = plt.subplot(gs01[1],gid='(d)')
        ax04 = plt.subplot(gs00[2],gid='(e)')
        ax05 = plt.subplot(gs01[2],gid='(f)')
    
    
    plot = outwr.plot('u',axes=[ax00,ax02,ax04],
                   add_median=True,add_title=False,
                   add_cbar=False, show=False,subax_label=False,
                   top_altitude=2000)
    
    outwr.plot('v',axes=[ax01,ax03,ax05],
                   add_median=True,add_title=False,
                   add_cbar=False,show=False,subax_label=True,
                   orientation='vertical',
                   top_altitude=2000)
    
    
    add_floating_colorbar(fig=fig,im=plot['im'],
                          loc='bottom',
                          position=[0.25,0.06,0.5,0.3],
                          label='Normalized frequency [%]')
    
    tx = ('U-comp')
    ax00.text(.5,1.05,tx,ha='center',va='center',
            fontsize=15,weight='bold',
            transform=ax00.transAxes)
    tx = ('V-comp')
    ax01.text(.5,1.05,tx,ha='center',va='center',
            fontsize=15,weight='bold',
            transform=ax01.transAxes)
    
    ax01.set_ylabel('')
    ax01.set_yticklabels('')
    ax03.set_yticklabels('')
    ax05.set_yticklabels('')
    
    ax00.set_xticklabels('')
    ax01.set_xticklabels('')
    ax02.set_xticklabels('')
    ax02.set_xlabel('')
    ax03.set_xticklabels('')
    ax03.set_xlabel('')
    ax04.set_xlabel('wind speed [ms-1]')
    ax05.set_xlabel('wind speed [ms-1]')
    
    
    
    axes = [ax00,ax01,ax02,ax03,ax04,ax05]
    
    for ax in axes:
        ax.text(0.05,0.9,ax.get_gid(),size=18,weight='bold',
                transform=ax.transAxes)
    
#    plt.show()
    
    sr = par['wdsurf'][1:-1].replace(',','-')
    wp = par['wdwpro']
    nh = par['nhours']
    name = 'cfad_wprof_rczd_{}-{}-{}.png'.format(sr,wp,nh)
    fname='/home/raul/Desktop/'+name
    #fname='/Users/raulv/Desktop/'+name
    plt.savefig(fname, dpi=150, format='png',papertype='letter',
                bbox_inches='tight')
