# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:27:13 2016

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

try:
    outnr
except NameError:
    outnr = cfad(year=[1998]+range(2001,2013),
                   wdsurf=130,
                   wdwpro=None,
                   rainbb=None,
                   raincz=None,
                   nhours=2)  

try:
    outwr
except NameError:    
    outwr = cfad(year=[1998]+range(2001,2013),
                   wdsurf=130,
                   wdwpro=None,
                   rainbb=None,
                   raincz=0.25,
                   nhours=2)
      
        
''' creates plot with seaborn style '''
with sns.axes_style("white"):
    sns.set_style('ticks',
              {'xtick.direction': u'in',
               'ytick.direction': u'in'}
              )    
    
    scale=1.3
    fig = plt.figure(figsize=(8.5*scale,11*scale))
    

    gsA = gridspec.GridSpec(2, 1,
                            hspace=0.2)

    gsAA = gssp(2, 1,
                subplot_spec=gsA[0],
                hspace=0.25)

    gsAB = gssp(2, 1,
                subplot_spec=gsA[1],
                hspace=0.25)
 
    gs00 = gssp(1, 3,
                subplot_spec=gsAA[0],
                wspace=0.15
                )
    
    gs01 = gssp(1, 3,
                subplot_spec=gsAA[1],
                wspace=0.15
                )

    gs02 = gssp(1, 3,
                subplot_spec=gsAB[0],
                wspace=0.15
                )
    
    gs03 = gssp(1, 3,
                subplot_spec=gsAB[1],
                wspace=0.15
                )   

    
    ax00 = plt.subplot(gs00[0],gid='(a)')
    ax01 = plt.subplot(gs00[1],gid='(b)')
    ax02 = plt.subplot(gs00[2],gid='(c)')
    ax03 = plt.subplot(gs01[0],gid='(d)')
    ax04 = plt.subplot(gs01[1],gid='(e)')
    ax05 = plt.subplot(gs01[2],gid='(f)')

    ax06 = plt.subplot(gs02[0],gid='(g)')
    ax07 = plt.subplot(gs02[1],gid='(h)')
    ax08 = plt.subplot(gs02[2],gid='(i)')
    ax09 = plt.subplot(gs03[0],gid='(j)')
    ax10 = plt.subplot(gs03[1],gid='(k)')
    ax11 = plt.subplot(gs03[2],gid='(l)')


axes = [ax00,ax01,ax02,ax03,ax04,ax05,
        ax06,ax07,ax08,ax09,ax10,ax11]



plot=outnr.plot('u',axes=[ax00,ax01,ax02],
                add_median=True,add_title=False,
                add_cbar=False,show=False,
                top_altitude=2000)

outnr.plot('v',axes=[ax03,ax04,ax05],
               add_median=True,add_title=False,
               add_cbar=False,show=False,subax_label=False,
                top_altitude=2000)

outwr.plot('u',axes=[ax06,ax07,ax08],
               add_median=True,add_title=False,
               add_cbar=False, subax_label=True, show=False,
                top_altitude=2000)

outwr.plot('v',axes=[ax09,ax10,ax11],
               add_median=True,add_title=False,
               add_cbar=False,show=False,subax_label=False,
                top_altitude=2000)


add_floating_colorbar(fig=fig,im=plot['im'],
                      loc='bottom',
                      position=[0.25,0.08,0.5,0.3],
                      label='Normalized frequency [%]')

target = (ax02,ax05,ax08,ax11)
text = ('All profiles','','Rain at CZD','')
for ax,tx in zip(target,text):
    ax.text(1.05,0.0,tx,ha='center',va='center',
            fontsize=15,weight='bold',rotation=-90,
            transform=ax.transAxes)

ax01.set_yticklabels('')
ax02.set_yticklabels('')
ax04.set_yticklabels('')
ax05.set_yticklabels('')
ax07.set_yticklabels('')
ax08.set_yticklabels('')
ax10.set_yticklabels('')
ax11.set_yticklabels('')


for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=18,weight='bold',
            transform=ax.transAxes)

#plt.show()

#fname='/home/raul/Desktop/cfad_windprof_wdsurf_0-2000.png'
fname='/Users/raulv/Desktop/cfad_windprof_wdsurf130_1h_0-2000.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')

