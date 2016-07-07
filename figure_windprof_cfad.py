# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:27:13 2016

@author: raul
"""
import matplotlib.pyplot as plt
import wprof_cfad as cfad
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp


from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15

try:
    out
except NameError:
    out=cfad(year=[1998]+range(2001,2013),
             wdsurf=125,wdwpro=170,
             rainbb=None,raincz=0.25,
             nhours=2)
    
''' creates plot with seaborn style '''
with sns.axes_style("white"):
    scale=1.2
    plt.figure(figsize=(8.5*scale,11*scale))
    
    gs0 = gridspec.GridSpec(2, 1,
                            hspace=0.15)
    
    gs00 = gssp(1, 3,
                subplot_spec=gs0[0],
                wspace=0.15
                )
    
    gs01 = gssp(1, 3,
                subplot_spec=gs0[1],
                wspace=0.15
                )

    ax0 = plt.subplot(gs00[0],gid='(a)')
    ax1 = plt.subplot(gs00[1],gid='(b)')
    ax2 = plt.subplot(gs00[2],gid='(c)')
    ax3 = plt.subplot(gs01[0],gid='(d)')
    ax4 = plt.subplot(gs01[1],gid='(e)')
    ax5 = plt.subplot(gs01[2],gid='(f)')

axes = [ax0,ax1,ax2,ax3,ax4,ax5]


''' adjsut axis with colorbar '''
w = 0.25
h = 0.360465116279
pos=[0.665151515152, 0.539534883721, w, h]
ax2.set_position(pos, which=u'both')

out.plot('u',axes=[ax0,ax1,ax2],add_median=True,add_title=False,
         cbar_label='[frequency]',show=False)

out.plot('v',axes=[ax3,ax4,ax5],add_median=True,add_title=False,
         add_cbar=False,show=False,subax_label=False)

ax1.set_yticklabels('')
ax2.set_yticklabels('')
ax4.set_yticklabels('')
ax5.set_yticklabels('')

for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=18,weight='bold',
            transform=ax.transAxes)

#plt.show()

fname='/home/raul/Desktop/cfad_windprof.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')

