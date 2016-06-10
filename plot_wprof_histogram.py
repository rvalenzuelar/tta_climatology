'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''

import parse_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rv_utilities import pandas2stack, add_colorbar
from tta_analysis import tta_analysis

def plot(year=None,target=None,pngsuffix=False,
		 normalized=True):

	name={'wdir':'Wind Direction',
	      'wspd':'Wind Speed'}

	if target == 'wdir':
		# vmin,vmax = [0,360]
		bins = np.arange(0,370,10)
		hist_xticks = np.arange(0,420,60)
		hist_xlim = [0,360]
	elif target == 'wspd':
		# vmin,vmax = [0,30]
		bins = np.arange(0,36,1)
		hist_xticks = np.arange(0,40,5)
		hist_xlim = [0,35]


	wdsurf=125
	wdwpro=170
	rainbb=0.25
	nhours=5
	ttastats = tta_analysis(year)
	ttastats.start(wdir_surf=wdsurf,
				   wdir_wprof=wdwpro,
				   rain_bby=rainbb,
				   nhours=nhours)
	stats_beg=ttastats.time_beg
	stats_end=ttastats.time_end
	stats_dates = pd.date_range(start=stats_beg,
								end=stats_end,
								freq='1H')
	tta_dates = stats_dates[ttastats.bool]
	notta_dates = stats_dates[~ttastats.bool]


	wprof_df = parse_data.windprof(year)

	wprof = wprof_df.dframe[target]
	wprof_tta = wprof.loc[tta_dates]
	wprof_notta = wprof.loc[notta_dates]

	wp = np.squeeze(pandas2stack(wprof))
	wp_tta = np.squeeze(pandas2stack(wprof_tta))
	wp_notta = np.squeeze(pandas2stack(wprof_notta))

	hist_array = np.empty((40,len(bins)-1,3))

	for hgt in range(wp.shape[0]):
	    
	    row1 = wp[hgt,:]
	    row2 = wp_tta[hgt,:]
	    row3 = wp_notta[hgt,:]

	    for n,r in enumerate([row1,row2,row3]):

		    freq,bins=np.histogram(r[~np.isnan(r)],
		                            bins=bins,
		                            density=normalized)
		    hist_array[hgt,:,n]=freq



	fig,axs = plt.subplots(1,3,sharey=True,figsize=(10,8))

	ax1=axs[0]
	ax2=axs[1]
	ax3=axs[2]

	hist_wp = np.squeeze(hist_array[:,:,0])
	hist_wptta = np.squeeze(hist_array[:,:,1])
	hist_wpnotta = np.squeeze(hist_array[:,:,2])

	x = bins
	y = wprof_df.hgt

	p = ax1.pcolormesh(x,y,hist_wp,cmap='viridis')
	amin = np.amin(hist_wp)
	amax = np.amax(hist_wp)
	ax1.set_xticks(hist_xticks)
	ax1.set_xlim(hist_xlim)
	ax1.text(0.5,0.95,'All profiles',fontsize=15,
			transform=ax1.transAxes,va='bottom',ha='center')
	ax1.set_ylabel('Altitude [m] MSL')

	p = ax2.pcolormesh(x,y,hist_wptta,cmap='viridis')
	amin = np.amin(hist_wptta)
	amax = np.amax(hist_wptta)
	ax2.set_xticks(hist_xticks)
	ax2.set_xlim(hist_xlim)
	ax2.set_xlabel(name[target])
	ax2.text(0.5,0.95,'TTA',fontsize=15,
			transform=ax2.transAxes,va='bottom',ha='center')

	p = ax3.pcolormesh(x,y,hist_wpnotta,cmap='viridis')
	amin = np.amin(hist_wpnotta)
	amax = np.amax(hist_wpnotta)
	cbar = add_colorbar(ax3,p,size='4%',ticks=[amin,amax])
	cbar.ax.set_yticklabels(['low','high'])
	ax3.set_xticks(hist_xticks)
	ax3.set_xlim(hist_xlim)
	ax3.set_xlim(hist_xlim)
	ax3.set_xlim(hist_xlim)
	ax3.text(0.5,0.95,'NO-TTA',fontsize=15,
			transform=ax3.transAxes,va='bottom',ha='center')

	wdsurf=125
	wdwpro=170
	rainbb=0.25
	nhours=5

	title1 = 'Normalized frequencies of BBY wind profiles year {} \n'
	title2 = '(TTA wdir_surf:{}, wdir_wp:{}, rain_bby:{}, nhours:{}'
	title = title1+title2
	plt.suptitle(title.format(year, wdsurf, wdwpro, rainbb, nhours),
				fontsize=15)
	plt.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.1,
						wspace=0.1)

	if pngsuffix:
	    out_name = 'wprof_{}_histograms_{}.png'
	    plt.savefig(out_name.format(target,pngsuffix))
	    plt.close()
	else:
	    plt.show(block=False)


