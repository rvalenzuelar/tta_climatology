'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

    Example:

    import plot_wprof_histogram as pwh

    pwh.plot(year=[2012],target='wdir')

'''

import parse_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rv_utilities import pandas2stack, add_colorbar
from tta_analysis import tta_analysis

def plot(year=[],target=None,pngsuffix=False, normalized=True,
        contourf=True, pdfsuffix=False, wdsurf=None, wdwpro=None,
        rainbb=None, raincz=None, nhours=None):
    
    name={'wdir':'Wind Direction',
          'wspd':'Wind Speed'}

    if target == 'wdir':
        bins = np.arange(0,370,10)
        hist_xticks = np.arange(0,420,60)
        hist_xlim = [0,360]
    elif target == 'wspd':
        bins = np.arange(0,36,1)
        hist_xticks = np.arange(0,40,5)
        hist_xlim = [0,35]

    first = True        
    for y in year:
        print('Processing year {}'.format(y))

        ' tta analysis '
        tta = tta_analysis(y)
        tta.start_df(wdir_surf=wdsurf,
                       wdir_wprof=wdwpro,
                       rain_bby=rainbb,
                       rain_czd=raincz,
                       nhours=nhours)


        ' retrieve dates '
        include_dates = tta.include_dates
        tta_dates = tta.tta_dates
        notta_dates = tta.notta_dates

        ' read wprof '
        wprof_df = parse_data.windprof(y)
        wprof = wprof_df.dframe[target]        

        ' wprof partition '
        wprof = wprof.loc[include_dates]    # all included
        wprof_tta = wprof.loc[tta_dates]    # only tta
        wprof_notta = wprof.loc[notta_dates]# only notta
        
        s1 = np.squeeze(pandas2stack(wprof))
        s2 = np.squeeze(pandas2stack(wprof_tta))
        s3 = np.squeeze(pandas2stack(wprof_notta))

        if first:
            wp = s1
            wp_tta = s2
            wp_notta = s3
            first = False
        else:
            wp = np.hstack((wp,s1))
            wp_tta = np.hstack((wp_tta,s2))
            wp_notta = np.hstack((wp_notta, s3))

    _,wp_hours = wp.shape
    _,tta_hours = wp_tta.shape
    _,notta_hours = wp_notta.shape

    ' makes CFAD '
    hist_array = np.empty((40,len(bins)-1,3))
    for hgt in range(wp.shape[0]):
        
        row1 = wp[hgt,:]
        row2 = wp_tta[hgt,:]
        row3 = wp_notta[hgt,:]

        for n,r in enumerate([row1,row2,row3]):

            ' following CFAD Yuter et al (1995) '
            freq,bins=np.histogram(r[~np.isnan(r)],
                                    bins=bins)
            if normalized:
                hist_array[hgt,:,n] = 100.*(freq/float(freq.sum()))
            else:
                hist_array[hgt,:,n] = freq


    fig,axs = plt.subplots(1,3,sharey=True,figsize=(10,8))

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    hist_wp = np.squeeze(hist_array[:,:,0])
    hist_wptta = np.squeeze(hist_array[:,:,1])
    hist_wpnotta = np.squeeze(hist_array[:,:,2])

    x = bins
    y = wprof_df.hgt

    if contourf:
        X,Y = np.meshgrid(x,y)
        nancol = np.zeros((40,1))+np.nan
        hist_wp = np.hstack((hist_wp,nancol))
        hist_wptta = np.hstack((hist_wptta,nancol))
        hist_wpnotta = np.hstack((hist_wpnotta,nancol))

        vmax=20
        nlevels = 10
        delta = int(vmax/nlevels)
        v = np.arange(2,vmax+delta,delta)

        cmap = cm.get_cmap('plasma')

        ax1.contourf(X,Y,hist_wp,v,cmap=cmap)
        p = ax2.contourf(X,Y,hist_wptta,v,cmap=cmap,extend='max')
        p.cmap.set_over(cmap(1.0))
        ax3.contourf(X,Y,hist_wpnotta,v,cmap=cmap)
        cbar = add_colorbar(ax3,p,size='4%')
    else:
        p = ax1.pcolormesh(x,y,hist_wp,cmap='viridis')
        ax2.pcolormesh(x,y,hist_wptta,cmap='viridis')
        ax3.pcolormesh(x,y,hist_wpnotta,cmap='viridis')
        amin = np.amin(hist_wpnotta)
        amax = np.amax(hist_wpnotta)
        cbar = add_colorbar(ax3,p,size='4%',ticks=[amin,amax])
        cbar.ax.set_yticklabels(['low','high'])


    ' --- setup ax1 --- '
    amin = np.amin(hist_wp)
    amax = np.amax(hist_wp)
    ax1.set_xticks(hist_xticks)
    ax1.set_xlim(hist_xlim)
    ax1.set_ylim([0,4000])
    txt = 'All profiles (n={})'.format(wp_hours)
    ax1.text(0.5,0.95,txt,fontsize=15,
            transform=ax1.transAxes,va='bottom',ha='center')
    ax1.set_ylabel('Altitude [m] MSL')

    ' --- setup ax2 --- '
    amin = np.amin(hist_wptta)
    amax = np.amax(hist_wptta)
    ax2.set_xticks(hist_xticks)
    ax2.set_xlim(hist_xlim)
    ax2.set_ylim([0,4000])
    ax2.set_xlabel(name[target])
    txt = 'TTA (n={})'.format(tta_hours)
    ax2.text(0.5,0.95,txt,fontsize=15,
            transform=ax2.transAxes,va='bottom',ha='center')

    ' --- setup ax3 --- '
    ax3.set_xticks(hist_xticks)
    ax3.set_xlim(hist_xlim)
    ax3.set_ylim([0,4000])
    txt = 'NO-TTA (n={})'.format(notta_hours)
    ax3.text(0.5,0.95,txt,fontsize=15,
            transform=ax3.transAxes,va='bottom',ha='center')


    title = 'Normalized frequencies of BBY wind profiles {} \n'
    title += 'TTA wdir_surf:{}, wdir_wp:{}, '
    title += 'rain_bby:{}, rain_czd:{}, nhours:{}'
    
    if len(year) == 1:
        yy = 'year {}'.format(year[0])
    else:
        yy = 'year {} to {}'.format(year[0],year[-1])
    plt.suptitle(title.format(yy, wdsurf, wdwpro, rainbb, raincz, nhours),
                fontsize=15)

    plt.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.1, wspace=0.1)
     
    if pngsuffix:
        out_name = 'wprof_{}_cfad{}.png'
        plt.savefig(out_name.format(target,pngsuffix))
        plt.close()
    elif pdfsuffix:
        out_name = 'wprof_{}_cfad{}.pdf'
        plt.savefig(out_name.format(target,pdfsuffix))
        plt.close()        
    else:
        plt.show()
