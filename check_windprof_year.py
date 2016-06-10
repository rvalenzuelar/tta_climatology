'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''

import parse_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import matplotlib.gridspec as gsp
import matplotlib.dates as mdates
from rv_utilities import pandas2stack, add_colorbar

def plot_with_lines(year=None,target=None):

    fig = plt.figure(figsize=(10,5))

    gs = gsp.GridSpec(1, 2,
                      width_ratios=[2,1]
                      )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    wprof = parse_data.windprof(year)
    wp = np.squeeze(pandas2stack(wprof.dframe[target]))
    wp_ma = ma.masked_where(np.isnan(wp),wp)
    X,Y=wprof.time,wprof.hgt
    ax1.pcolormesh(X,Y,wp_ma,vmin=0,vmax=360)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.set_xlabel(r'$ Time \rightarrow$')
    ax1.set_ylabel('height gate')

    for prof in range(wp.shape[1]):
        x = wp[:,prof]
        y = range(wp.shape[0])
        ax2.plot(x,y,color='r',alpha=0.05)
        # ax2.scatter(x,y,color='r',alpha=0.05)
    ax2.set_yticklabels('')
    ax2.set_xlabel(target)

    ax1.set_title('BBY Windprof wdir')
    plt.tight_layout()
    plt.show(block=False)

def plot_with_hist(year=None,target=None,normalized=True,
                    pngsuffix=None):

    name={'wdir':'Wind Direction',
          'wspd':'Wind Speed'}

    if target == 'wdir':
        vmin,vmax = [0,360]
        bins = np.arange(0,370,10)
        hist_xticks = np.arange(0,400,40)
        hist_xlim = [0,360]
    elif target == 'wspd':
        vmin,vmax = [0,30]
        bins = np.arange(0,36,1)
        hist_xticks = np.arange(0,40,5)
        hist_xlim = [0,35]

    fig = plt.figure(figsize=(20,5))

    gs = gsp.GridSpec(1, 2,
                      width_ratios=[3,1]
                      )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    wprof = parse_data.windprof(year)
    wp = np.squeeze(pandas2stack(wprof.dframe[target]))
    wp_ma = ma.masked_where(np.isnan(wp),wp)
    X,Y = wprof.time,wprof.hgt
    p = ax1.pcolormesh(X,Y,wp_ma,vmin=vmin,vmax=vmax)
    add_colorbar(ax1,p)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.set_xlabel(r'$ Time \rightarrow$')
    ax1.set_ylabel('Altitude [m] MSL')
    ax1.set_title('BBY Windprof '+name[target])

    array = np.empty((40,len(bins)-1))
    for hgt in range(wp.shape[0]):
        row = wp[hgt,:]
        freq,bins=np.histogram(row[~np.isnan(row)],
                                bins=bins,
                                density=normalized)
        array[hgt,:]=freq

    x = bins
    y = wprof.hgt
    p = ax2.pcolormesh(x,y,array,cmap='viridis')
    amin = np.amin(array)
    amax = np.amax(array)
    cbar = add_colorbar(ax2,p,size='4%',ticks=[amin,amax])
    cbar.ax.set_yticklabels(['low','high'])
    ax2.set_xticks(hist_xticks)
    ax2.set_yticklabels('')
    ax2.set_xlabel(name[target])
    ax2.set_xlim(hist_xlim)
    ax2.set_title('Normalized frequency')

    plt.tight_layout()
    if pngsuffix:
        out_name = 'wprof_{}_{}.png'
        plt.savefig(out_name.format(target,pngsuffix))
    else:
        plt.show(block=False)
