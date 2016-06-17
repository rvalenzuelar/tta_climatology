# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:23:07 2016

@author: raul
"""

'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

    Example:

    from wprof_cfad import cfad

    out=cfad(year=[1998],wdsurf=125,wdwpro=170,
             rainbb=0.25,raincz=0.25,nhours=2)
             
    out.plot('wspd',add_average=True,add_median=True)


'''

import parse_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rv_utilities import pandas2stack, add_colorbar
from tta_analysis import tta_analysis

class cfad:

    def __init__(self,year=[],wdsurf=None, wdwpro=None,
                     rainbb=None, raincz=None, nhours=None):
              

        out = process(year=year,wdsurf=wdsurf,
                       wdwpro=wdwpro,rainbb=rainbb,
                       raincz=raincz, nhours=nhours)
      
        self.spd_hist = out[0]
        self.dir_hist = out[1]
        self.spd_cfad = out[2]
        self.dir_cfad = out[3]
        self.bins_spd = out[4]
        self.bins_dir = out[5]
        self.hgts = out[6]
        self.wp_hours  = out[7]
        self.tta_hours = out[8]
        self.notta_hours  = out[9]
        self.spd_average  = out[10]
        self.dir_average  = out[11]
        self.spd_median  = out[12]
        self.dir_median  = out[13]
        self.wdsurf = wdsurf
        self.wdwpro = wdwpro
        self.rainbb = rainbb
        self.raincz = raincz
        self.nhours = nhours
        self.year = year
    
    
    
    def plot(self,target,pngsuffix=False, pdfsuffix=False,contourf=True,
             add_median=False,add_average=False):
        
        name={'wdir':'Wind Direction',
              'wspd':'Wind Speed'}
    
        if target == 'wdir':
            cfad = self.dir_cfad
            median = self.dir_median
            average = self.dir_average
            bins = self.bins_dir
            hist_xticks = np.arange(0,420,60)
            hist_xlim = [0,360]
        elif target == 'wspd':
            cfad = self.spd_cfad
            median = self.spd_median
            average = self.spd_average
            bins = self.bins_spd
            hist_xticks = np.arange(0,40,5)
            hist_xlim = [0,35]    
    
        fig,axs = plt.subplots(1,3,sharey=True,figsize=(10,8))
    
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
    
        hist_wp = np.squeeze(cfad[:,:,0])
        hist_wptta = np.squeeze(cfad[:,:,1])
        hist_wpnotta = np.squeeze(cfad[:,:,2])
    
        x = bins
        y = self.hgts
    
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

            lw=3
            if add_median:
                ax1.plot(median[:,0],self.hgts,color='w',lw=lw)
                ax2.plot(median[:,1],self.hgts,color='w',lw=lw)
                ax3.plot(median[:,2],self.hgts,color='w',lw=lw)
            if add_average:
                ax1.plot(average[:,0],self.hgts,color='w',lw=lw)
                ax2.plot(average[:,1],self.hgts,color='w',lw=lw)
                ax3.plot(average[:,2],self.hgts,color='w',lw=lw)            
            
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
        txt = 'All profiles (n={})'.format(self.wp_hours)
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
        txt = 'TTA (n={})'.format(self.tta_hours)
        ax2.text(0.5,0.95,txt,fontsize=15,
                transform=ax2.transAxes,va='bottom',ha='center')
    
        ' --- setup ax3 --- '
        ax3.set_xticks(hist_xticks)
        ax3.set_xlim(hist_xlim)
        ax3.set_ylim([0,4000])
        txt = 'NO-TTA (n={})'.format(self.notta_hours)
        ax3.text(0.5,0.95,txt,fontsize=15,
                transform=ax3.transAxes,va='bottom',ha='center')
    
    
        title = 'Normalized frequencies of BBY wind profiles {} \n'
        title += 'TTA wdir_surf:{}, wdir_wp:{}, '
        title += 'rain_bby:{}, rain_czd:{}, nhours:{}'
        
        if len(self.year) == 1:
            yy = 'year {}'.format(self.year[0])
        else:
            yy = 'year {} to {}'.format(self.year[0],self.year[-1])
        plt.suptitle(title.format(yy, self.wdsurf, 
                    self.wdwpro, self.rainbb, self.raincz, self.nhours),
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


def process(year=[],wdsurf=None,
               wdwpro=None,rainbb=None,
               raincz=None, nhours=None):
        
        
        binss={'wdir':np.arange(0,370,10),
               'wspd':np.arange(0,36,1)}
        target = ['wdir','wspd']
        arrays = {}
        for t in target:
        
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
            
                wprof = wprof_df.dframe[t]        
        
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
            
            arrays[t]=[wp,wp_tta,wp_notta]

    
        ' makes CFAD '
        hist_array_spd = np.empty((40,len(binss['wspd'])-1,3))
        hist_array_dir = np.empty((40,len(binss['wdir'])-1,3))
        cfad_array_spd = np.empty((40,len(binss['wspd'])-1,3))
        cfad_array_dir = np.empty((40,len(binss['wdir'])-1,3))
        
        average_spd = np.empty((40,3))
        average_dir = np.empty((40,3))
        median_spd = np.empty((40,3))
        median_dir = np.empty((40,3))
        
        for k,v in arrays.iteritems():        
        
            hist_array = np.empty((40,len(binss[k])-1,3))
            cfad_array = np.empty((40,len(binss[k])-1,3))
            average = np.empty((40,3))
            median = np.empty((40,3))
            wp = v[0]
            wp_tta = v[1]
            wp_notta = v[2]
        
            for hgt in range(wp.shape[0]):
                
                row1 = wp[hgt,:]
                row2 = wp_tta[hgt,:]
                row3 = wp_notta[hgt,:]
        
                for n,r in enumerate([row1,row2,row3]):
        
                    ' following CFAD Yuter et al (1995) '
                    freq,bins=np.histogram(r[~np.isnan(r)],
                                            bins=binss[k])
                    hist_array[hgt,:,n] = freq
                    cfad_array[hgt,:,n] = 100.*(freq/float(freq.sum()))
        
                    bin_middle = (bins[1:]+bins[:-1])/2.
                    average[hgt,n] = np.sum(freq*bin_middle)/freq.sum()
                    median[hgt,n] = np.percentile(r[~np.isnan(r)],50)
            
            if k == 'wspd':
                hist_array_spd = hist_array
                cfad_array_spd = cfad_array
                average_spd = average
                median_spd = median
            else:                
                hist_array_dir = hist_array
                cfad_array_dir = cfad_array
                average_dir = average
                median_dir = median
    
        return [hist_array_spd,
                hist_array_dir,
                cfad_array_spd,
                cfad_array_dir,
                binss['wspd'],
                binss['wdir'],
                wprof_df.hgt,
                wp_hours,
                tta_hours,
                notta_hours,
                average_spd,
                average_dir,
                median_spd,
                median_dir]
        