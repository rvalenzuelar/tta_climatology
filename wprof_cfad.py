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
from tta_analysis2 import tta_analysis

class cfad:

    def __init__(self,year=[],wdsurf=None, wdwpro=None,
                     rainbb=None, raincz=None, nhours=None):
              

        out = processv2(year=year,wdsurf=wdsurf,
                        wdwpro=wdwpro,rainbb=rainbb,
                        raincz=raincz, nhours=nhours)
      
        self.spd_hist = out[0]
        self.dir_hist = out[1]
        self.u_hist   = out[2]
        self.v_hist   = out[3]
        self.spd_cfad = out[4]
        self.dir_cfad = out[5]
        self.u_cfad   = out[6]
        self.v_cfad   = out[7]
        self.bins_spd = out[8]
        self.bins_dir = out[9]
        self.bins_u   = out[10]
        self.bins_v   = out[11]
        self.hgts      = out[12]
        self.wp_hours  = out[13]
        self.tta_hours = out[14]
        self.notta_hours  = out[15]
        self.spd_average  = out[16]
        self.dir_average  = out[17]
        self.u_average  = out[18]
        self.v_average  = out[19]
        self.spd_median  = out[20]
        self.dir_median  = out[21]
        self.u_median  = out[22]
        self.v_median  = out[23]
        self.wdsurf = wdsurf
        self.wdwpro = wdwpro
        self.rainbb = rainbb
        self.raincz = raincz
        self.nhours = nhours
        self.year = year
    
    
    
    def plot(self,target,axes=None,pngsuffix=False, pdfsuffix=False,
             contourf=True, add_median=False,add_average=False,
             add_title=True, add_cbar=True,cbar_label=None,show=True,
             subax_label=True,top_altitude=4000):
        
        name={'wdir':'Wind Direction',
              'wspd':'Wind Speed',
              'u':'u-wind',
              'v':'v-wind'}
    
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
        elif target == 'u':
            cfad = self.u_cfad
            median = self.u_median
            average = self.u_average
            bins = self.bins_u
            hist_xticks = np.arange(-14,22,4)
            hist_xlim = [-14,20] 
        elif target == 'v':
            cfad = self.v_cfad
            median = self.v_median
            average = self.v_average
            bins = self.bins_v
            hist_xticks = np.arange(-14,24,4)
            hist_xlim = [-14,20] 
    
        if axes is None:
            fig,axs = plt.subplots(1,3,sharey=True,figsize=(10,8))
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]
    
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
    
            vmax=15
            nlevels = 6
            delta = int(vmax/nlevels)
            v = np.arange(2,vmax+delta,delta)
    
            cmap = cm.get_cmap('plasma')
    
            ax1.contourf(X,Y,hist_wp,v,cmap=cmap)
            p = ax2.contourf(X,Y,hist_wptta,v,cmap=cmap)
            ax3.contourf(X,Y,hist_wpnotta,v,cmap=cmap)
            
            lcolor = (0.6,0.6,0.6)
            lw = 3
            ax1.vlines(0,0,4000,linestyle='--',color=lcolor,lw=lw)
            ax2.vlines(0,0,4000,linestyle='--',color=lcolor,lw=lw)
            ax3.vlines(0,0,4000,linestyle='--',color=lcolor,lw=lw)
                        
            
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

        ''' add color bar '''            
        if add_cbar is True:   
            add_colorbar(ax3,p,size='4%',loc='right',
                         label=cbar_label)
    
    
        ' --- setup ax1 --- '
        ax1.set_xticks(hist_xticks)
        ax1.set_xlim(hist_xlim)
        ax1.set_ylim([0,top_altitude])
        ax1.set_ylabel('Altitude [m] MSL')
    
        ' --- setup ax2 --- '
        ax2.set_xticks(hist_xticks)
        ax2.set_xlim(hist_xlim)
        ax2.set_ylim([0,top_altitude])
        ax2.set_xlabel(name[target])

    
        ' --- setup ax3 --- '
        ax3.set_xticks(hist_xticks)
        ax3.set_xlim(hist_xlim)
        ax3.set_ylim([0,top_altitude])


        ''' add subaxis label '''
        vpos=1.05
        if subax_label is True:
            txt = 'All profiles (n={})'.format(self.wp_hours)
            ax1.text(0.5,vpos,txt,fontsize=15,weight='bold',
                    transform=ax1.transAxes,va='center',ha='center')            
            txt = 'TTA (n={})'.format(self.tta_hours)
            ax2.text(0.5,vpos,txt,fontsize=15,weight='bold',
                    transform=ax2.transAxes,va='center',ha='center')
            txt = 'NO-TTA (n={})'.format(self.notta_hours)
            ax3.text(0.5,vpos,txt,fontsize=15, weight='bold',
                    transform=ax3.transAxes,va='center',ha='center')
        
        ''' add title '''
        if add_title is True:
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
         
        if pngsuffix:
            out_name = 'wprof_{}_cfad{}.png'
            plt.savefig(out_name.format(target,pngsuffix))
            plt.close()
        elif pdfsuffix:
            out_name = 'wprof_{}_cfad{}.pdf'
            plt.savefig(out_name.format(target,pdfsuffix))
            plt.close()        
        
        if show is True:
            plt.show()

        return {'axes':[ax1,ax2,ax3],'im':p}

def processv2(year=[],wdsurf=None,
               wdwpro=None,rainbb=None,
               raincz=None, nhours=None):
        
        ''' v2: target loop moved into year loop '''
        
        
        binss={'wdir': np.arange(0,370,10),
               'wspd': np.arange(0,36,1),
               'u': np.arange(-15,21,1),
               'v': np.arange(-14,21,1),
               }
               
        target = ['wdir','wspd']
        arrays = {}
        wsp = np.empty((40,1))
        wsp_tta = np.empty((40,1))
        wsp_notta = np.empty((40,1))
        wdr = np.empty((40,1))
        wdr_tta = np.empty((40,1))
        wdr_notta = np.empty((40,1))
        
        for y in year:
            print('Processing year {}'.format(y))
            
            ' tta analysis '
            tta = tta_analysis(y)
            tta.start_df(wdir_surf  = wdsurf,
                         wdir_wprof = wdwpro,
                         rain_bby   = rainbb,
                         rain_czd   = raincz,
                         nhours     = nhours)
    
            ' retrieve dates '
            include_dates = tta.include_dates
            tta_dates     = tta.tta_dates
            notta_dates   = tta.notta_dates
    
            ' read wprof '
            wprof_df = parse_data.windprof(y)
            
            for n,t in enumerate(target):
                
                wprof = wprof_df.dframe[t]        
        
                ' wprof partition '
                wprof = wprof.loc[include_dates]    # all included
                wprof_tta = wprof.loc[tta_dates]    # only tta
                wprof_notta = wprof.loc[notta_dates]# only notta
                
                s1 = np.squeeze(pandas2stack(wprof))
                if wprof_tta.size > 0:
                    s2 = np.squeeze(pandas2stack(wprof_tta))
                    ttaok = True
                else:
                    ttaok =False
                s3 = np.squeeze(pandas2stack(wprof_notta))
        
                if t == 'wdir':
                    wdr = np.hstack((wdr,s1))
                    if ttaok is True:
                        wdr_tta = np.hstack((wdr_tta,s2))
                    wdr_notta = np.hstack((wdr_notta, s3))                    
                else:
                    wsp = np.hstack((wsp,s1))
                    if ttaok is True:
                        wsp_tta = np.hstack((wsp_tta,s2))
                    wsp_notta = np.hstack((wsp_notta, s3))

        arrays['wdir']=[wdr,wdr_tta,wdr_notta]
        arrays['wspd']=[wsp,wsp_tta,wsp_notta]
                
        uw = -wsp*np.sin(np.radians(wdr))
        uw_tta = -wsp_tta*np.sin(np.radians(wdr_tta))
        uw_notta = -wsp_notta*np.sin(np.radians(wdr_notta))

        vw = -wsp*np.cos(np.radians(wdr))
        vw_tta = -wsp_tta*np.cos(np.radians(wdr_tta))
        vw_notta = -wsp_notta*np.cos(np.radians(wdr_notta))        

        arrays['u']=[uw,uw_tta,uw_notta]
        arrays['v']=[vw,vw_tta,vw_notta]

        ''' total hours, first rows are empty '''                
        _,wp_hours = wsp.shape
        _,tta_hours = wsp_tta.shape
        _,notta_hours = wsp_notta.shape    
        wp_hours -= 1
        tta_hours-= 1
        notta_hours -= 1
        
        ' initialize arrays '
        hist_array_spd = np.empty((40,len(binss['wspd'])-1,3))
        hist_array_dir = np.empty((40,len(binss['wdir'])-1,3))
        cfad_array_spd = np.empty((40,len(binss['wspd'])-1,3))
        cfad_array_dir = np.empty((40,len(binss['wdir'])-1,3))        
        average_spd = np.empty((40,3))
        average_dir = np.empty((40,3))
        median_spd = np.empty((40,3))
        median_dir = np.empty((40,3))
        
        ' loop for variable (wdir,wspd) '
        for k,v in arrays.iteritems():        
        
            hist_array = np.empty((40,len(binss[k])-1,3))
            cfad_array = np.empty((40,len(binss[k])-1,3))
            average = np.empty((40,3))
            median = np.empty((40,3))
            
            ' extract value'
            wp = v[0]
            wp_tta = v[1]
            wp_notta = v[2]
        
            ' makes CFAD '
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
            elif k == 'wdir':                
                hist_array_dir = hist_array
                cfad_array_dir = cfad_array
                average_dir = average
                median_dir = median
            elif k == 'u':
                hist_array_u = hist_array
                cfad_array_u = cfad_array
                average_u = average
                median_u = median                
            elif k == 'v':
                hist_array_v = hist_array
                cfad_array_v = cfad_array
                average_v = average
                median_v = median
    
        return [hist_array_spd,
                hist_array_dir,
                hist_array_u,
                hist_array_v,
                cfad_array_spd,
                cfad_array_dir,
                cfad_array_u,
                cfad_array_v,
                binss['wspd'],
                binss['wdir'],
                binss['u'],
                binss['v'],
                wprof_df.hgt,
                wp_hours,
                tta_hours,
                notta_hours,
                average_spd,
                average_dir,
                average_u,
                average_v,
                median_spd,
                median_dir,
                median_u,
                median_v,
                ]



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
        