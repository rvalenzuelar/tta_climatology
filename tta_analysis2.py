'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu

    Example:

    from tta_analysis import tta_analysis

    tta = tta_analysis(1998)

    tta.start(wdir_surf=125,wdir_wprof=170,
                    rain_bby=0.25,nhours=5)
    tta.print_stats()

    --- or ---

    # creates pandas dataframe

    tta.start_df(wdir_surf=125,wdir_wprof=170,
                    rain_bby=0.25,nhours=5)

    df = tta.df

'''

import numpy as np
import parse_data
# import ctext
from datetime import timedelta

class tta_analysis:

    def __init__(self, year=None):
        # type: (object) -> object
        """

        :rtype: object
        """
        self.year = year

    def start(self, wdir_surf=None, wdir_wprof=None, 
              rain_bby=None,rain_czd=None,nhours=None):

        ''' this is an old verion
            prefer start_df that uses pandas dataframe
            for analysis
        '''

        bby = parse_data.surface('bby', self.year)
        czd = parse_data.surface('czd', self.year)
        wprof = parse_data.windprof(self.year)

        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()
        beg_wpr, end_wpr = wprof.check_beg_end()

        ''' the latest of the beg '''
        time_beg = max(beg_bby, beg_czd, beg_wpr)

        ''' the earliest of the end '''
        time_end = min(end_bby, end_czd, end_wpr)

        ''' rainfall before all obs start '''
        rbby_before = np.nansum(bby.dframe.loc[:time_beg].precip)
        rczd_before = np.nansum(czd.dframe.loc[:time_beg].precip)

        ''' rainfall after all obs end '''
        rbby_after = np.nansum(bby.dframe.loc[time_end:].precip)
        rczd_after = np.nansum(czd.dframe.loc[time_end:].precip)

        ''' number of windprofiles before (after)
            all obs start (end) '''
        nwprof_before = len(wprof.dframe.loc[:time_beg].wdir)
        nwprof_after = len(wprof.dframe.loc[time_end:].wdir)

        onehr = timedelta(hours=1)
        time = time_beg
        bool_buffer = np.array([False] * nhours)
        tta_bool = np.array([])
        rainfall_czd = np.array([])
        rainfall_bby = np.array([])
#        wpr_wd_inc = []
#        wpr_ws_inc = []
        count = 0
        count_while = 0
        count_exclude = 0

        while (time <= time_end):
                
            surf_wd = bby.dframe.loc[time].wdir
            wpr_wd0 = wprof.dframe.loc[time].wdir[0]  # first gate
            pbby = bby.dframe.loc[time].precip
            pczd = czd.dframe.loc[time].precip

            ''' exclude data when there is nan in 
                surf obs or windprof first gate '''
            if surf_wd is None or np.isnan(surf_wd) or np.isnan(wpr_wd0):
                # tta_bool = np.append(tta_bool, [False])
                count_exclude += 1
                time += onehr
                continue

            ''' these are obs included in the analysis, then we
                determine if they are tta or no-tta '''
            rainfall_bby=np.append(rainfall_bby,pbby)
            rainfall_czd=np.append(rainfall_czd,pczd)


            ''' check conditions '''
            cond1 = (surf_wd <= wdir_surf)
            cond2 = (wpr_wd0 <= wdir_wprof)
            if rain_bby and rain_czd:
                cond3 = (pbby >= rain_bby)
                cond4 = (pczd >= rain_czd)
                tta_condition = cond1 and cond2 and \
                                cond3 and cond4
            elif rain_czd:
                cond3 = (pczd >= rain_czd)
                tta_condition = cond1 and cond2 and cond3
            elif rain_bby:
                cond3 = (pbby >= rain_bby)
                tta_condition = cond1 and cond2 and cond3
            else:
                tta_condition = cond1 and cond2

            ''' construct boolean array indicating
                hourly TTA conditions with minumm
                of nhours '''
            if tta_condition and bool_buffer.all():
                tta_bool = np.append(tta_bool, [True])
            elif tta_condition:
                bool_buffer[count] = True
                count += 1
                if bool_buffer.all():
                    tta_bool = np.append(tta_bool, bool_buffer)
            else:
                bufsum = bool_buffer.sum()
                if bufsum == 0 or bufsum == nhours:
                    tta_bool = np.append(tta_bool, [False])
                else:
                    tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
                # reset buffer
                bool_buffer = np.array([False] * nhours)
                count = 0

            count_while += 1
            time += onehr

        tta_bool = np.array(tta_bool).astype(bool)
        tta_hours = tta_bool.sum()
        notta_hours = count_while-tta_hours
        self.tta_hours = tta_hours
        self.notta_hours = notta_hours
        self.time_beg = time_beg
        self.time_end = time_end
        self.count_while = count_while
        self.count_exclude = count_exclude
        self.total_rainfall_bby = np.nansum(rainfall_bby)
        self.total_rainfall_czd = np.nansum(rainfall_czd)
        self.bool = tta_bool
        self.tta_rainfall_czd = np.nansum(rainfall_czd[tta_bool])
        self.tta_rainfall_bby = np.nansum(rainfall_bby[tta_bool])
        self.notta_rainfall_czd = np.nansum(rainfall_czd[~tta_bool])
        self.notta_rainfall_bby = np.nansum(rainfall_bby[~tta_bool])
        self.rainfall_bby_before_analysis = rbby_before
        self.rainfall_bby_after_analysis = rbby_after
        self.rainfall_czd_before_analysis = rczd_before
        self.rainfall_czd_after_analysis = rczd_after
        self.nwprof_before = nwprof_before
        self.nwprof_after = nwprof_after
        self.wprof_hgt = wprof.hgt



        print('TTA analysis finished')


    def start_df(self, wdir_surf   = None,
                       wdir_wprof  = None,
                       wprof_gate  = 0,
                       rain_bby    = None,
                       rain_czd    = None,
                       nhours      = None):

        '''
            this version uses pandas dataframe, 
            it should be more accurate and simpler
            than start method
        '''

        import pandas as pd

        bby = parse_data.surface('bby', self.year)
        czd = parse_data.surface('czd', self.year)
        wprof = parse_data.windprof(self.year)

        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()
        beg_wpr, end_wpr = wprof.check_beg_end()

        ''' trim the head and tail of dataset depending
            on the latest time of the beginning and 
            earliest of the ending '''
        time_beg = max(beg_bby, beg_czd, beg_wpr)
        time_end = min(end_bby, end_czd, end_wpr)

        ''' initializations '''
        onehr = timedelta(hours=1)
        bool_buffer = np.array([False] * nhours)
        tta_bool = np.array([])
        count = 0
        rng = pd.date_range(start=time_beg,
                            end=time_end,
                            freq='1H')

        ''' columns included in the dataframe '''        
        cols = []        
        cols.append('wdsrf')
        wprofcol = 'wdwpr{}'.format(wprof_gate)
        cols.append(wprofcol)
        cols.append('rbby')
        cols.append('rczd')
        cols.append('tta')
        cols.append('consecutive')
        
        ''' create dataframe '''
        df = pd.DataFrame(index=rng,columns=cols)       
        
        
        ''' loop evaluates each time '''
        time = time_beg
        while (time <= time_end):

            surf_wd = bby.dframe.loc[time].wdir
            df.loc[time].wdsrf = surf_wd            
            
            wpr_wd0 = wprof.dframe.loc[time].wdir[wprof_gate] 
            df.loc[time][wprofcol] = wpr_wd0            
            
            pbby = bby.dframe.loc[time].precip
            df.loc[time].rbby = pbby            
            
            pczd = czd.dframe.loc[time].precip
            df.loc[time].rczd = pczd
                
#            if surf_wd is None:
#                surf_wd = np.nan
           
#            df.loc[time].wssrf = bby.dframe.loc[time].wspd
#            df.loc[time].wswpr = wprof.dframe.loc[time].wspd[0]

            ''' check conditions '''               
            if wdir_surf:
                if isinstance(wdir_surf,int):
                    cond1 = (surf_wd <= wdir_surf)
                elif isinstance(wdir_surf,str):
                    cond1 = parse_operator(surf_wd,wdir_surf)

            if wdir_wprof:
                if isinstance(wdir_wprof,int):
                    cond2 = (wpr_wd0 <= wdir_wprof) 
                elif isinstance(wdir_wprof,str):
                    cond2 = parse_operator(wpr_wd0,wdir_wprof)

            if rain_czd:
                cond3 = (pczd >= rain_czd)

            if rain_bby:            
                cond4 = (pbby >= rain_bby)
              
            ''' create joint condition '''
            if wdir_surf and wdir_wprof and rain_bby and rain_czd:
                tta_condition = cond1 and cond2 and cond3 and cond4
            elif wdir_surf and wdir_wprof and rain_czd:
                tta_condition = cond1 and cond2 and cond3
            elif wdir_surf and wdir_wprof and rain_bby:
                tta_condition = cond1 and cond2 and cond4
            elif wdir_surf and rain_czd:
                tta_condition = cond1 and cond3
            elif wdir_wprof and rain_czd:
                tta_condition = cond2 and cond3                
            elif wdir_surf and rain_bby:
                tta_condition = cond1 and cond4
            elif wdir_wprof and rain_bby:
                tta_condition = cond2 and cond4                
            elif wdir_surf and wdir_wprof:
                tta_condition = cond1 and cond2
            else:
                tta_condition = cond1 


            df.loc[time].tta = tta_condition

            ''' construct boolean array indicating
                hourly TTA conditions with minumm
                of nhours '''
            if tta_condition and bool_buffer.all():
                tta_bool = np.append(tta_bool, [True])
            elif tta_condition:
                bool_buffer[count] = True
                count += 1
                if bool_buffer.all():
                    tta_bool = np.append(tta_bool, bool_buffer)
            else:
                bufsum = bool_buffer.sum()
                if bufsum == 0 or bufsum == nhours:
                    tta_bool = np.append(tta_bool, [False])
                else:
                    tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
                ' reset buffer '
                bool_buffer = np.array([False] * nhours)
                count = 0

            time += onehr

        df.consecutive = tta_bool.astype(bool)

        ar_wdsrf = df.wdsrf.values.astype(float)
        ar_wdwpr = df[wprofcol].values.astype(float)
        ar_rbby = df.rbby.values.astype(float)
        ar_rczd = df.rczd.values.astype(float)
        
        wdsrfIsNan = np.isnan(ar_wdsrf)
        wdwprIsNan = np.isnan(ar_wdwpr)
        rbbyIsNan = np.isnan(ar_rbby)
        rczdIsNan = np.isnan(ar_rczd)
        
        
        if rain_czd is None:
            exclude = wdsrfIsNan | wdwprIsNan | rbbyIsNan | rczdIsNan        
        elif rain_czd >= 0.25:
            ''' this boolean excludes dates when there is no
                precip at CZD '''       
            zeros = np.zeros((1,len(ar_rbby)))
            rczdIsZero = np.squeeze(np.equal(ar_rczd,zeros).T)                  
            exclude = wdsrfIsNan | wdwprIsNan | rbbyIsNan | rczdIsNan \
                    | rczdIsZero


        tot_rbby = np.round(df.rbby.sum(),3)
        tot_rczd = np.round(df.rczd.sum(),3)

        exc_rbby = np.round(df[exclude].rbby.sum(),3)
        exc_rczd = np.round(df[exclude].rczd.sum(),3)

        inc_rbby = tot_rbby - exc_rbby
        inc_rczd = tot_rczd - exc_rczd

        tot_hrs   = np.round(df.index.size,0).astype(int)
        exc_hours = np.round(exclude.sum(),0).astype(int)
        inc_hours = tot_hrs - exc_hours

        tta_rbby   = np.round(df[df.consecutive].rbby.sum(),3)
        tta_rczd   = np.round(df[df.consecutive].rczd.sum(),3)
        notta_rbby = inc_rbby - tta_rbby
        notta_rczd = inc_rczd - tta_rczd

        exclude_dates = df[exclude].index
        include_dates = df[~exclude].index
        tta_dates     = df[~exclude & df.consecutive].index
        notta_dates   = df[~exclude & ~df.consecutive].index

        tta_hours   = tta_dates.size
        notta_hours = notta_dates.size

        self.time_beg           = time_beg
        self.time_end           = time_end
        self.count_hrs_include  = inc_hours
        self.count_hrs_exclude  = exc_hours
        self.tot_rainfall_bby   = tot_rbby
        self.tot_rainfall_czd   = tot_rczd
        self.inc_rainfall_bby   = inc_rbby
        self.inc_rainfall_czd   = inc_rczd
        self.exc_rainfall_bby   = exc_rbby
        self.exc_rainfall_czd   = exc_rczd        
        self.tta_rainfall_bby   = tta_rbby
        self.tta_rainfall_czd   = tta_rczd
        self.notta_rainfall_bby = notta_rbby
        self.notta_rainfall_czd = notta_rczd
        self.tta_hours          = tta_hours
        self.notta_hours        = notta_hours
        self.wprof_hgt          = wprof.hgt
        self.exclude_dates      = exclude_dates
        self.include_dates      = include_dates
        self.tta_dates          = tta_dates
        self.notta_dates        = notta_dates
        self.df                 = df

        # print('TTA analysis finished')


    def start_df_layer(self,
                       wdir_thres  = None,
                       wdir_layer  = [None,None],  # [meters]
                       rain_bby    = None,
                       rain_czd    = None,
                       nhours      = None):

        '''
            this version uses pandas dataframe similar
            to start_df but uses a layer instead of a 
            level            
        '''

        import pandas as pd

        bby = parse_data.surface('bby', self.year)
        czd = parse_data.surface('czd', self.year)
        wprof = parse_data.windprof(self.year)
        
        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()
        beg_wpr, end_wpr = wprof.check_beg_end()

        ''' trim the head and tail of dataset depending
            on the latest time of the beginning and 
            earliest of the ending '''
        time_beg = max(beg_bby, beg_czd, beg_wpr)
        time_end = min(end_bby, end_czd, end_wpr)

        ''' initializations '''
        onehr = timedelta(hours=1)
        bool_buffer = np.array([False] * nhours)
        tta_bool = np.array([])
        count = 0
        rng = pd.date_range(start = time_beg,
                            end   = time_end,
                            freq  = '1H')


        idx = np.where((wprof.hgt>=wdir_layer[0]) & 
                       (wprof.hgt<wdir_layer[1]))[0]

        wphgt = wprof.hgt[idx]

        ''' columns included in the dataframe '''        
        cols = []        
        wdircol = 'wd_{}-{:2.0f}m'.format(wdir_layer[0],wphgt[-1])
        cols.append(wdircol)
        cols.append('rbby')
        cols.append('rczd')
        cols.append('tta')
        cols.append('consecutive')
        
        ''' create dataframe '''
        df = pd.DataFrame(index=rng,columns=cols)       
        
        
        ''' loop evaluates each time '''
        time = time_beg
        while (time <= time_end):

            if wdir_layer[0] == 0:
                surf_wd = np.array(bby.dframe.loc[time].wdir)
                surf_ws = np.array(bby.dframe.loc[time].wspd)
            else:
                surf_wd = np.array([])
                surf_ws = np.array([])
                
            wpro_wd = np.array(wprof.dframe.loc[time].wdir)[idx]
            wpro_ws = np.array(wprof.dframe.loc[time].wspd)[idx]

            wd = np.append(surf_wd,wpro_wd)
            ws = np.append(surf_ws,wpro_ws)
            
            u = -ws*np.sin(np.radians(wd))
            v = -ws*np.cos(np.radians(wd))
            u_mean = u.mean()
            v_mean = v.mean()
#            ws_mean = np.sqrt(u_mean**2+v_mean**2)
            wd_mean = 270 - np.arctan2(v_mean,u_mean)*180./np.pi
            if wd_mean > 360:
                wd_mean -= 360
            
            
            df.loc[time][wdircol] = wd_mean
            
            pbby = bby.dframe.loc[time].precip
            df.loc[time].rbby = pbby            
            
            pczd = czd.dframe.loc[time].precip
            df.loc[time].rczd = pczd
                
           
#            df.loc[time].wssrf = bby.dframe.loc[time].wspd
#            df.loc[time].wswpr = wprof.dframe.loc[time].wspd[0]

            ''' check conditions '''               
            if wdir_thres:
                if isinstance(wdir_thres,int):
                    cond1 = (wd_mean < wdir_thres)
                elif isinstance(wdir_thres,str):
                    cond1 = parse_operator(wd_mean,wdir_thres)

            if rain_czd:
                cond3 = (pczd >= rain_czd)

            if rain_bby:            
                cond4 = (pbby >= rain_bby)
              
            ''' create joint condition '''
            if wdir_thres and rain_bby and rain_czd:
                tta_condition = cond1 and cond3 and cond4
            elif wdir_thres and rain_czd:
                tta_condition = cond1 and cond3
            elif wdir_thres and rain_bby:
                tta_condition = cond1 and cond4
            else:
                tta_condition = cond1 


            df.loc[time].tta = tta_condition

            ''' construct boolean array indicating
                hourly TTA conditions with minimum
                of nhours '''
            if tta_condition and bool_buffer.all():
                tta_bool = np.append(tta_bool, [True])
            elif tta_condition:
                bool_buffer[count] = True
                count += 1
                if bool_buffer.all():
                    tta_bool = np.append(tta_bool, bool_buffer)
            else:
                bufsum = bool_buffer.sum()
                if bufsum == 0 or bufsum == nhours:
                    tta_bool = np.append(tta_bool, [False])
                else:
                    tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
                ' reset buffer '
                bool_buffer = np.array([False] * nhours)
                count = 0

#
            time += onehr

        df.consecutive = tta_bool.astype(bool)

        ar_wdir = df[wdircol].values.astype(float)
        ar_rbby = df.rbby.values.astype(float)
        ar_rczd = df.rczd.values.astype(float)
        
        wdirIsNan = np.isnan(ar_wdir)
        rbbyIsNan = np.isnan(ar_rbby)
        rczdIsNan = np.isnan(ar_rczd)
        
        
        if rain_czd is None:
            exclude = wdirIsNan | rbbyIsNan | rczdIsNan        
        elif rain_czd >= 0.25:
            ''' this boolean excludes dates when there is no
                precip at CZD '''       
            zeros = np.zeros((1,len(ar_rbby)))
            rczdIsZero = np.squeeze(np.equal(ar_rczd,zeros).T)                  
            exclude = wdirIsNan | rbbyIsNan | rczdIsNan | rczdIsZero


        tot_rbby = np.round(df.rbby.sum(),3)
        tot_rczd = np.round(df.rczd.sum(),3)

        exc_rbby = np.round(df[exclude].rbby.sum(),3)
        exc_rczd = np.round(df[exclude].rczd.sum(),3)

        inc_rbby = tot_rbby - exc_rbby
        inc_rczd = tot_rczd - exc_rczd

        tot_hrs   = np.round(df.index.size,0).astype(int)
        exc_hours = np.round(exclude.sum(),0).astype(int)
        inc_hours = tot_hrs - exc_hours

        tta_rbby   = np.round(df[df.consecutive].rbby.sum(),3)
        tta_rczd   = np.round(df[df.consecutive].rczd.sum(),3)
        notta_rbby = inc_rbby - tta_rbby
        notta_rczd = inc_rczd - tta_rczd

        exclude_dates = df[exclude].index
        include_dates = df[~exclude].index
        tta_dates     = df[~exclude & df.consecutive].index
        notta_dates   = df[~exclude & ~df.consecutive].index

        tta_hours   = tta_dates.size
        notta_hours = notta_dates.size

        self.time_beg           = time_beg
        self.time_end           = time_end
        self.count_hrs_include  = inc_hours
        self.count_hrs_exclude  = exc_hours
        self.tot_rainfall_bby   = tot_rbby
        self.tot_rainfall_czd   = tot_rczd
        self.inc_rainfall_bby   = inc_rbby
        self.inc_rainfall_czd   = inc_rczd
        self.exc_rainfall_bby   = exc_rbby
        self.exc_rainfall_czd   = exc_rczd        
        self.tta_rainfall_bby   = tta_rbby
        self.tta_rainfall_czd   = tta_rczd
        self.notta_rainfall_bby = notta_rbby
        self.notta_rainfall_czd = notta_rczd
        self.tta_hours          = tta_hours
        self.notta_hours        = notta_hours
        self.wprof_hgt          = wprof.hgt
        self.exclude_dates      = exclude_dates
        self.include_dates      = include_dates
        self.tta_dates          = tta_dates
        self.notta_dates        = notta_dates
        self.df                 = df

        # print('TTA analysis finished')


    def print_stats(self,header=False,
                         return_results=False,
                         skip_print=False,
                         bootstrap=False):

        if header:
            hdr='YR TOT TBB NBB TOT TCZ NCZ TTA NTT TTA NTT BBY(%) CZD(%)'.split()

            fmt='{:>6} ' + \
                '{:>6} {:>6} {:>6} ' + \
                '{:>6} {:>6} {:>6} ' + \
                '{:>5} {:>5} ' + \
                '{:>5} {:>5} ' + \
                '{:>5} {:>5}'

            print(fmt.format(*hdr))


        bby_inc = self.inc_rainfall_bby
        bby_tta =  self.tta_rainfall_bby
        bby_notta =  self.notta_rainfall_bby

        czd_inc = self.inc_rainfall_czd
        czd_tta =  self.tta_rainfall_czd
        czd_notta =  self.notta_rainfall_czd        

        tta_ratio = czd_tta/float(bby_tta)
        notta_ratio = czd_notta/float(bby_notta)

        tta_hours = self.tta_hours
        notta_hours = self.notta_hours

        rain_perc_bby = 100*(bby_tta/float(bby_inc))
        rain_perc_czd = 100*(czd_tta/float(czd_inc))

        
        bby_col = '{:6.1f} {:6.1f} {:6.1f} '
        czd_col = '{:6.1f} {:6.1f} {:6.1f} '
        rto_col = '{:5.1f} {:5.1f} '
        hrs_col = '{:5.0f} {:5.0f} '
        prc_col = '{:5.0f} {:5.0f}'


        col0 = '{:6d} '.format(self.year)
        col1 = bby_col.format(bby_inc, bby_tta,bby_notta)
        col2 = czd_col.format(czd_inc, czd_tta, czd_notta)
        col3 = rto_col.format(tta_ratio, notta_ratio)
        col4 = hrs_col.format(tta_hours, notta_hours)
        col5 = prc_col.format(rain_perc_bby, rain_perc_czd)

        if skip_print is False:
            print(col0+col1+col2+col3+col4+col5)

        if return_results is True:
            return np.array([bby_inc, bby_tta, bby_notta,
                             czd_inc, czd_tta, czd_notta,
                             tta_ratio, notta_ratio,
                             tta_hours, notta_hours,
                             rain_perc_bby, rain_perc_czd])

def bootstrap_ratio(data1, data2, num_samples, alpha):

    import numpy.random as npr

    '''
        Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
    '''

    ''' number of data points (data1=data2)
    '''
    n = len(data1)

    ''' num_samples arrays with random indices of data
        allowing repeated indices (sampling with replacement)
    '''
    idx = npr.randint(0, n, (num_samples, n))

    ''' get the samples of random indices '''
    samples1 = data1[idx]
    samples2 = data2[idx]

    ''' get the statistic along axis 1 and sort it'''
    stat1 = np.mean(samples1, axis=1)
    stat2 = np.mean(samples2, axis=1)

    ratio = np.sort(stat1/stat2)

    ''' confidence interval '''
    bot_CI = ratio[int((alpha/2.0)*num_samples)]
    top_CI = ratio[int((1-alpha/2.0)*num_samples)]

    return (bot_CI, top_CI)



def bootstrap(data, num_samples, statistic, alpha):
    
    import numpy.random as npr    
    
    '''
        Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
        (source:http://people.duke.edu/~ccc14/pcfb/analysis.html)
    '''
    
    ''' number of data points '''
    n = len(data)
    
    ''' num_samples arrays with random indices of data
        allowing repeated indices (sampling with replacement)
    '''
    idx = npr.randint(0, n, (num_samples, n))
    
    ''' get the samples of random indices '''
    samples = data[idx]
    
    ''' get the statistic along axis 1 and sort it'''
    stat = np.sort(statistic(samples, 1))

    ''' confidence interval '''    
    bot_CI = stat[int((alpha/2.0)*num_samples)] 
    top_CI = stat[int((1-alpha/2.0)*num_samples)]
    
    return (bot_CI,top_CI)

        
def parse_operator(target, query):

    import operator as op

    split = query.split(',')

    '''one tail query '''
    if len(split) == 1:

        part = split[0].partition('=')  
        if part[-1].isdigit():
            if part[0] == '>':
                resp = op.ge(target,int(part[-1]))
            else:
                resp = op.le(target,int(part[-1]))    
        else:
            if part[0][0] == '>':
                resp = op.gt(target,int(part[0][1:]))
            else:
                resp = op.lt(target,int(part[0][1:]))  

    elif len(split) == 2:
        
        top = split[0][0]  
        va1 = int(split[0][1:])        
        bot = split[1][-1]
        va2 = int(split[1][:-1])

        if top == '[' and bot == ']':
            resp = op.ge(target,va1) & op.le(target,va2)
            
        elif top == '[' and bot == '[':
            resp = op.ge(target,va1) & op.lt(target,va2)
            
        elif top == ']' and bot == ']':
            resp = op.gt(target,va1) & op.le(target,va2)
            
        elif top == ']' and bot == '[':
            resp = op.gt(target,va1) & op.lt(target,va2)
        
    return resp
        
        
        
        
        
    
    