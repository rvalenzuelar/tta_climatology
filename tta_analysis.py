import numpy as np
import parse_data
import ctext
from datetime import timedelta


class tta_analysis:

    def __init__(self, year=None):
        self.year = year

    def start(self, wdir_surf=None, wdir_wprof=None, 
              rain_bby=None,rain_czd=None,nhours=None):

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
        wpr_wd_inc = []
        wpr_ws_inc = []
        count = 0
        count_while = 0
        count_exclude = 0

        while (time <= time_end):
                
            surf_wd = bby.dframe.loc[time].wdir
            pbby = bby.dframe.loc[time].precip
            pczd = czd.dframe.loc[time].precip
            wpr_wd0 = wprof.dframe.loc[time].wdir[0]  # first gate


            ''' these are obs included in the analysis, then we
                determine if they are tta or no-tta '''
            rainfall_bby=np.append(rainfall_bby,pbby)
            rainfall_czd=np.append(rainfall_czd,pczd)


            ''' exclude data when there is nan in 
                surf obs or windprof first gate
            '''
            if surf_wd is None or np.isnan(surf_wd) or np.isnan(wpr_wd0):
                tta_bool = np.append(tta_bool, [False])
                count_exclude += 1
                time += onehr
                continue



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

        ' fix yr 2001 warning'
        ntotal = count_while+count_exclude
        if self.year == 2001 and len(tta_bool) < ntotal:
            tta_bool = np.append(tta_bool, [False])

        tta_bool = np.array(tta_bool).astype(bool)
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

    def print_stats(self,header=False):

        if header:
            hdr='TOT TBB NBB TOT TCZ NCZ TTA NTT TTA NTT BBY CZD'.split()
            fmt='{:3} {:3} {:3} ' + \
                '{:4} {:3} {:4} ' + \
                '{:2} {:2} ' + \
                '{:3} {:4} ' + \
                '{:2} {:2}'
            print(fmt.format(*hdr))


        bby_total = self.total_rainfall_bby
        bby_tta =  self.tta_rainfall_bby
        bby_notta =  self.notta_rainfall_bby

        czd_total = self.total_rainfall_czd
        czd_tta =  self.tta_rainfall_czd
        czd_notta =  self.notta_rainfall_czd        

        tta_ratio = czd_tta/float(bby_tta)
        notta_ratio = czd_notta/float(bby_notta)

        tta_hours = self.bool.sum()
        notta_hours = self.count_while-tta_hours

        rain_perc_bby = 100*(bby_tta/float(bby_total))
        rain_perc_czd = 100*(czd_tta/float(czd_total))

        bby_col = '{:3.0f} {:3.0f} {:3.0f} '
        czd_col = '{:4.0f} {:3.0f} {:4.0f} '
        rto_col = '{:2.1f} {:2.1f} '
        hrs_col = '{:3.0f} {:4.0f} '
        prc_col = '%{:2.0f} %{:2.0f}'

        col1 = bby_col.format(bby_total,bby_tta,bby_notta)
        col2 = czd_col.format(czd_total,czd_tta,czd_notta)
        col3 = rto_col.format(tta_ratio, notta_ratio)
        col4 = hrs_col.format(tta_hours, notta_hours)
        col5 = prc_col.format(rain_perc_bby,rain_perc_czd)

        print(col1+col2+col3+col4+col5)
        