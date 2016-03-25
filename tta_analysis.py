import numpy as np
import parse_data
import ctext
from datetime import timedelta


class tta_analysis:

    def __init__(self, year=None):
        self.year = year
        self.execpt_bby = None
        self.execpt_czd = None
        self.execpt_wprof = None
        self.tta_precip_czd = None
        self.tta_precip_bby = None
        self.notta_precip_czd = None
        self.notta_precip_bby = None
        self.precip_bby_excluded = None
        self.precip_czd_excluded = None
        self.precip_bby = None
        self.precip_czd = None
        self.bool = None
        self.precip_bby_before_analysis = None
        self.precip_bby_after_analysis = None
        self.precip_czd_before_analysis = None
        self.precip_czd_after_analysis = None
        self.average_wprof_spd = None
        self.average_wprof_dir = None

    def start(self):

        bby = parse_data.surface('bby', self.year)
        czd = parse_data.surface('czd', self.year)
        wprof = parse_data.windprof(self.year)

        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()
        beg_wpr, end_wpr = wprof.check_beg_end()

        time_beg = max(beg_bby, beg_czd, beg_wpr)
        time_end = min(end_bby, end_czd, end_wpr)

        pbby_before = np.nansum(bby.dframe.loc[:time_beg].precip)
        pbby_after = np.nansum(bby.dframe.loc[time_end:].precip)
        pczd_before = np.nansum(czd.dframe.loc[:time_beg].precip)
        pczd_after = np.nansum(czd.dframe.loc[time_end:].precip)

        nwprof_before = len(wprof.dframe.loc[:time_beg].wdir)
        nwprof_after = len(wprof.dframe.loc[time_end:].wdir)

        onehr = timedelta(hours=1)
        time = time_beg
        bool_buffer = np.array([False] * 5)
        precip_czd = []
        precip_bby = []
        precip_bby_excluded = []
        precip_czd_excluded = []
        wpr_wd_inc = []
        wpr_ws_inc = []
        tta_bool = np.array([])
        count = 0
        excepts_bby = 0
        excepts_czd = 0
        excepts_wprof = 0
        count_try = 0
        count_except = 0
        count_while = 0
        while (time <= time_end):
            try:
                surf_wd = bby.dframe.loc[time].wdir
                pbby = bby.dframe.loc[time].precip
                pczd = czd.dframe.loc[time].precip
                wpr_wd0 = wprof.dframe.loc[time].wdir[0]  # first gate

                ''' version 2 results:
                exclude data when there is nan in surf or wp first gate;
                side effect: precip excluded is not counted in KeyError'''
                if surf_wd is None or np.isnan(surf_wd) or np.isnan(wpr_wd0):
                    time += onehr
                    continue

                ''' these are obs included in the analysis, then we
                determine if they are tta or no-tta '''
                wpr_wd_inc.append(wprof.dframe.loc[time].wdir)
                wpr_ws_inc.append(wprof.dframe.loc[time].wspd)
                precip_bby.append(pbby)
                precip_czd.append(pczd)

                tta_condition = (surf_wd <= 125) and (wpr_wd0 <= 170)

                if tta_condition and bool_buffer.all():
                    tta_bool = np.append(tta_bool, [True])
                elif tta_condition:
                    bool_buffer[count] = True
                    count += 1
                    if bool_buffer.all():
                        tta_bool = np.append(tta_bool, bool_buffer)
                else:
                    bufsum = bool_buffer.sum()
                    if bufsum == 0 or bufsum == 5:
                        tta_bool = np.append(tta_bool, [False])
                    else:
                        tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
                    bool_buffer = np.array([False] * 5)
                    count = 0
                count_try += 1
            except KeyError:
                '''
                there is a time gap in one of these:
                '''
                if time in bby.dframe.index:
                    precip_bby_excluded.append(bby.dframe.loc[time].precip)
                else:
                    excepts_bby += 1

                if time in czd.dframe.index:
                    precip_czd_excluded.append(czd.dframe.loc[time].precip)
                else:
                    excepts_czd += 1

                if time in wprof.dframe.index:
                    'do nothing'
                else:
                    excepts_wprof += 1
                count_except += 1

            count_while += 1
            time += onehr

        # print [count_try, count_except, count_while]

        ' fix yr 2001 warning'
        if self.year == 2001 and len(tta_bool) < count_try:
            tta_bool = np.append(tta_bool, [False])

        tta_bool = np.array(tta_bool).astype(bool)
        precip_bby = np.array(precip_bby)
        precip_czd = np.array(precip_czd)
        precip_bby_excluded = np.array(precip_bby_excluded)
        precip_czd_excluded = np.array(precip_czd_excluded)

        self.time_beg = time_beg
        self.time_end = time_end
        self.count_try = count_try
        self.count_except = count_except
        self.count_while = count_while
        self.precip_bby = precip_bby
        self.precip_czd = precip_czd
        self.bool = tta_bool
        self.tta_precip_czd = np.nansum(precip_czd[tta_bool])
        self.tta_precip_bby = np.nansum(precip_bby[tta_bool])
        self.notta_precip_czd = np.nansum(precip_czd[~tta_bool])
        self.notta_precip_bby = np.nansum(precip_bby[~tta_bool])
        self.except_bby = excepts_bby
        self.except_czd = excepts_czd
        self.except_wprof = excepts_wprof
        self.precip_bby_excluded = np.nansum(precip_bby_excluded)
        self.precip_czd_excluded = np.nansum(precip_czd_excluded)
        self.precip_bby_before_analysis = pbby_before
        self.precip_bby_after_analysis = pbby_after
        self.precip_czd_before_analysis = pczd_before
        self.precip_czd_after_analysis = pczd_after
        self.nwprof_before = nwprof_before
        self.nwprof_after = nwprof_after
        self.wprof_wd = np.array(wpr_wd_inc)
        self.wprof_ws = np.array(wpr_ws_inc)
        self.wprof_hgt = wprof.hgt

    def print_count(self):

        ntta_bools = self.bool.size
        nczds = self.precip_czd.size
        nbbys = self.precip_bby.size
        nexcept_bby = self.except_bby
        nexcept_czd = self.except_czd
        nexcept_wprof = self.except_wprof
        nwprof_before = self.nwprof_before
        nwprof_after = self.nwprof_after

        string = 'Year:{}, tta:{}, bby:{}, czd:{}, ' + \
            'exbby:{}, exczd:{}, exwprof:{}, wprof_bef:{}, ' + \
            'wprof_aft:{}'

        if ntta_bools == nczds == nbbys:
            t = string
        else:
            t = ctext(string).red()

        print t.format(self.year,
                       ntta_bools,
                       nbbys,
                       nczds,
                       nexcept_bby,
                       nexcept_czd,
                       nexcept_wprof,
                       nwprof_before,
                       nwprof_after)
