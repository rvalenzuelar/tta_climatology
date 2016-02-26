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

    def start(self):

        bby = parse_data.surface('bby', self.year)
        czd = parse_data.surface('czd', self.year)
        wprof = parse_data.windprof(self.year, first_gate=True)

        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()
        beg_wpr, end_wpr = wprof.check_beg_end()

        time_beg = max(beg_bby, beg_czd, beg_wpr)
        time_end = min(end_bby, end_czd, end_wpr)

        pbby_before = np.nansum(bby.dframe.loc[:time_beg].precip)
        pbby_after = np.nansum(bby.dframe.loc[time_end:].precip)
        pczd_before = np.nansum(czd.dframe.loc[:time_beg].precip)
        pczd_after = np.nansum(czd.dframe.loc[time_end:].precip)

        self.precip_bby_before_analysis = pbby_before
        self.precip_bby_after_analysis = pbby_after
        self.precip_czd_before_analysis = pczd_before
        self.precip_czd_after_analysis = pczd_after

        onehr = timedelta(hours=1)
        time = time_beg
        bool_buffer = np.array([False] * 5)
        precip_czd = []
        precip_bby = []
        precip_bby_excluded = []
        precip_czd_excluded = []
        tta_bool = np.array([])
        count = 0
        excepts_bby = 0
        excepts_czd = 0
        excepts_wprof = 0
        while (time <= time_end):

            try:
                surf_wd = bby.dframe.loc[time].wdir
                pbby = bby.dframe.loc[time].precip
                pczd = czd.dframe.loc[time].precip
                wpr_wd = wprof.dframe.loc[time].wdir

                precip_bby.append(pbby)
                precip_czd.append(pczd)

                tta_condition = (surf_wd <= 125) and (wpr_wd <= 170)

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

            time += onehr

        tta_bool = np.array(tta_bool).astype(bool)
        precip_bby = np.array(precip_bby)
        precip_czd = np.array(precip_czd)
        precip_bby_excluded = np.array(precip_bby_excluded)
        precip_czd_excluded = np.array(precip_czd_excluded)

        self.precip_bby = precip_bby
        self.precip_czd = precip_czd
        self.bool = tta_bool
        self.tta_precip_czd = np.nansum(precip_czd[tta_bool])
        self.tta_precip_bby = np.nansum(precip_bby[tta_bool])
        self.notta_precip_czd = np.nansum(precip_czd[~tta_bool])
        self.notta_precip_bby = np.nansum(precip_bby[~tta_bool])
        self.execpt_bby = excepts_bby
        self.execpt_czd = excepts_czd
        self.execpt_wprof = excepts_wprof
        self.precip_bby_excluded = np.nansum(precip_bby_excluded)
        self.precip_czd_excluded = np.nansum(precip_czd_excluded)

    def print_check(self):

        ntta_bools = self.bool.size
        nczds = self.precip_czd.size
        nbbys = self.precip_bby.size
        nexcept_bby = self.execpt_bby
        nexcept_czd = self.execpt_czd

        string = 'Year:{}, ntta:{}, nbby:{}, nexbby:{}, nczd:{}, nexczd:{}'

        if ntta_bools == nczds == nbbys:
            t = string
        else:
            t = ctext(string).red()

        print t.format(self.year,
                       ntta_bools,
                       nbbys, nexcept_bby,
                       nczds, nexcept_czd)
