

import scipy.io as sio
import numpy as np
import pandas as pd
import os

from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from rv_utilities import datenum_to_datetime

''' global variables '''
base_dir = os.path.expanduser('~')
windprofpath = base_dir + '/WINDPROF/climatology/BBY{}_915lapwind'
surfacepath_bby = base_dir + '/SURFACE/climatology/BBY{}_Sfcmet'
surfacepath_czd = base_dir + '/SURFACE/climatology/avg60_CZC{}_nortype'


class windprof:

    def __init__(self, year=None, first_gate=False):
        '''
        The lowest gate common to all wprof dataset is 158 m
        and the highest is 3753 m. This function interpolate
        to a common grid between 160 and 3750 m with
        92 m of vertical resolution (40 gates)
        '''

        y = str(year)[-2:]
        f = windprofpath.format(y)
        mat = sio.loadmat(f)
        timest = mat['bby915lapwind']['begdayt'][0]
        timestamp = [datenum_to_datetime(t) for t in timest]

        ws = mat['bby915lapwind']['wspd'][0]
        wd = mat['bby915lapwind']['wdir'][0]
        hgt = mat['bby915lapwind']['htmsl'][0]

        newh = np.linspace(160, 3750, 40)
        wspd = np.zeros(len(newh))
        wdir = np.zeros(len(newh))

        first = True
        for s, d, h in zip(ws, wd, hgt):
            ''' for each hourly profile '''
            fs = interp1d(h[0], s[0])
            fd = interp1d(h[0], d[0])
            news = fs(newh)
            newd = fd(newh)
            if first:
                wspd = news
                wdir = newd
                first = False
            else:
                wspd = np.vstack((wspd, news))
                wdir = np.vstack((wdir, newd))

        if first_gate:
            ws = wspd[:, 0]
            wd = wdir[:, 0]
            d = {'wspd': ws, 'wdir': wd}
            self.dframe = pd.DataFrame(data=d, index=timestamp)
            self.time = np.array(timestamp)
        else:
            # self.ws = wspd.T
            # self.wd = wdir.T
            d = {'wspd': wspd.tolist(), 'wdir': wdir.tolist()}
            self.dframe = pd.DataFrame(data=d, index=timestamp)
            self.time = np.array(timestamp)

        self.hgt = newh
        self.year = year

    def check(self):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.imshow(self.ws, aspect='auto', origin='lower', interpolation='none')

        fig.suptitle('BBY Windprof wspd [mm]')
        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)
        plt.show(block=False)

    def check_beg_end(self):

        beg = self.time[0]
        end = self.time[-1]

        return beg, end

    def check_hgt(self, year=None):
        '''
        check altitude of gates  during the
        seasonal observations
        '''
        y = str(self.year)[-2:]
        f = windprofpath.format(y)
        hgt = sio.loadmat(f)['bby915lapwind']['htmsl'][0]
        ng, fg, lg = [], [], []
        for h in hgt:
            ng.append(len(h[0]))
            fg.append(h[0][0])
            lg.append(h[0][-1])

        ngates = np.array(ng)
        firstg = np.array(fg)
        lastg = np.array(lg)

        txt = 'Year: {:4s}, ngates_min:{:4d}, ngates_max:{:4d},' + \
            ' first_gate:{:4d}, last_gate:{:4d}'

        print txt.format(y, int(ngates.min()), int(ngates.max()),
                         int(firstg.max()), int(lastg.min()))

    def check_time_gaps(self):

        gidx, ghrs, gdys = time_gaps(self.time)
        print 'Gaps index'
        print gidx
        print 'Gaps hours'
        print ghrs
        print 'Gaps days'
        print gdys


class surface:

    def __init__(self, location=None, year=None, hourly=True):

        if location == 'bby':

            y = str(year)[-2:]
            f = surfacepath_bby.format(y)
            sfc = sio.loadmat(f)['Sfcmet_bby']
            # cols=sfc.dtype.names
            date, tempc, rh, pmb, wspd, wdir, precip = \
                [], [], [], [], [], [], []
            for n in range(sfc.size):
                date.append(datenum_to_datetime(sfc['dayt'][0][n][0][0]))
                # tempc.append(sfc['tamb'][0][n][0][0])
                # rh.append(sfc['rh'][0][n][0][0])
                # pmb.append(sfc['pmb'][0][n][0][0])
                wspd.append(sfc['wspd'][0][n][0][0])
                wdir.append(sfc['wdir'][0][n][0][0])
                precip.append(sfc['precip'][0][n][0][0])
            d = {'wspd': wspd, 'wdir': wdir, 'precip': precip}
            dframe = pd.DataFrame(data=d, index=date)
            if year == 2001:
                '2001 has weird values beginning the mat file'
                dframe = dframe.ix[67:]
            dframe = quality_control(dframe)

            if hourly:
                self.dframe = get_statistical(dframe, minutes=60)
                self.hourly = True
            else:
                self.dframe = dframe
                self.hourly = False

        elif location == 'czd':

            y = str(year)[-2:]
            f = surfacepath_czd.format(y)
            mat = sio.loadmat(f)
            rainczd = mat['avg_czc_sprof_rtype_precip60'][0]
            begd = mat['avg_czc_sprof_begdayt60'][0]
            # endd=mat['avg_czc_sprof_enddayt60'][0]
            date = []
            for n in range(begd.size):
                date.append(datenum_to_datetime(begd[n]))
            d = {'precip': rainczd}
            dframe = pd.DataFrame(data=d, index=date)
            dframe = quality_control(dframe)
            self.dframe = dframe
            self.hourly = True

    def check_beg_end(self):

        dates = self.dframe.index
        beg = dates[0]
        end = dates[-1]

        return beg, end

    def check_time_gaps(self):

        if self.hourly:
            gidx, ghrs, gdys = time_gaps(self.dframe.index.astype(datetime))
            print 'Gaps index'
            print gidx
            print 'Gaps hours'
            print ghrs
            print 'Gaps days'
            print gdys
        else:
            print 'Data needs to be hourly'


'''
    Common functions
***************************************************
'''


def get_statistical(df, minutes=None):
    '''
    mean for tempc
    mean for rh
    mean for pmb
    mean for wspd, wdir
    sum for precip
    '''
    tempc, rh, pmb, wspd, wdir, precip = [None] * 6

    grp = pd.TimeGrouper(str(minutes) + 'T')
    dfg = df.groupby(grp)

    precip = dfg['precip'].sum()
    wdirh, wspdh = dfg.wdir, dfg.wspd
    wd, ws = [], []
    for gwd, gws in zip(wdirh, wspdh):
        d, s = average_wind(gwd, gws)
        wd.append(d)
        ws.append(s)
    wdir = np.asarray(wd)
    wspd = np.asarray(ws)
    newIndex = precip.index

    d = {'wspd': wspd, 'wdir': wdir, 'precip': precip}
    newdf = pd.DataFrame(data=d, index=newIndex)

    if 'tempc' in df:
        newdf['tempc'] = dfg['tempc'].mean()
    if 'rh' in df:
        newdf['rh'] = dfg['rh'].mean()
    if 'pmb' in df:
        newdf['pmb'] = dfg['pmb'].mean()

    return newdf


def quality_control(df):

    if 'precip' in df:
        cond = df.precip < 0.
        df.precip[cond] = np.nan

    if 'wspd' in df:
        cond = (df.wspd < 0.)
        df.wspd[cond] = np.nan

    if 'wdir' in df:
        cond = (df.wdir < 0.) | (df.wdir > 360.)
        df.wdir[cond] = np.nan

    if 'rh' in df:
        cond = (df.rh < 0) | (df.rh > 100)
        df.rh[cond] = np.nan

    missing_value = -9999.999
    df[df == missing_value] = np.nan

    return df


def average_wind(wdir, wspd):
    '''
    source:
    http://www.intellovations.com/2011/01/16/
    wind-observation-calculations-in-fortran-and-python/

    wdir  -- Pandas DataFrame group (or numpy array)
                of wind directions
    wspd  -- Pandas DataFrame group (or numpy array)
                of wind speeds

    If array is 1D each value is function of time
    If array is 2D axis=0 is height and axis=1 is time
    '''

    import cmath
    import math

    if len(wdir) == 2 and isinstance(wdir, tuple):
        ' pandas instance '
        wdir = wdir[1].values.copy()
        wspd = wspd[1].values.copy()
    else:
        ' numpy instance '
        wdir = wdir.copy()
        wspd = wspd.copy()

    array_dim = len(wdir.shape)
    wdir += 180
    wdir = np.radians(wdir)

    if array_dim == 1:
        ''' if 1D array '''
        n_wind = len(wspd)
        if n_wind >= 2:
            wind_vector_sum = None
            for i in range(n_wind):
                wind_polar = cmath.rect(wspd[i], wdir[i] - math.pi)
                if wind_vector_sum is None:
                    wind_vector_sum = wind_polar
                else:
                    wind_vector_sum += wind_polar

            r, phi = cmath.polar(wind_vector_sum / n_wind)
            if np.isnan(r) and np.isnan(phi):
                return np.nan, np.nan
            else:
                av_wdir = np.round(math.degrees(phi), 1)
                if av_wdir < 0:
                    av_wdir += 360
                av_wspd = int(round(r * 10)) / 10.0
                return av_wdir, av_wspd
        else:
            return None, None
    elif array_dim == 2:
        ''' if 2D array '''
        hn, tn = wdir.shape
        av_wdir_array = np.zeros((hn, 1))
        av_wspd_array = np.zeros((hn, 1))
        av_wsstd_array = np.zeros((hn, 1))
        av_wdstd_array = np.zeros((hn, 1))
        wsnans = np.zeros((hn, 1))
        wdnans = np.zeros((hn, 1))
        for n in range(hn):
            if tn >= 2:
                spd = wspd[n, :]
                dirr = wdir[n, :]
                wind_vector_sum = None
                wind_vectors = np.array([])
                wsnan = 0
                wdnan = 0
                for i in range(tn):
                    if ~np.isnan(spd[i]) and ~np.isnan(dirr[i]):
                        wind_polar = cmath.rect(spd[i], dirr[i] - math.pi)
                        wind_vectors = np.append(wind_vectors, wind_polar)
                    else:
                        wsnan += 1
                        wdnan += 1
                ''' mean '''
                r, phi = cmath.polar(wind_vectors.mean())
                av_wdir = np.round(math.degrees(phi), 1)
                if av_wdir < 0:
                    av_wdir += 360
                av_wspd = int(round(r * 10)) / 10.0
                av_wdir_array[n] = av_wdir
                av_wspd_array[n] = av_wspd
                ''' std dev '''
                re = wind_vectors.real.std()
                im = wind_vectors.imag.std()
                r_std, phi_std = cmath.polar(complex(re, im))
                wdir_std = np.round(math.degrees(phi_std), 1)
                wspd_std = np.round(math.degrees(r_std), 1)
                av_wdstd_array[n] = wdir_std
                av_wsstd_array[n] = wspd_std
                ''' nans '''
                wsnans[n] = wsnan
                wdnans[n] = wdnan
            else:
                av_wdir_array[n] = None
                av_wspd_array[n] = None
                av_wdstd_array[n] = None
                av_wsstd_array[n] = None
        return av_wdir_array, av_wspd_array, av_wdstd_array, av_wsstd_array, wdnans, wsnans
    else:
        print('Arrays need to be 1D or 2D')


def format_xaxis(ax, time_array):
    '''
    assume time_array is in hours
    '''

    nhrs = len(time_array)
    if nhrs > 1 and nhrs <= 72:
        date_fmt1 = '%H\n%d-%b'
        date_fmt2 = '%H'
    else:
        date_fmt1 = '%d\n%b'
        date_fmt2 = '%d'

    ' time is start hour'
    new_xticks = np.asarray(range(len(time_array)))
    xtlabel = []
    for t in time_array:
        if np.mod(t.hour, 3) == 0:
            xtlabel.append(t.strftime(date_fmt))
        else:
            xtlabel.append('')
    ax.set_xticks(new_xticks)
    ax.set_xticklabels(xtlabel)


def format_yaxis(ax, hgt_array, **kwargs):

    hgt_res = np.unique(np.diff(hgt_array))[0]
    if 'toplimit' in kwargs:
        toplimit = kwargs['toplimit']
        ''' extentd hgt to toplimit km so all
        time-height sections have a common yaxis'''
        hgt = np.arange(hgt_array[0], toplimit, hgt_res)
    f = interp1d(hgt, range(len(hgt)))
    ys = np.arange(np.ceil(hgt[0]), hgt[-1], 0.2)
    new_yticks = f(ys)
    ytlabel = ['{:2.1f}'.format(y) for y in ys]
    ax.set_yticks(new_yticks + 0.5)
    ax.set_yticklabels(ytlabel)


def get_date_index(datetime_array, mode):

    good = []
    i = 0
    if mode == 'start_day':
        for i, t in enumerate(datetime_array):
            if (t.hour == 0) and (t.minute == 0):
                good.append(i)
                i += 1
    elif mode == 'start_month':
        for i, t in enumerate(datetime_array):
            if (t.day == 1) and (t.hour == 0) and (t.minute == 0):
                good.append(i)
                i += 1

    return np.array(good)


def time_gaps(datetime_array):
    '''
    assumes hourly spaced datetime_array
    '''

    diff = np.diff(datetime_array)
    gaps_idx = np.where(diff != timedelta(0, 3600))[0]
    if gaps_idx.size > 0:
        gaps = diff[gaps_idx]
        gaps_hours = [(g.seconds / 3600) - 1 for g in gaps]
        gaps_days = [g.days for g in gaps]
        return gaps_idx, gaps_hours, gaps_days
    else:
        return None, None, None
