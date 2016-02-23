

import scipy.io as sio
import numpy as np
import pandas as pd
import os


base_dir = os.path.expanduser('~')


def bby(year=None, hourly=False):

    y = str(year)[-2:]
    fbby = base_dir+'/SURFACE/climatology/BBY'+y+'_Sfcmet'
    matbby = sio.loadmat(fbby)
    sfcbby = matbby['Sfcmet_bby']
    # cols=sfc.dtype.names
    date, tempc, rh, pmb, wspd, wdir, precip = [], [], [], [], [], [], []
    for n in range(sfcbby.size):
        date.append(datenum_to_datetime(sfcbby[:, n][0][0][0][0]))
        tempc.append(sfcbby[:, n][0][1][0][0])
        rh.append(sfcbby[:, n][0][2][0][0])
        pmb.append(sfcbby[:, n][0][3][0][0])
        wspd.append(sfcbby[:, n][0][4][0][0])
        wdir.append(sfcbby[:, n][0][5][0][0])
        precip.append(sfcbby[:, n][0][6][0][0])
    d = {'tempc': tempc, 'rh': rh, 'pmb': pmb, 'wspd': wspd, 'wdir': wdir,
         'precip': precip}
    BBY = pd.DataFrame(data=d, index=date)
    if year == 2001:
        '2001 has weird values beginning the mat file'
        BBY = BBY.ix[67:]
    BBY = quality_control(BBY)

    if hourly:
        return get_statistical(BBY, minutes=60)
    else:
        return BBY


def get_statistical(df, minutes=None):

    '''
    mean for tempc
    mean for rh
    mean for pmb
    mean for wspd, wdir
    sum for precip
    '''
    grp = pd.TimeGrouper(str(minutes)+'T')
    dfg = df.groupby(grp)
    tempc = dfg['tempc'].mean()
    rh = dfg['rh'].mean()
    pmb = dfg['pmb'].mean()
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

    d = {'tempc': tempc, 'rh': rh, 'pmb': pmb, 'wspd': wspd, 'wdir': wdir,
         'precip': precip}
    newdf = pd.DataFrame(data=d, index=newIndex)

    return newdf


def czd(year=None):

    y = str(year)[-2:]
    fczd = base_dir+'/SURFACE/climatology/avg60_CZC'+y+'_nortype'
    matczd = sio.loadmat(fczd)
    rainczd = matczd['avg_czc_sprof_rtype_precip60'][0]
    begd = matczd['avg_czc_sprof_begdayt60'][0]
    # endd=matczd['avg_czc_sprof_enddayt60'][0]
    date = []
    for n in range(begd.size):
        date.append(datenum_to_datetime(begd[n]))
    d = {'precip': rainczd}
    CZD = pd.DataFrame(data=d, index=date)
    CZD = quality_control(CZD)

    return CZD


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

    wind_vector_sum = None
    wdir += 180
    wdir = np.radians(wdir)
    n_wind = len(wspd)
    # print 'new wind'
    if n_wind >= 2:
        for i in range(0, n_wind):
            wind_polar = cmath.rect(wspd[i], wdir[i] - math.pi)
            # print wind_polar
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
            av_wspd = int(round(r*10))/10.0
            return av_wdir, av_wspd
    else:
        return None, None


def datenum_to_datetime(datenum):
    from datetime import datetime, timedelta
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.

    source: https://gist.github.com/vicow
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=int(days)) \
        + timedelta(hours=int(hours)) \
        + timedelta(minutes=int(minutes)) \
        + timedelta(seconds=round(seconds)) \
        - timedelta(days=366)