"""
    Raul Valenzuela
    raul.valenzuela@colorado.edu

    Replaces tta_analysis2 with a more efficient algorithm
    based on pandas Series.

    It collects and retrieves pandas Series with
    hourly wind direction and speed
    profiles including surface observations.

    It also retrieves precip hours at CZD and BBY

    First version of this algorithm was implemented in
    windprof_per_component.py

    Moved bootstrap functions from tta_analysis2.py


    Example:
        
        import tta_analysis3 as ta3
        out1 = ta3.preprocess(years=[2003], layer=[0,500])
        params = dict(wdir_thres=150,
                      rain_czd=0.25,
                      nhours=2
                      )
        out2 = ta3.analysis(out1,params) 
"""

import numpy as np


def preprocess(years=None, layer=None, verbose=True):

    import pandas as pd
    import parse_data

    WD = pd.Series()
    WS = pd.Series()
    WD_rain = pd.Series()
    WS_rain = pd.Series()
    precip_good = pd.DataFrame()

    for year in years:

        wpr = parse_data.windprof(year=year)
        bby = parse_data.surface('bby', year=year)
        czd = parse_data.surface('czd', year=year)
        hgt = wpr.hgt

        ' find common time period '
        first_bby = bby.dframe.index[0]
        first_czd = czd.dframe.index[0]
        first_wpr = wpr.dframe.index[0]
        last_bby = bby.dframe.index[-1]
        last_czd = czd.dframe.index[-1]
        last_wpr = wpr.dframe.index[-1]
        first = max(first_bby, first_czd, first_wpr)
        last = min(last_bby, last_czd, last_wpr)

        ' reduce time interval so all start and end at same time '
        wpr = wpr.dframe.loc[first:last]
        bby = bby.dframe.loc[first:last]
        czd = czd.dframe.loc[first:last]

        ' append surface values to windprof to make entire profile '
        surf_wsp = iter(bby.wspd.values.tolist())
        surf_wdr = iter(bby.wdir.values.tolist())
        wsp = wpr.wspd.map(lambda x: [surf_wsp.next()] + x)
        wdr = wpr.wdir.map(lambda x: [surf_wdr.next()] + x)
        hgt = np.append([0],hgt)

        ' check nans on precip '
        precip = pd.concat([bby.precip, czd.precip], axis=1)
        precip.columns = ['bby', 'czd']
        precip_nans = precip.apply(lambda x: x.isnull().any(),
                                   axis=1, reduce=True)
        precip_nans.name = 'precip_nan'
        tx = 'year:{}, any_precip_nan:{:4d}'
        if verbose:
            print(tx.format(year, precip_nans.sum()))

        ' check entire profile nans ( same for ws and wd)'
        prof_nans = wsp.apply(lambda x: np.isnan(x).all())
        prof_nans.name = 'prof_nan'

        ' include only hours when surf and the entire' \
        ' profile is non-missing (profile is allowed to have' \
        ' at least one non-missing)'
        nan_df = pd.concat([precip_nans, prof_nans], axis=1)
        any_nan = nan_df.apply(lambda x: x.any(), axis=1, reduce=True)
        include = ~any_nan
        precip_good = precip_good.append(precip[include])

        ' rainy days at CZD '
        rain_czd = czd.precip > 0

        ' reduce and save to big Series '
        wdr = wdr[include]
        wsp = wsp[include]
        wdr_rain = wdr[rain_czd]
        wsp_rain = wsp[rain_czd]
        WD = WD.append(wdr)
        WS = WS.append(wsp)
        WD_rain = WD_rain.append(wdr_rain)
        WS_rain = WS_rain.append(wsp_rain)

    " compute components "
    WD_sin = WD.apply(lambda x: sin(x))
    WD_cos = WD.apply(lambda x: cos(x))
    U_df = -1 * WS.multiply(WD_sin)
    V_df = -1 * WS.multiply(WD_cos)
    wind_flow_180 = -(U_df * sin(180) + V_df * cos(180))
    wind_flow_90 = U_df * sin(90) + V_df * cos(90)

    " layer-mean"
    layer_idx = np.where((hgt >= layer[0]) &
                         (hgt < layer[1]))[0]
    mean_V = wind_flow_180.apply(lambda x: np.nanmean(x[layer_idx]))
    mean_U = wind_flow_90.apply(lambda x: np.nanmean(x[layer_idx]))
    wd_layer = 270-(np.arctan2(mean_V, mean_U)*180/np.pi)
    wd_layer[wd_layer > 360] -= 360
    wd_layer.name = '{:2.0f}-{:2.0f}m'.format(hgt[layer_idx[0]],
                                        hgt[layer_idx[-1]])

    return dict(WD=WD,
                WS=WS,
                WD_rain=WD_rain,
                WS_rain=WS_rain,
                wd_layer=wd_layer,
                precip=precip,
                precip_good=precip_good)


def analysis(indict, param):

    import tta_continuity

    precip_good = indict['precip_good']

    if param['rain_czd'] is not None:
        precip_good = precip_good[precip_good.czd > param['rain_czd']]

    " filter by wind direction "
    wd_layer = indict['wd_layer']
    wd_layer = wd_layer[precip_good.index]
    wd_tta = wd_layer[wd_layer < param['wdir_thres']]
    # wd_notta = wd_layer[wd_layer >= param['wdir_thres']]
    precip_good['wd_layer'] = np.round(wd_layer,1)

    " filter by continuity "
    time_df = tta_continuity.get_df(wd_tta)
    hist = time_df.clasf.value_counts()
    query = hist[hist >= param['nhours']].index
    tta_dates = time_df.loc[time_df['clasf'].isin(query)].index

    " flag timestamps accordingly "
    precip_good['tta'] = False
    precip_good['tta'].loc[tta_dates] = True

    tta_hours = precip_good[precip_good.tta].index.size
    notta_hours = precip_good[~precip_good.tta].index.size

    rain_bby_tta = precip_good.bby[precip_good.tta].sum()
    rain_czd_tta = precip_good.czd[precip_good.tta].sum()
    rain_bby_ntta = precip_good.bby[~precip_good.tta].sum()
    rain_czd_ntta = precip_good.czd[~precip_good.tta].sum()

    bby_tta = np.round(rain_bby_tta / tta_hours, 1)
    czd_tta = np.round(rain_czd_tta / tta_hours, 1)
    tta_ratio = czd_tta / bby_tta

    bby_notta = np.round(rain_bby_ntta / notta_hours, 1)
    czd_notta = np.round(rain_czd_ntta / notta_hours, 1)
    notta_ratio = czd_notta / bby_notta

    return dict(czd_tta=czd_tta,
                bby_tta=bby_tta,
                tta_ratio=tta_ratio,
                tta_hours=tta_hours,
                czd_notta=czd_notta,
                bby_notta=bby_notta,
                notta_ratio=notta_ratio,
                notta_hours=notta_hours,
                precip=precip_good)


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

    ratio = np.sort(stat1 / stat2)

    ''' confidence interval '''
    bot_CI = ratio[int((alpha / 2.0) * num_samples)]
    top_CI = ratio[int((1 - alpha / 2.0) * num_samples)]

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
    bot_CI = stat[int((alpha / 2.0) * num_samples)]
    top_CI = stat[int((1 - alpha / 2.0) * num_samples)]

    return (bot_CI, top_CI)

def sin(arg):
    return np.sin(np.radians(arg))


def cos(arg):
    return np.cos(np.radians(arg))