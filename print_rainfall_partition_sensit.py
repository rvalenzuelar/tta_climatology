"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu

"""

import tta_analysis3 as tta
import tta_continuity
import numpy as np

years = [2004]
# years = [1998] + range(2001, 2013)

params_hours = [{'wdir_thres': 140,
                 'wdir_layer': [0, 500],
                 'rain_czd': 0.25,
                 'nhours': a
                 } for a in [1,2,4,8]]

params_wdir = [{'wdir_thres': a,
                'wdir_layer': [0, 500],
                'rain_czd': 0.25,
                'nhours': 1
                } for a in [120,130,150,160]]

params_layer = [{'wdir_thres': 140,
                 'wdir_layer': [0, 1000],
                 'rain_czd': 0.25,
                 'nhours': 1
                 }]
params_final = [{'wdir_thres': 150,
                 'wdir_layer': [0, 500],
                 'rain_czd': 0.25,
                 'nhours': 2
                 }]

# params = params_hours + params_wdir + params_final
params = params_final

first = True
try:
    wd_layer
except NameError:
    out = tta.start(years=years, layer=params[0]['wdir_layer'])
    precip_good = out['precip_good']

for param in params:

    precip_good = precip_good[precip_good.czd > param['rain_czd']]

    " filter by wind direction "
    wd_layer = out['wd_layer']
    wd_layer = wd_layer[precip_good.index]
    wd_tta = wd_layer[wd_layer < param['wdir_thres']]
    wd_notta = wd_layer[wd_layer >= param['wdir_thres']]

    " filter by continuity "
    time_df = tta_continuity.get_df(wd_tta)
    hist = time_df.clasf.value_counts()
    min_hours = param['nhours']
    query = hist[hist >= min_hours].index
    tta_dates = time_df.loc[time_df['clasf'].isin(query)].index

    " flag timestamps or precip accordingly "
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

    if first:
        # print('layer {}-{}m'.format(layer[0],
        #                             layer[1]))
        cols = ['layer', 'mnhours', 'Wd_Thres',
                'TTczd', 'TTbby', 'ratio', 'hours',
                'NTczd', 'NTbby', 'ratio', 'hours']
        header = '{:>7} ' * len(cols)
        print(header.format(*cols))
        first = False

    col = ' {:d}-{:3d} {:7d} {:7} '
    col += '{:7.1f} {:7.1f} {:7.1f} {:7.0f} '
    col += '{:7.1f} {:7.1f} {:7.1f} {:7.0f}'

    print(col.format(
        param['wdir_layer'][0],
        param['wdir_layer'][1],
        param['nhours'],
        param['wdir_thres'],
        czd_tta, bby_tta, tta_ratio, tta_hours,
        czd_notta, bby_notta, notta_ratio,
        notta_hours
    )
    )

    try:
        print precip_good[precip_good.tta].loc['2004-02-16']
    except KeyError:
        pass
