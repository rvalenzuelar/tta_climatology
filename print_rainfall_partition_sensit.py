"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu

"""

import tta_analysis3 as tta

# years = [2003]
years = [1998] + range(2001, 2013)

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

params = params_hours + params_wdir + params_final
# params = params_final

first = True
try:
    wd_layer
except NameError:
    out = tta.preprocess(years=years, layer=params[0]['wdir_layer'])

for param in params:

    result = tta.analyis(out, param)

    if first:
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
        result['czd_tta'],
        result['bby_tta'],
        result['tta_ratio'],
        result['tta_hours'],
        result['czd_notta'],
        result['bby_notta'],
        result['notta_ratio'],
        result['notta_hours']
    )
    )

    # try:
    #     print precip_good[precip_good.tta].loc['2003-01']
    # except KeyError:
    #     pass
