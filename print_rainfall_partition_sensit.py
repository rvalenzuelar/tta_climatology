"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu


"""


import tta_analysis3 as tta

years = [1998]
# years = [1998] + range(2001, 2013)




try:
    wd_layer
except NameError:
    out = tta.start(years=years, layer=[0, 500])
    precip_good = out['precip_good']
    precip_good = precip_good[precip_good.czd > 0.25]
    wd_layer = out['wd_layer']
    wd_layer = wd_layer[precip_good.index]

    wd_tta = wd_layer[wd_layer < 150]
    wd_notta = wd_layer[wd_layer >= 150]
    rain_tta = precip_good.loc[wd_tta.index]
    rain_notta = precip_good.loc[wd_notta.index]