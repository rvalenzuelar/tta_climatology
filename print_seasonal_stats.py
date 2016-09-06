'''
    Compute seasonal statistics of TTA and NO-TTA
    precipitation partition

    Raul Valenzuela
    raul.valenzuela@colorado.edu

'''
import numpy as np
from tta_analysis2 import tta_analysis


def params(wdir_surf=None,wdir_wprof=None,
           wdir_thres=None,wdir_layer=None,
           rain_bby=None, rain_czd=None,
           nhours=None):
    
    
    years = [1998]+range(2001,2013)

    if wdir_surf is not None:
        params = dict(wdir_surf  = wdir_surf,
                      wdir_wprof = wdir_wprof,
                      rain_bby   = rain_bby,
                      rain_czd   = rain_czd,
                      nhours     = nhours)
    elif wdir_thres is not None:
        params = dict(wdir_thres  = wdir_thres,
                      wdir_layer  = wdir_layer,
                      rain_bby    = rain_bby,
                      rain_czd    = rain_czd,
                      nhours      = nhours)

    print(params)
    
    for y in years:
        
        tta=tta_analysis(y)
        
        if wdir_surf is not None:
            tta.start_df(**params)
        elif wdir_thres is not None:
            tta.start_df_layer(**params)
        
        if y == 1998:
            results = tta.print_stats(header=True,return_results=True)
        else:
            r = tta.print_stats(return_results=True)
            results = np.vstack((results,r))


    ''' print totals '''
    print('-'*((6*13)+12))
    
    yer_col = '{:6}'
    bby_col = '{:7.0f}{:7.0f}{:7.0f}'
    czd_col = '{:7.0f}{:7.0f}{:7.0f}'
    rto_col = '{:6.1f}{:6.1f}'
    hrs_col = '{:6.0f}{:6.0f}'
    prc_col = '{:6.0f}{:6.0f}'
    
    bby_total   = results[:,0].sum()
    bby_tta     = results[:,1].sum()
    bby_notta   = results[:,2].sum()
    czd_total   = results[:,3].sum()
    czd_tta     = results[:,4].sum()
    czd_notta   = results[:,5].sum()
    tta_ratio   = czd_tta/bby_tta
    notta_ratio = czd_notta/bby_notta
    tta_hours   = results[:,8].sum()
    notta_hours = results[:,9].sum()
    rain_perc_bby = np.round(100.*(bby_tta/bby_total),0).astype(int)
    rain_perc_czd = np.round(100.*(czd_tta/czd_total),0).astype(int)
    
    col0 = yer_col.format('')
    col1 = bby_col.format(bby_total, bby_tta, bby_notta)
    col2 = czd_col.format(czd_total, czd_tta, czd_notta)
    col3 = rto_col.format(tta_ratio, notta_ratio)
    col4 = hrs_col.format(tta_hours, notta_hours)
    col5 = prc_col.format(rain_perc_bby, rain_perc_czd)
    
    print(col0+col1+col2+col3+col4+col5)
    
    