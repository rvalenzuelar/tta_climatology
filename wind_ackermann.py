
'''

Raul Valenzuela
raul.valenzuela@colorado.edu

Standard deviation of wind direction
based on:

Ackermann 1983 Journal of Climate and Applied Meteorology

'''


def wind_stats(u, v):

    """
    :param u: time series
    :param v: time series
    :return: mean and std dev of wind speed and dir
    """

    m = np.vstack((u, v))
    bad = np.sum(np.isnan(m), axis=0).astype(bool)
    ug = u[~bad]
    vg = v[~bad]

    N = ug.size

    " mean "
    U = ug.sum() / N
    V = vg.sum() / N

    " variance "
    sqr_u = (ug - U) ** 2
    sqr_v = (vg - V) ** 2
    var_u = sqr_u.sum() / N
    var_v = sqr_v.sum() / N

    " covariance "
    a = (ug - U) * (vg - V)
    covar_uv = a.sum() / N

    " mean wind speed and direction "
    S = np.sqrt(U ** 2 + V ** 2)
    D = 270 - np.arctan2(V, U) * 180 / np.pi

    S_std = np.sqrt(U ** 2 * var_u + \
                    V ** 2 * var_v + \
                    2 * U * V * covar_uv) / S

    D_std = np.sqrt(V ** 2 * var_u + \
                    U ** 2 * var_v - \
                    2 * U * V * covar_uv) / S ** 2

    return S, S_std, D, D_std
