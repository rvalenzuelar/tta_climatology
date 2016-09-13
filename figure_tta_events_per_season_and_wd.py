"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu


"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tta_analysis3 as tta
import pandas as pd
from matplotlib import rcParams
from rv_utilities import discrete_cmap

sns.reset_orig()

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['mathtext.default'] = 'sf'


# years = [1998]
years = [1998] + range(2001, 2013)

try:
    wd_layer
except NameError:
    out = tta.start(years=years,layer=[0,500])
    wd_layer = out['wd_layer'][out['WD_rain'].index]

thres = 150

for year in years:

    target = wd_layer[str(year)][wd_layer[str(year)] < thres]
    target_time = pd.Series(target.index)
    offset = pd.offsets.Hour(1).delta
    time_del = target_time - target_time.shift()
    time_del.index = target.index

    del_val = time_del.values
    del_clas = np.array([1])
    clas = 1
    ntotal = del_val[1:].size
    h = np.timedelta64(1, 'h')
    for n in range(1,ntotal+1):

        if (del_val[n] != h) and (del_val[n-1] != h) or\
                        (del_val[n] != h) and (del_val[n - 1] == h):
            clas += 1

        del_clas = np.append(del_clas, [clas])

    asd = pd.Series(del_clas)
    asd.index = time_del.index
    time_df = pd.concat([time_del,asd], axis=1)
    time_df.columns = ['time_del','clasf']

hist_events = time_df.clasf.value_counts()
hist_events.hist(bins=hist_events.max())
