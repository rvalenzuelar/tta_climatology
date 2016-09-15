
'''
    Raul Valenzuela
    raul.valenzuela@colorado.edu


'''


def get_df(target):

    """
        Input:
        target -> pandas Series with tta timestamp index

        Output:
        time_df -> pandas DataFrame with time gaps and
                   clasification of events
    """

    import pandas as pd
    import numpy as np

    target_time = pd.Series(target.index)
    offset = pd.offsets.Hour(1).delta
    time_del = target_time - target_time.shift()
    time_del.index = target.index
    time_del[0] = offset  # replace NaT

    del_val = time_del.values
    del_clas = np.array([1])
    clas = 1
    ntotal = del_val[1:].size
    h = np.timedelta64(1, 'h')
    for n in range(1, ntotal + 1):

        cond1 = (del_val[n] != h) and (del_val[n - 1] != h)
        cond2 = (del_val[n] != h) and (del_val[n - 1] == h)
        if cond1 or cond2:
            clas += 1

        del_clas = np.append(del_clas, [clas])

    asd = pd.Series(del_clas)
    asd.index = time_del.index
    time_df = pd.concat([time_del, asd], axis=1)
    time_df.columns = ['time_del', 'clasf']

    return time_df
