import numpy as np
import parse_data

from ctext import ctext

txtHeader1 = '{:^35} ||| {:^35}'
print txtHeader1.format('beg', 'end')

txtHeader2 = '{:^16} | {:^16} ||| {:^16} {:^16}'
print txtHeader2.format('bby', 'czd', 'czd', 'bby')

t = ctext('{}')

for y in [1998] + range(2001, 2013):
        beg_bby, end_bby = parse_data.bby_surf_dates(y)
        beg_czd, end_czd = parse_data.czd_surf_dates(y)

        if beg_czd < beg_bby:
                tb = t.text + ' | ' + t.red()
        else:
                tb = t.text + ' | ' + t.text

        if end_czd > end_bby:
                te = t.red() + ' | ' + t.text
        else:
                te = t.text + ' | ' + t.text

        txtDate = tb + ' ||| ' + te

        fmt = '%Y-%m-%d %H:%M'
        b_bby = beg_bby.strftime(fmt)
        b_czd = beg_czd.strftime(fmt)
        e_bby = end_bby.strftime(fmt)
        e_czd = end_czd.strftime(fmt)

        print txtDate.format(b_bby, b_czd, e_czd, e_bby)
