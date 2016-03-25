import parse_data
from ctext import ctext

txtHeader1 = '\nSurface\n{:^35} || {:^35}'
print txtHeader1.format('Beg', 'End')

txtHeader2 = '{:^16} | {:^16} || {:^16} {:^16}'
print txtHeader2.format('BBY', 'CZD', 'CZD', 'BBY')

t = ctext('{}')

for y in [1998] + range(2001, 2013):
        bby = parse_data.surface('bby', y)
        czd = parse_data.surface('czd', y)

        beg_bby, end_bby = bby.check_beg_end()
        beg_czd, end_czd = czd.check_beg_end()

        if beg_czd < beg_bby:
                tb = t.text + ' | ' + t.red()
        else:
                tb = t.text + ' | ' + t.text

        if end_czd > end_bby:
                te = t.red() + ' | ' + t.text
        else:
                te = t.text + ' | ' + t.text

        txtDate = tb + ' || ' + te

        fmt = '%Y-%m-%d %H:%M'
        b_bby = beg_bby.strftime(fmt)
        b_czd = beg_czd.strftime(fmt)
        e_bby = end_bby.strftime(fmt)
        e_czd = end_czd.strftime(fmt)

        print txtDate.format(b_bby, b_czd, e_czd, e_bby)

print '\nBBY windprof dates'
txtHeader = '{:^16} | {:^16}'
print txtHeader.format('Beg', 'End')
for y in [1998] + range(2001, 2013):
        wprof = parse_data.windprof(y)
        beg, end = wprof.check_beg_end()

        txtDate = '{} | {}'
        fmt = '%Y-%m-%d %H:%M'
        b = beg.strftime(fmt)
        e = end.strftime(fmt)
        print txtDate.format(b, e)
