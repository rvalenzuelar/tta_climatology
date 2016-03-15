'''  Climatology analysis of
    terrain-trapped airflows

    Run with iPython:
    %run -i tta_climatology.py
    so script can check local namespace

    Raul Valenzuela
    raul.valenzuela@colorado.edu

    Feb-2016
'''


import matplotlib.pyplot as plt
import numpy as np
from tta_analysis import tta_analysis
from ctext import ctext


class clist(list):

    ''' add method to list class
    by subclassing '''

    def insert_nan(self, index=None, qty=None):
        for i in range(qty):
            self.insert(index, np.nan)

'''    Print Table 1
********************************************************'''
txtHeader = '\n{:^5} | {:^9} | {:^9} | {:^9} | {:^9} | {:^9} | {:^9}' +\
    ' | {:^9} | {:^9}'
header = txtHeader.format('Year',
                          'BBY-tta', 'CZD-tta', 'tta-total', 'tta-ratio',
                          'BBY-notta', 'CZD-notta', 'ntta-tot',
                          'ntta-ratio')
print header

totaltxt = ctext('{:9.1f}').green()
ratiotxt = ctext('{:9.1f}').yellow()
txtData = '{:5d} | {:9.1f} | {:9.1f} | ' + totaltxt + ' | ' + \
    ratiotxt + ' | ' + '{:9.1f} | {:9.1f} | ' + totaltxt + ' | ' + ratiotxt
ptta_sum = 0
pnotta_sum = 0

''' if tta_list does not exist in ipython then create it'''
if 'tta_list' not in locals():
    tta_list = []

tta_ratio_list = clist([])
notta_ratio_list = clist([])
ptta_total_list = clist([])
pnotta_total_list = clist([])
tta_precip_czd_list = clist([])
tta_precip_bby_list = clist([])
notta_precip_czd_list = clist([])
notta_precip_bby_list = clist([])

for n, y in enumerate([1998] + range(2001, 2013)):

    ''' if tta_list does not exist then make it '''
    if len(tta_list) == n:
        tta = tta_analysis(y)
        tta.start()
        tta_list.append(tta)

    ''' if tta_list exist then make the table '''
    tta = tta_list[n]
    tta_precip_czd = tta.tta_precip_czd
    tta_precip_bby = tta.tta_precip_bby
    notta_precip_czd = tta.notta_precip_czd
    notta_precip_bby = tta.notta_precip_bby

    ptta = tta_precip_bby + tta_precip_czd
    ptta_sum += ptta
    pnotta = notta_precip_bby + notta_precip_czd
    pnotta_sum += pnotta
    tta_ratio = tta_precip_czd / tta_precip_bby
    notta_ratio = notta_precip_czd / notta_precip_bby
    print txtData.format(y,
                         tta_precip_bby,
                         tta_precip_czd,
                         ptta,
                         tta_ratio,
                         notta_precip_bby,
                         notta_precip_czd,
                         pnotta,
                         notta_ratio)

    tta_ratio_list.append(tta_ratio)
    notta_ratio_list.append(notta_ratio)
    ptta_total_list.append(ptta)
    pnotta_total_list.append(pnotta)
    tta_precip_czd_list.append(tta_precip_czd)
    tta_precip_bby_list.append(tta_precip_bby)
    notta_precip_czd_list.append(notta_precip_czd)
    notta_precip_bby_list.append(notta_precip_bby)

# print '-' * len(header)
# txtFooter = '{:^9} | {:9.1f} | {:9.1f} | ' + totaltxt + ' | ' + ratiotxt + \
#             ' | {:9.1f} | {:9.1f} | ' + totaltxt + ' | ' + ratiotxt
# footer = txtFooter.format('Total',
#                           tta_precip_bby.sum(),
#                           tta_precip_czd.sum(),
#                           ptta_sum,
#                           tta_precip_czd.sum() / tta_precip_bby.sum(),
#                           notta_precip_bby.sum(),
#                           notta_precip_czd.sum(),
#                           pnotta_sum,
#                           notta_precip_czd.sum() / tta_precip_bby.sum(),
#                           )
# print footer

'''    Print Table 2
********************************************************'''
txtHeader = '\n{:^5} | {:^13} | {:^13} | {:^13} | {:^13} | {:^13} | {:^13}'
header = txtHeader.format('Year',
                          'BBY(tta+ntta)', 'BBY-ex', 'BBY-out',
                          'CZD(tta+ntta)', 'CZD-ex', 'CZD-out')
print header

txtData = '{:5d} | {:13.1f} | {:13.1f} | {:13.1f} | {:13.1f} ' +\
    '| {:13.1f} | {:13.1f}'
for n, y in enumerate([1998] + range(2001, 2013)):
    tta = tta_list[n]
    bby_total = tta.tta_precip_bby + tta.notta_precip_bby
    bby_excluded = tta.precip_bby_excluded
    czd_total = tta.tta_precip_czd + tta.notta_precip_czd
    czd_excluded = tta.precip_czd_excluded
    bby_before = tta.precip_bby_before_analysis
    bby_after = tta.precip_bby_after_analysis
    bby_out = bby_before + bby_after
    czd_before = tta.precip_czd_before_analysis
    czd_after = tta.precip_czd_after_analysis
    czd_out = czd_before + czd_after
    if czd_excluded is None:
        czd_excluded = 0
    print txtData.format(y,
                         bby_total,
                         bby_excluded,
                         bby_out,
                         czd_total, czd_out,
                         czd_excluded,
                         czd_out)


'''    Plot ratios
********************************************************'''
tta_ratio_list.insert_nan(1, 2)
notta_ratio_list.insert_nan(1, 2)
fig, ax = plt.subplots()
ax.plot(range(1998, 2013), tta_ratio_list, 'o-', label='TTA ratio')
ax.plot(range(1998, 2013), notta_ratio_list, 'o-', label='NOTTA ratio')
ax.set_xlim([1997.8, 2012.2])
plt.legend(loc=2)
plt.show(block=False)


'''    Plot tta and notta totals
********************************************************'''
ptta_total_list.insert_nan(1, 2)
pnotta_total_list.insert_nan(1, 2)
tta_precip_czd_list.insert_nan(1, 2)
tta_precip_bby_list.insert_nan(1, 2)
notta_precip_czd_list.insert_nan(1, 2)
notta_precip_bby_list.insert_nan(1, 2)
fig, ax = plt.subplots()
ax.plot(range(1998, 2013), ptta_total_list, 'o-',
        label='TTA total precip', color=(0, 0, 1))
ax.plot(range(1998, 2013), tta_precip_czd_list,
        's-', label='TTA czd', color=(0, 0.1, 0.8))
ax.plot(range(1998, 2013), tta_precip_bby_list,
        's-', label='TTA bby', color=(0.1, 0, 0.6))
ax.plot(range(1998, 2013), pnotta_total_list, 'o-',
        label='NOTTA total precip', color=(0, 1, 0))
ax.plot(range(1998, 2013), notta_precip_czd_list,
        's-', label='NOTTA czd', color=(0, 0.8, 0.1))
ax.plot(range(1998, 2013), notta_precip_bby_list,
        's-', label='NOTTA bby', color=(0.1, 0.6, 0))
ax.set_xlim([1997.8, 2012.2])
ax.set_ylim([0, 4000])
plt.legend(loc=2, ncol=2)
plt.show(block=False)
