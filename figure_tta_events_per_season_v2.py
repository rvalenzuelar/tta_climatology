"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu


"""

import matplotlib.pyplot as plt
import numpy as np
import tta_analysis3 as tta
import pandas as pd
import tta_continuity
from matplotlib import rcParams
from rv_utilities import discrete_cmap

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['mathtext.default'] = 'sf'

# years = [1998]
years = [1998] + range(2001, 2013)

try:
    wd_layer
except NameError:
    out = tta.preprocess(years=years,layer=[0,500])
    wd_layer = out['wd_layer'][out['WD_rain'].index]

hours_df = pd.DataFrame()
events_df = pd.DataFrame()

thres = [140]
max_events = 55
n_event = range(1, max_events)
catalog = pd.DataFrame(index=n_event,
                          columns=years)

for th in thres:

    hours = np.array([])
    events = np.array([])

    for year in range(1998,2013):

        if year in [1999, 2000]:
            hours = np.append(hours, [0])
            events = np.append(events, [0])
        else:
            target = wd_layer[str(year)][wd_layer[str(year)] < th]
            time_df = tta_continuity.get_df(target)
            hist = time_df.clasf.value_counts()

            catalog[year][hist.index] = hist.values

dcmap = discrete_cmap(7, base_cmap='Set1')
cls = [dcmap(0),dcmap(1),dcmap(2)]
scale = 1.3


''' histogram '''
# axes = catalog.fillna(0).hist(bins=np.arange(0.5,22.5),
#                           sharex=True,
#                           sharey=True,
#                           grid=False)
# axes = axes.flatten()
# means = list()
# for ax in axes[:-3]:
#     ax.set_xlim([0, 21.5])
#     ax.set_ylim([0, 35])
#     title = ax.get_title()
#     ax.set_title(title, x=0.8, y=0.75)
#     median = np.percentile(catalog[:][int(title)].dropna(),q=50)
#     mean = np.mean(catalog[:][int(title)].dropna())
#     ax.text(0.95,0.65,'median:{}\nmean:{:2.1f}'.format(median,mean),
#             ha='right',va='top',
#             transform=ax.transAxes)
#     means.append(mean)
#
# for ax in [axes[0], axes[4], axes[8], axes[12]]:
#     ytl = ax.get_yticklabels()
#     for l in ytl[1::2]:
#         l.set_visible(False)
#
# axes[12].set_xlabel('n$^{\circ}$ of hours')
# axes[4].set_ylabel('n$^{\circ}$ of events')
#
# plt.subplots_adjust(top=0.98, hspace=0.15, wspace=0.1)

''' boxplot '''
data = list()
for year in years:
    data.append(catalog[:][year].dropna().tolist())

fig,ax = plt.subplots()
# bp = ax.boxplot(data,whis='range',showmeans=True)
bp = ax.boxplot(data,
                 whis='range',
                 sym='',
                 showmeans=True,
                 showcaps=False,
                 whiskerprops={
                     'linestyle': '-',
                     'color': 'k'},
                 meanprops={
                     'marker': 'd',
                     'markersize': 0,
                     'markeredgecolor': None,
                     'markerfacecolor': 'r'},
                 medianprops={'color': 'r',
                              'linewidth':0},
                 boxprops={'color': 'k'}
                 )
labs = [str(int(np.mod(y,100.))).zfill(2)
        for y in years]

# median = np.mean([p.get_ydata()[0] for p in bp['medians']])
# mean = np.mean([p.get_ydata()[0] for p in bp['means']])

d=list()
for l in data:
    d.extend(l)
median = np.median(d)
mean = np.mean(d)

ax.hlines(y=median, xmin=0, xmax=22, linestyle='--',
          color='r', zorder=10000)

ax.hlines(y=mean, xmin=0, xmax=22, linestyle='-',
          color='r', zorder=10000)

yticks = [0] + [np.round(median,1), np.round(mean,1)] + range(5,25,5)

ax.set_yticks(yticks)

for label in ax.get_yticklabels()[1:3]:
    label.set_color('r')

ax.set_ylim([0,22])
ax.set_xticklabels(labs)
ax.set_xlabel('winter season [year]')
ax.set_ylabel('TTA duration [hours]')

place = '/Users/raulvalenzuela/Documents/'
fname = place+'fig_events_per_season_boxplot.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')


