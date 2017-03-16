"""

    Raul Valenzuela
    raul.valenzuela@colorado.edu


"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tta_analysis3 as tta
import pandas as pd
import tta_continuity
from matplotlib import rcParams
from rv_utilities import discrete_cmap

# sns.reset_orig()
sns.set_style("whitegrid")

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
catalog_df = pd.DataFrame(index=n_event,
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
            hours = np.append(hours, hist.sum())
            events = np.append(events, hist.count())

            catalog_df[year][hist.index] = hist.values

    data = {'h': hours,
            'year': range(1998, 2013),
            'th': [th]*hours.size}
    df = pd.DataFrame(data=data)
    hours_df = hours_df.append(df)

    data = {'h': events,
            'year': range(1998, 2013),
            'th': [th] * events.size}
    df = pd.DataFrame(data=data)
    events_df = events_df.append(df)

labs = [str(int(np.mod(y,100.))).zfill(2)
        for y in range(1998,2013)]
panels = ('(a)',
          '(b) Events of 1 or more hours',
          '(c)')
ylims = [[0,250],[0,100],[1.8,4]]
# cls = sns.color_palette('Set3')
dcmap = discrete_cmap(7, base_cmap='Set1')
cls = [dcmap(0),dcmap(1),dcmap(2)]
scale = 1.3
fig,axes = plt.subplots(3,1,
                        figsize=(5*scale, 6*scale),
                        sharex=True)
for n in range(3):
    axes[n].set_gid(n)

sns.barplot(x='year',
            y='h',
            # hue='th',
            data=hours_df,
            ax=axes[0],
            # palette='GnBu',
            color=cls[0]
            )
for p in axes[0].patches:
    val = p.get_height()
    if val != 0.:
        axes[0].text(p.get_x(), val + 3, '{:1.0f}'.format(val))

sns.barplot(x='year',
            y='h',
            # hue='th',
            data=events_df,
            ax=axes[1],
            # palette='GnBu'
            color=cls[1]
            )

for p in axes[1].patches:
    val = p.get_height()
    if val != 0.:
        axes[1].text(p.get_x(), val + 3,
                     '{:1.0f}'.format(val),
                     )

x = hours_df['year'].values
hpe = hours_df['h']/events_df['h']
sns.pointplot(x, hpe, ax=axes[2], color=cls[2])


tx = '$\overline{X}$ = '
for ax, panel_tx, ylim in zip(axes, panels, ylims):
    transf = ax.transAxes
    ax.text(0.05,0.9,panel_tx,
            fontsize=15,
            weight='bold',
            transform=transf)
    ax.set_ylim(ylim)
    " mean line "
    if ax.get_gid() == 2:
        ax.hlines(hpe.mean(), -1, 15,
                  color=cls[2],
                  linestyle='--')
        ax.text(0.5, hpe.mean()+0.02,
                tx+'{:2.1f}'.format(hpe.mean()),
                color=cls[2],
                weight='bold',
                fontsize=15
                )

axes[0].set_xlabel('')
axes[0].set_ylabel('Hours')
# axes[0].legend_.remove()

axes[1].set_xlabel('')
axes[1].set_xticklabels(labs)
# axes[1].legend_.remove()
axes[1].set_ylabel('Events')

axes[2].set_ylabel('Hours/event')
axes[2].set_xlabel('winter season [year]')

tx  = 'Layer-mean Surf-500m hourly winds '
tx += '$<$ {}$^\circ$'.format(thres[0])
plt.suptitle(tx,fontsize=18,weight='bold',y=0.95)

plt.show()

# place = '/Users/raulvalenzuela/Documents/'
# fname = place+'fig_events_per_season_wd140_surf-500m_rain.png'
# plt.savefig(fname, dpi=150, format='png',papertype='letter',
#             bbox_inches='tight')


