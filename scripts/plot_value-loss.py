import pandas
from matplotlib import pyplot
import argparse
import os
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', help='List of log directories', nargs = '*')
args = parser.parse_args()

ax = None
fig, ax = pyplot.subplots()
legend = list()
for dir_ in args.dirs:
    df = pandas.read_csv(os.path.join(dir_, 'progress.csv'))

    legend.append('value ' + os.path.basename(dir_))
    ax = df.plot(x='total_timesteps', y='value_loss', ax=ax, logy=True)

    if 'policy_loss' in df:
        legend.append('policy ' + os.path.basename(dir_))
        df['policy_abs_loss'] = abs(df['policy_loss'])
        # ax = df.plot(x='total_timesteps', y='policy_abs_loss', ax=ax, logy=True, style='--')
        ax = df.plot(x='total_timesteps', y='policy_loss', ax=ax, logy=True, style='--')

pyplot.tight_layout()
fig.canvas.mpl_connect('resize_event', lambda event: pyplot.tight_layout())
pyplot.legend(legend)
pyplot.grid(True)
pyplot.title('Loss')
pyplot.show()

