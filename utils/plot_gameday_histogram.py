from matplotlib import pyplot as plt


def plot_gameday_hist(df_plot, label):
    plot_data = df_plot[label]
    nbins = int(1 + plot_data.max() - plot_data.min())
    plot_data.hist(bins=nbins)
    plt.axvline(plot_data.mean(), color='k', linestyle='dashed', linewidth=2)
    plt.ylabel('Count')
    plt.xlabel('Points per game day')
    plt.title(label + ' (mean: ' + str(round(plot_data.mean(), 4)) +
              ' | median: ' + str(round(plot_data.median(), 4)) + ')')
