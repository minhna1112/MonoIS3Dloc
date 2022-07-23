import matplotlib.pyplot as plt
import pandas as pd
from path import  Path


def plot_loss(log_path: str):
    log_folder = Path('../ivsr_logs')
    plots_folders = Path('../ivsr_plots')


    df = pd.read_csv(log_folder / log_path)
    df = df.dropna()
    df = df.drop(labels='Unnamed: 0', axis=1)
    plot = df.plot()
    #fig = plot[0].get_figure()
    fig = plot.get_figure()
    fig.savefig(plots_folders / log_path[:-3] + 'png' )
    #savefig(plots_folders / log_path[:-3] + 'png' )

    plt.show()

if __name__ == '__main__':
    plot_loss('log0401_3.csv')