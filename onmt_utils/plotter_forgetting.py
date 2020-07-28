import matplotlib.pyplot as plt
from matplotlib import cm
from onmt_utils.plotter_helper import create_directory
import math
import numpy as np


def plot_forgotten_events(evaluation_obj, png_path='', save=False):
    df = evaluation_obj.split_evaluator.df

    data = df.loc[df[f'first_learnt'] != math.inf, f'forgotten_events'].values

    nbins = max(data) + 1

    u, inv = np.unique(data, return_inverse=True)
    counts = np.bincount(inv)

    plt.bar(u, counts / len(data) * 100, width=1, edgecolor='thistle', color='dodgerblue')
    plt.xticks(np.arange(0, nbins, 1))

    plt.xlabel('Number of forgotten events')
    plt.ylabel('Frequency of examples [%]')
    if save:
        create_directory(png_path)
        plt.savefig(png_path + "/Forgotten_events.pdf", bbox_inches='tight')


def plot_neverlearnt_events(evaluation_obj, png_path='', save=False):
    df = evaluation_obj.split_evaluator.df

    data = df.loc[df[f'first_learnt'] != math.inf, f'forgotten_events'].values
    data2 = df.loc[df[f'first_learnt'] == math.inf, f'forgotten_events'].values

    counts = np.array([len(data), len(data2)])

    plt.bar(['yes', 'no'], counts / len(df) * 100, width=1, edgecolor='thistle', color='dodgerblue')
    plt.axis([-0.5, 1.5, 0, 100])

    plt.xlabel('Learnt events')
    plt.ylabel('Frequency of examples [%]')

    if save:
        create_directory(png_path)
        plt.savefig(png_path + "/NevLearnt_events.pdf", bbox_inches='tight')


def plot_forgotten_events_by_class(evaluation_obj, superclasses=range(12), ylimit=100.0, png_path='', save=False, **kargs):
    plt.figure(**kargs)
    subplot_rows = len(superclasses) + 1
    subplot_cols = 3
    colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(superclasses))]

    for i, cl in enumerate(superclasses):
        plt.subplot(subplot_rows, subplot_cols, i + 1)

        df = evaluation_obj.split_evaluator.df.loc[
            evaluation_obj.split_evaluator.df['superclass_ground_truth'] == str(cl)]
        data = df.loc[df[f'first_learnt'] != math.inf, f'forgotten_events'].values

        nbins = max(data) + 1

        u, inv = np.unique(data, return_inverse=True)
        counts = np.bincount(inv)

        plt.bar(u, counts / len(data) * 100, width=1, edgecolor='thistle', color=colors[i])
        plt.axis([0, nbins, 0, ylimit])
        plt.xticks(np.arange(0, nbins, 1))

        plt.xlabel('# of forgotten events')
        plt.ylabel('Frequency of examples [%]')
        plt.title(f'Class {cl}')

    if save:
        create_directory(png_path)
        plt.savefig(png_path + "/Forgotten_events_byclass.pdf", bbox_inches='tight')


def plot_neverlearnt_events_by_class(evaluation_obj, superclasses=range(12), ylimit=100.0, png_path='', save=False, **kargs):
    plt.figure(**kargs)
    subplot_rows = len(superclasses) + 1
    subplot_cols = 3
    colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(superclasses))]

    for i, cl in enumerate(superclasses):
        plt.subplot(subplot_rows, subplot_cols, i + 1)

        df = evaluation_obj.split_evaluator.df.loc[
            evaluation_obj.split_evaluator.df['superclass_ground_truth'] == str(cl)]
        data = df.loc[df[f'first_learnt'] != math.inf, f'forgotten_events'].values
        data2 = df.loc[df[f'first_learnt'] == math.inf, f'forgotten_events'].values
        counts = np.array([len(data), len(data2)])

        plt.bar(['yes', 'no'], counts / len(df) * 100, width=0.5, edgecolor='thistle', color=colors[i])
        plt.axis([-0.5, 1.5, 0, ylimit])

        plt.xlabel('Learnt events')
        plt.ylabel('Frequency of examples [%]')
        plt.title(f'Class {cl}')

    if save:
        create_directory(png_path)
        plt.savefig(png_path + "/NevLearnt_events_by_class.pdf", bbox_inches='tight')
