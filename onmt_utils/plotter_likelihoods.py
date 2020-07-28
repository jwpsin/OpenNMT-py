import matplotlib.pyplot as plt
from matplotlib import cm
from onmt_utils.metrics import kde_sklearn
from onmt_utils.plotter_helper import create_directory
import numpy as np
import pandas as pd
import math


def likelihoods_pdf(evaluation_obj, n=1, modeltype='FWD', save=False, png_path='',
                    figsize=(6, 5), font=8, bandwidth=0.2, x_min=0.5,
                    kernel='gaussian', histogram=False, ylimit=10,
                    nbins=100):
    """
    modeltype 'FWD' : Function to plot the likelihoods for all experiments. Only the 
     CORRECT predictions are considered. If n > 1 also the likelihoods of the those 
     predictions (if correct) are plotted IN ADDITION to all the previous topn.
 
    modeltype 'RETRO': function to plot the fwd likelihoods for each prediction of 
     all RETRO experiments. Only the CORRECT predictions (eg. the ones which recover 
     the original product) are considered.                                                                                                     
        evaluation_obj : forward or retro model
        n : degree of topn accuracy
        bandwidth : for the fitting of the histogram with Kernel density
        x_min : minimum value of likelihood to be considered
        kernel : for the fitting of the histogram
    """
    colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(evaluation_obj.name_basename_dict))]
    x_grid = np.linspace(x_min, 1.0, 1000)

    plt.figure(figsize=figsize)  # One plot for every split

    for i, exp_name in enumerate(evaluation_obj.name_basename_dict):

        df = evaluation_obj.split_evaluator.df
        data = []

        if modeltype == 'FWD':
            for k in range(1, n + 1):
                x = df[df[f'{exp_name}_top_{k}_correct'] &
                       (df[f'{exp_name}_top{k}_prob'] >= x_min)][f'{exp_name}_top{k}_prob'].values
                data.extend(x)
        elif modeltype == 'RETRO':
            for k in range(1, n + 1):
                x = df[df[f'fwd_{exp_name}_top_{k}_correct'] &
                       (df[f'fwd_{exp_name}_top_{k}_prob'] >= x_min)][f'fwd_{exp_name}_top_{k}_prob'].values
                data.extend(x)
        else:
            assert False, "Sorry, I don't know this type of model"

        pdf = kde_sklearn(np.array(data), x_grid, bandwidth=bandwidth, kernel=kernel)
        plt.plot(x_grid, pdf, color=colors[i], linewidth=2, label=exp_name)

        if histogram:
            plt.hist(data, bins=nbins, density=1, facecolor=colors[i], alpha=0.2,
                     range=(x_min, 1.0), label='hist {}'.format(exp_name))

    plt.xlabel(f'Top{n} Likelihood', fontsize=font)
    plt.ylabel('Density', fontsize=font)

    plt.axis([x_min, 1.0, 0, ylimit])
    plt.legend(loc='upper left')
    plt.title('{}'.format(evaluation_obj.split + ' set'))

    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/Top{n}_Likel{modeltype}_{evaluation_obj.split}.pdf', bbox_inches='tight')


plt.show()


def likelihoods_pdf_allclasses(evaluation_obj, exp_name, n=1, modeltype='FWD', superclasses=range(12),
                               save=False, png_path='', figsize=(6, 5), font=8,
                               bandwidth=0.2, nbins=100, x_min=0.5, kernel='gaussian',
                               histogram=False, ylimit='inf', hspace=0.2, wspace=0.2):
    """
    modeltype 'FWD': Function to plot the likelihoods divided by classes for one experiment.
     Again, just for the CORRECT predictions. Topn plots look also at the topn 
     likelihoods which are correct.
    modeltype 'RETRO': same thigs with the forward likelihoods on the retro predictions
        evaluation_obj : forward or retro models
        exp_name : name of the experiment on which to do the plotting
        n : degree of topn accuracy
        save : if set to True, saves the plot in the path specified by png_path
        png_path : path were to save the plot (needs to end with "/")
        bandwidth : for the fitting of the data with kernel density
        nbins : number of bins for the istogram
        x_min : minimum likelihoods values to consider for the pdf estimation
        kernel : for the fitting of the data with kernel density
    """
    colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(superclasses))]
    x_grid = np.linspace(x_min, 1.0, 1000)

    plt.figure(figsize=figsize)

    subplot_cols = 3
    subplot_rows = 4

    for i, cl in enumerate(superclasses):

        df = evaluation_obj.split_evaluator.df
        data = []

        if modeltype == 'FWD':
            for k in range(1, n + 1):
                x = df[(df[f'{exp_name}_top_{k}_correct']) & (df['superclass_ground_truth'] == str(cl)) &
                       (df[f'{exp_name}_top{k}_prob'] >= x_min)][f'{exp_name}_top{k}_prob'].values
                data.extend(x)
        elif modeltype == 'RETRO':
            for k in range(1, n + 1):
                x = df[(df[f'fwd_{exp_name}_top_{k}_correct']) &
                       (df[f'classes_{exp_name}_top_{k}'].apply(lambda x: x[0]) == str(cl)) &
                       (df[f'fwd_{exp_name}_top_{k}_prob'] >= x_min)][f'fwd_{exp_name}_top_{k}_prob'].values
                data.extend(x)
        else:
            assert False, "Sorry, I don't know this type of model"

        if data:
            pdf = kde_sklearn(np.array(data), x_grid, bandwidth=bandwidth, kernel=kernel)

            plt.subplot(subplot_rows, subplot_cols, i + 1)
            plt.plot(x_grid, pdf, color=colors[cl], linewidth=2, label=cl)
            plt.xlabel(f'Top{n} Likelihood', fontsize=font)
            plt.ylabel('Density', fontsize=font)
            plt.legend(loc='upper left')
            plt.axis([x_min, 1.0, 0, ylimit])

            if histogram:
                plt.hist(data, bins=nbins, density=1, facecolor=colors[i], range=(x_min, 1.0),
                         alpha=0.2, label='hist {}'.format(exp_name))
        else:
            print(f'SORRY! No correct predictions found for class {cl}.')

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('SET: {} EXP: {}'.format(evaluation_obj.split, exp_name), fontsize=font)

    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/Top{n}_Likel{modeltype}_{exp_name}_{evaluation_obj.split}.pdf', bbox_inches='tight')

    plt.show()


def likelihoods_cdf_allclasses(evaluation_obj, exp_name, n=1, modeltype='FWD', superclasses=range(12),
                               save=False, png_path='', figsize=(6, 5), font=8,
                               x_min=0.5, ylimit='inf', hspace=0.2, wspace=0.2):
    """
    modeltype 'FWD': Function to plot the likelihoods divided by classes for one experiment.
     The plots are visualized in terms of the cumulative distributions' 
     Again, just for the CORRECT predictions. Topn plots look also at the topn
     likelihoods which are correct.
    modeltype 'RETRO': same thigs with the forward likelihoods on the retro predictions
        evaluation_obj : forward or retro models
        exp_name : name of the experiment on which to do the plotting
        n : degree of topn accuracy
        save : if set to True, saves the plot in the path specified by png_path
        png_path : path were to save the plot (needs to end with "/")
        x_min : minimum likelihoods values to consider for the pdf estimation
    """
    colors = [cm.twilight(x) for x in np.linspace(0, 1.0, 12)]

    plt.figure(figsize=figsize)

    subplot_cols = 3
    subplot_rows = 4

    for i, cl in enumerate(superclasses):

        df = evaluation_obj.split_evaluator.df
        data = []

        if modeltype == 'FWD':
            for k in range(1, n + 1):
                x = df[(df[f'{exp_name}_top_{k}_correct']) & (df['superclass_ground_truth'] == str(cl)) &
                       (df[f'{exp_name}_top{k}_prob'] >= x_min)][f'{exp_name}_top{k}_prob'].values
                data.extend(x)
        elif modeltype == 'RETRO':
            for k in range(1, n + 1):
                x = df[(df[f'fwd_{exp_name}_top_{k}_correct']) &
                       (df[f'classes_{exp_name}_top_{k}'].apply(lambda x: x[0]) == str(cl)) &
                       (df[f'fwd_{exp_name}_top_{k}_prob'] >= x_min)][f'fwd_{exp_name}_top_{k}_prob'].values
                data.extend(x)
        else:
            assert False, "Sorry, I don't know this type of model"

        if data:

            plt.subplot(subplot_rows, subplot_cols, i + 1)
            data.extend([1.0, x_min])
            sorted_data = sorted(data)

            # plot the cumulative distribution as a step function
            y = [0.0]
            added = np.arange(1, len(data) - 1) / (len(data) - 2)
            y.extend(added)
            y.extend([0.0])
            plt.step(sorted_data, y, linewidth=2, label=cl, color=colors[i], where='post')

            plt.xlabel(f'Top{n} Likelihood', fontsize=font)
            plt.ylabel('CDF', fontsize=font)
            plt.legend(loc='upper left')
            plt.axis([x_min, 1.0, 0, ylimit])

        else:
            print(f'SORRY! No correct predictions found for class {cl}.')

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('SET: {} EXP: {}'.format(evaluation_obj.split, exp_name), fontsize=font)

    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/Top{n}_Likel{modeltype}_{exp_name}_{evaluation_obj.split}_cum.pdf',
                    bbox_inches='tight')

    plt.show()
