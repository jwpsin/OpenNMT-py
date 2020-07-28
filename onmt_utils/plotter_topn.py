import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from onmt_utils.plotter_helper import autolabel, create_directory


def topn(evaluation_obj, n=1, save=False, png_path='', figsize=(6, 5),
         vert_off=(0, 3), font=8, ylimit=110.0):
    """
    Function to plot the topn accuracy for ALL experiments in evaluation_obj. Meaning  
    the percentage of predicted sequences which match the target in the top-n list.    
    Note that the percentage is meant over the number of targets. So a positive output 
    is given if any of the topn sequences is correct. As a consequence, the higher the 
    specified n, the higher the accuracy. Requires the previous computation of the     
    accuracies.
        evaluation_obj : forward or retro model
        n : type of topn accuracy
        save : if set to True, saves the plot in the path specified by png_path
        png_path : path were to save the plot (needs to end with "/")
    """

    topn_split, labels = evaluation_obj.get_allexp_top_n_accuracy(topn=n)

    if topn_split:  # if for that split I have results

        plt.figure(figsize=figsize)
        colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(labels))]

        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots(figsize=figsize)
        rects1 = ax.bar(x, topn_split, width, color=colors)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('%')
        plt.ylim(0.0, ylimit)
        ax.set_title('Top{} accuracy - {}'.format(n, evaluation_obj.split))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical', fontsize=font)
        autolabel(rects1, ax, vert_off=vert_off, font=font)

        if save:
            create_directory(png_path)
            plt.savefig(png_path + f'/Top{n}_{evaluation_obj.split}.pdf', bbox_inches='tight')

        plt.show()
    return

def topn_byclass(evaluation_obj, n=1, modeltype='FWD', save=False, png_path='', figsize=(6, 5),
                 vert_off=(0, 3), font=8, ylimit=110.0, superclasses=range(12), hspace=0.2,
                 wspace=0.2):
    """
    Function to plot the topn accuracies divided by classes for ALL experiments in     
    evaluation_obj. Meaning  that for each experiment in a certain split there will a  
    subplot where the percentage of predicted sequences which much the target in the   
    top-n list are plotted and subdivided by classes.                                  
    Note that the percentage is meant over the number of targets. So a positive output 
    is given if any of the topn sequences is correct. As a consequence, the higher the 
    specified n, the higher the accuracy. Requires the previous computation of the     
    accuracies                                                                         
        evaluation_obj : forward or retro model
        n : degree of TOPN accuracy
        save : if set to True, saves the plot in the path specified by png_path
        png_path : path were to save the plot (needs to end with "/")
        superclasses : number of superclasses in which to visualize the accuracy
    """
    plt.figure(figsize=figsize)

    subplot_cols = 3
    subplot_rows = len(evaluation_obj.name_basename_dict)

    for i, exp_name in enumerate(evaluation_obj.name_basename_dict):

        class_topn = []
        class_labels = []

        for cl in superclasses:

            if modeltype == 'FWD':
                df = evaluation_obj.split_evaluator.df.loc[
                    evaluation_obj.split_evaluator.df['superclass_ground_truth'] == str(cl)]

            # to be changed
            elif modeltype == 'RETRO':
                cols = [f'classes_{exp_name}_top_{s}' for s in range(1, n + 1)]
                df = evaluation_obj.split_evaluator.df.loc[evaluation_obj.split_evaluator.df[cols] == str(cl)]
            else:
                raise ValueError("Not known model type")

            class_labels.append(cl)

            correct = 0
            df_filter = [True for i in range(len(df))]

            for k in range(1, n + 1):
                correct += df[df_filter][f'{exp_name}_top_{k}_correct'].sum()
                df_filter = df_filter & (df[f'{exp_name}_top_{k}_correct'] == False)

            class_topn.append(correct / len(df) * 100)

        colors = [cm.twilight(x) for x in np.linspace(0, 1.0, len(class_labels))]
        x = np.arange(len(class_labels))  # the label locations
        width = 0.3  # the width of the bars

        ax = plt.subplot(subplot_rows, subplot_cols, i + 1)
        rects1 = plt.bar(x, class_topn, width, color=colors, tick_label=class_labels)
        autolabel(rects1, ax, vert_off=vert_off, font=font)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel('%')
        plt.ylim(0.0, ylimit)
        plt.xticks(rotation='horizontal')
        plt.title('EXP: {}'.format(exp_name))

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('SET: {} TOP: {}'.format(evaluation_obj.split, n), fontsize=font)
    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/Top{n}_byclass_allexp_{evaluation_obj.split}.pdf', bbox_inches='tight')

    plt.show()

    return
