import matplotlib.pyplot as plt
from onmt_utils.plotter_helper import autolabel, create_directory
import numpy as np


def bar_plot_metrics(labels, cov, class_div, rt_acc, CJSD, split, save=False, png_path='',
                     figsize=(6, 5), font=8, vert_off=(0, 3)):
    labels = labels
    cov = cov
    class_div = class_div
    rt_acc = rt_acc
    CJSD_div = [np.sqrt(elem) * 1e2 for elem in CJSD]

    if labels:  # if the split is not empty

        plt.figure()
        x = np.arange(len(labels))  # the label locations
        width = 0.95  # the width of the bars
        lab_width = width / 4

        fig, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        rects1 = ax.bar(x - width / 2 + lab_width / 2, cov, lab_width, label='Cov', color='steelblue')
        rects2 = ax.bar(x - width / 4 + lab_width / 2, rt_acc, lab_width, label='RT', color='lightsteelblue')
        rects3 = ax2.bar(x + width / 4 - lab_width / 2, CJSD_div, lab_width, label='sqrt(CJSD)*1e2', color='goldenrod')
        rects4 = ax2.bar(x + width / 2 - lab_width / 2, class_div, lab_width, label='ClassDiv', color='indianred')
        ax2.set_ylabel('abs', color='tab:red', fontsize=font)  # we already handle the x-label with ax1
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('%', fontsize=font)
        # plt.ylim(0.0, 110.0)
        ax.set_title(f'Metrics {split} set')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical', fontsize=font)
        ax.legend(loc='center left')
        ax2.legend(loc='center right')
        ax2.set_xticklabels(labels, rotation='vertical', fontsize=font)

        autolabel(rects1, ax, vert_off=vert_off, font=font)
        autolabel(rects2, ax, vert_off=vert_off, font=font)
        autolabel(rects3, ax2, vert_off=vert_off, font=font)
        autolabel(rects4, ax2, vert_off=vert_off, font=font)

        if save:
            create_directory(png_path)
            plt.savefig(png_path + f'/Metrics_byexp_{split}.pdf', bbox_inches='tight')

    plt.show()
