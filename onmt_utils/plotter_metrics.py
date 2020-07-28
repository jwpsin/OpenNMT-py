import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from collections import defaultdict
from onmt_utils.plotter_helper import create_directory
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')


def CJSD_vs_bins(evaluation_obj, save=False, png_path='', figsize=(6, 5),
                 superclasses=range(12), x_min=0.5, bins_range=range(10, 310, 10)):
    CJSDs = defaultdict(list)
    CJSDs_inv = defaultdict(list)
    CJSDs_sqrt = defaultdict(list)
    labels = []

    for exp_name in evaluation_obj.name_basename_dict:
        labels.append(exp_name)

    for bins in bins_range:
        for exp_name in labels:
            CJSD, _, _ = evaluation_obj.get_CJSD(exp_name, superclasses=superclasses, x_min=x_min, nbins=bins)
            CJSDs[exp_name].append(CJSD)
            CJSDs_inv[exp_name].append(1 / CJSD)
            CJSDs_sqrt[exp_name].append(np.sqrt(CJSD))

    col = [cm.twilight(x) for x in np.linspace(0, 1.0, len(labels) + 1)]

    # CJSD
    plt.figure(figsize=figsize)
    for i, exp_name in enumerate(labels):
        plt.plot(bins_range, CJSDs[exp_name], label=exp_name, marker='o', color=col[i])
    plt.xlabel('# of bins')
    plt.ylabel(f'CJSD')
    plt.legend(prop=fontP, loc='upper center', bbox_to_anchor=(0.5, -0.13),
               fancybox=True, shadow=True, ncol=3)
    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/CJSD_{evaluation_obj.split}_{superclasses}.pdf', bbox_inches='tight')
    plt.show()

    # 1/CJSD
    plt.figure(figsize=figsize)
    for i, exp_name in enumerate(labels):
        plt.plot(bins_range, CJSDs_inv[exp_name], label=exp_name, marker='o', color=col[i])
    plt.xlabel('number of bins')
    plt.ylabel('1/CJSD')
    plt.title(f'superclasses {superclasses}, {evaluation_obj.split} set')
    plt.legend(prop=fontP, loc='upper center', bbox_to_anchor=(0.5, -0.13),
               fancybox=True, shadow=True, ncol=3)
    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/CJSD-1_{evaluation_obj.split}_{superclasses}.pdf', bbox_inches='tight')

    # sqrt(CJSD)
    plt.figure(figsize=figsize)
    for i, exp_name in enumerate(labels):
        plt.plot(bins_range, CJSDs_sqrt[exp_name], label=exp_name, marker='o', color=col[i])
    plt.xlabel('number of bins')
    plt.ylabel('sqrt(CJSD)')
    plt.title(f'superclasses {superclasses}, {evaluation_obj.split} set')
    plt.legend(prop=fontP, loc='upper center', bbox_to_anchor=(0.5, -0.13),
               fancybox=True, shadow=True, ncol=3)
    if save:
        create_directory(png_path)
        plt.savefig(png_path + f'/sqrtCJSD_{evaluation_obj.split}_{superclasses}.pdf', bbox_inches='tight')

    return
