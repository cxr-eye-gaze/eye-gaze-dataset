import os, logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from textwrap import wrap
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

logger = logging.getLogger('eyegaze')


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, server, env_name='main'):
        self.viz = server
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def get_cmap(n, name='Set1'):
    """
    Source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html
    :param n: Number of classes.
    :param name: The color map to be selected from.
    :return: a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def plot_roc_curve(tpr, fpr, class_names, aurocs, filename, name_variable):
    """
    Modified from scikit-example. Plots the ROC curve for the different classes.
    :param tpr: true positive rate computed
    :param fpr: false positive rate computed
    :param class_names:
    :param aurocs: computed area under the rocs
    :param filename: the output directory where the file should be written
    :param name_variable: the filename with the extension.
    :return: None (writes the file to disk)
    """
    #
    # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    a = []
    for i in range(len(class_names)):
        if (i in fpr) == True:
            # print(class_names[i])
            # The np.concatenate does not work on scalars and zero dimensions hence the expand dims.
            a.append(np.concatenate(np.expand_dims(fpr[i], axis=1)))
        else:
            logger.info('--'*30)
            logger.info(f"WARNING!!! No: {class_names[i]} found in the set ")
            logger.info('--'*30)

    #flatten the concatenated list.
    flat_list = []
    for sublist in a:
        for item in sublist:
            flat_list.append(item)
    flat_np = np.asarray(flat_list)
    all_fpr = np.unique(flat_np)

    #Now interpolate them results
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        # This is essentially saying interpolate all the curves to the combined
        # number of elements using the fpr and tpr of each of these.
        if(i in fpr) == True:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    # mean_tpr /= len(class_names)
    # it's actually not considering the missing classes;
    mean_tpr /= len(a)
    fpr["all_val"] = all_fpr
    tpr["all_val"] = mean_tpr
    roc_auc = auc(fpr["all_val"], tpr["all_val"])
    lw = 2
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111)
    if len(class_names) > 1:
        ax.plot(fpr["all_val"], tpr["all_val"],
                 label='average ROC(area = {0:0.2f})'
                       ''.format(roc_auc),
                 color='navy', linestyle=':', linewidth=4)

    colors = get_cmap(len(class_names))
    for i in range(len(class_names)):
        if( i in fpr) == True:
            ax.plot(fpr[i], tpr[i], color=colors(i), lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(class_names[i], aurocs[i]))
            # ax.plot(fpr[i], tpr[i], color=colors(i), lw=lw)
    # Plot the 45 degree line.
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR/Fallout)')
    ax.set_ylabel('True Positive Rate (TPR/Sensitivity)')
    ax.set_title("\n".join(wrap(f'{name_variable}')))

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # # Customize the minor grid
    ax.grid(which='minor', linestyle='solid', linewidth='0.25', color='black')
    # # ax.show()

    fig.savefig(os.path.join(filename, 'ROC_'+name_variable), bbox_inches='tight')
    # --- Close the figure and clear up the axis to free memory.
    plt.cla()
    plt.close()


def plot_precision_recall(y, y_hat, class_names, filename, name_variable):
    """
    From Scikit example. Plots the precision-recall for multiple classes
    :param y: Ground truth labels
    :param y_hat: Model predicted labels
    :param class_names:
    :param filename: the output directory where the file should be written
    :param name_variable: the filename with the extension.
    :return: None (file gets written to disk)
    """
    Y_test = np.asarray(y)
    y_score = np.asarray(y_hat)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(class_names)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[i],
                                                            y_score[i])
        average_precision[i] = average_precision_score(Y_test[i], y_score[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    logger.info('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # setup plot details
    colors = get_cmap(n_classes)

    # plt.figure(figsize=(7, 8))
    plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)

    if n_classes > 1:
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], color=colors(i), lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.2f})'
                      ''.format(class_names[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.title("\n".join(wrap(f'Precision-Recall -- {name_variable}')))
    # Put a legend to the right of the current axis
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # Turn on the minor TICKS, which are required for the minor GRID
    plt.minorticks_on()
    # Customize the major grid
    plt.savefig(os.path.join(filename,"Precision-Recall_"+name_variable), bbox_inches='tight')
    # --- Close the figure and clear up the axis to free memory.
    plt.cla()
    plt.close()
