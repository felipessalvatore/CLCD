import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa
import seaborn as sns  # noqa
import matplotlib.pyplot as plt  # noqa
import itertools  # noqa
from sklearn.metrics import confusion_matrix  # noqa


def simple_step_plot(ylist,
                     yname,
                     title,
                     path,
                     figsize=(4, 4),
                     labels=None):
    """
    Plots values over time.

    :param ylist: list of values lists
    :type ylist: list
    :param yname: value name
    :type yname: str
    :param title: plot's title
    :type title: str
    :param path: path to save plot, use None for not saving
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    :param labels: label for each values list in ylist
    :type range_points: list
    """
    y0 = ylist[0]
    x = np.arange(1, len(y0) + 1, 1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for y in ylist:
        ax.plot(x, y)
    plt.xlabel('step')
    plt.ylabel(yname)
    plt.title(title,
              fontsize=14,
              fontweight='bold')
    plt.grid(True)
    if labels is not None:
        plt.legend(labels,
                   loc='upper left')
    if path is not None:
        plt.savefig(path)


def bar_plot(Xaxis,
             Yaxis,
             df,
             title,
             path,
             xlabel="task",
             ylabel="accuracy",
             hue=None,
             figsize=(9, 9)):
    """
    Plot a barplot

    :param Xaxis: collum
    :type Xaxis: str
    :param Yaxis: collum
    :type Yaxis: str
    :param df: data
    :type df: pd.DataFrame
    :param title: plot's title
    :type title: str
    :param path: path to save plot, use None for not saving
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.barplot(x=Xaxis, y=Yaxis, hue=hue, data=df)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=6),
                    (p.get_x() + p.get_width() / 2.,
                     p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontweight='bold',
                    color='black')
    fig.suptitle(title, fontsize=18, fontweight='bold')
    if path is not None:
        plt.savefig(path)


def plot_histogram_from_labels(labels, labels_legend, path):
    """
    Plot dataset histogram
    :param label_path: array of labels
    :type label_path: np.array
    :param labels_legend: list with the name of labels
    :type labels_legend: list
    :param path: name to save histogram
    :type path: np.str
    """

    data_hist = plt.hist(labels,
                         bins=np.arange(len(labels_legend) + 1) - 0.5,
                         edgecolor='black')
    axes = plt.gca()
    axes.set_ylim([0, len(labels)])

    plt.title("Histogram of {} data points".format(len(labels)))
    plt.xticks(np.arange(len(labels_legend) + 1), labels_legend)
    plt.xlabel("Label")
    plt.ylabel("Frequency")

    for i in range(len(labels_legend)):
        plt.text(data_hist[1][i] + 0.25,
                 data_hist[0][i] + (data_hist[0][i] * 0.01),
                 str(int(data_hist[0][i])))
    plt.savefig(path)
    plt.show()
    plt.close()


def plot_confusion_matrix(truth,
                          predictions,
                          classes,
                          normalize=False,
                          save=False,
                          cmap=plt.cm.Oranges,
                          path="confusion_matrix.png"):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    'cmap' controls the color plot. colors:
    https://matplotlib.org/1.3.1/examples/color/colormaps_reference.html
    :param truth: true labels
    :type truth: np array
    :param predictions: model predictions
    :type predictions: np array
    :param classes: list of classes in order
    :type classes: list
    :param normalize: param to normalize cm matrix
    :type normalize: bool
    :param save: param to save cm plot
    :type save: bool
    :param cmap: plt color map
    :type cmap: plt.cm
    :param path: path to save image
    :type path: str
    """
    acc = np.array(truth) == np.array(predictions)
    size = float(acc.shape[0])
    acc = np.sum(acc.astype("int32")) / size
    title = "Confusion matrix of {0} examples\n accuracy = {1:.6f}".format(int(size),  # noqa
                                                                           acc)
    cm = confusion_matrix(truth, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    if save:
        plt.savefig(path)
