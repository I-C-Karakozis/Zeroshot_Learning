################### Functions for Logging ###################
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def print_border():
    print("-" * 20)

def print_mode_info(args, ckpt, device, n_classes):
    if args.model is None: print("Mode: Training {}".format(ckpt))
    else: print("Mode: Evaluating {}".format(ckpt)) 
    print("Python {}".format(sys.version))
    print("Device: {}".format(device))
    print("Number of Classes: {}".format(n_classes))
    print("Parameters: ", args)
    sys.stdout.flush()

def log_metrics(loss, n_batches, correct=None, total=None, cls_threshold=None):
    stats = 'Loss: %.2f ' % (loss / (n_batches))
    if correct is not None:
        acc = torch.mean(correct).item() / total
        stats += '| Mean per Class Accuracy: %.2f @ t=%.2f' % (acc, cls_threshold) 

    print(stats); print_border()
    sys.stdout.flush()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        title = 'Normalized Confusion Matrix'
        file_path = 'figures/confusion_matrix_normalized'
    else:
        title = 'Confusion Matrix'
        file_path = 'figures/confusion_matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(file_path)
