
from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def norm_confusion_matrix(y_true, y_pred):
    print("Normalized Confusion Matrix:")
    matrix =  metrics.confusion_matrix(y_true, y_pred)
    norm_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print(norm_matrix)

def overall_accuracy(y_true, y_pred):
    print("Overall Accuracy Score:")
    print(metrics.accuracy_score(y_true, y_pred))

def get_roc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def print_roc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print ("Area Under the ROC Curve")
    print roc_auc

def plot_roc(y_true, y_pred, fname, title):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    fig=plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    fig.savefig(fname)

def plot_precision_recall(y_true,y_pred, fname):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    plt.clf()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(fname)

def main():
    'plot_roc([0,1,1,0],[1,0,1,0],"")'
    return

if __name__=='__main__':
    main()
