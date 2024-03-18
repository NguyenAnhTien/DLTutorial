"""
@author : Tien Nguyen
@date   : 2024-Mar-18
"""
import numpy

import matplotlib.pyplot as plt

def cal_tpr_fpr(preds, y_test):
    true_positive = numpy.equal(preds, 1) & numpy.equal(y_test, 1)
    true_negative = numpy.equal(preds, 0) & numpy.equal(y_test, 0)
    false_positive = numpy.equal(preds, 1) & numpy.equal(y_test, 0)
    false_negative = numpy.equal(preds, 0) & numpy.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def cal_roc(probs, y_test, thresholds):
    roc = []
    for threshold in thresholds:
        preds = numpy.greater_equal(probs, threshold).astype(int)
        tpr, fpr = cal_tpr_fpr(preds, y_test)
        roc.append([fpr, tpr])
    return roc

def cal_auc(roc):
    area = numpy.trapz(sorted(roc[:, 0]), sorted(roc[:, 1]), dx=0.1)
    return area

if __name__ == '__main__':
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    probs = [0.2, 0.8, 0.7, 0.4, 0.5, 0.6, 0.1, 0.3, 0.9, 1.]
    y_test = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    roc = cal_roc(probs, y_test, thresholds)
    roc = numpy.array(roc)
    auc = cal_auc(roc)
    plt.scatter(roc[:, 0], roc[:, 1])
    plt.plot(roc[:, 0], roc[:, 1])
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.text(0.6, 0.2, 'AUC = {:.2f}'.format(auc), fontsize=12, color='blue')
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.show()
