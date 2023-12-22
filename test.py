import sys
import torch
import os
import csv
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc,precision_recall_curve

# import util
import numpy as np
import pandas as pd

"""
fpr, tpr, thresholds = sklearn.metrics.roc_curve(actual, pred)
plt.plot(fpr, tpr, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
"""
def built_ROC(actual,pred):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()

    fpr["micro"], tpr["micro"], _ = sk.roc_curve(actual.ravel(), pred.ravel())
    roc_auc["micro"] = sk.auc(fpr["micro"], tpr["micro"])
    precision["micro"], recall["micro"], _ = sk.precision_recall_curve(actual.ravel(), pred.ravel())
    average_precision["micro"] = sk.average_precision_score(actual, pred, average="micro")
    print(roc_auc["micro"])
    print(average_precision["micro"])

    return fpr, tpr, roc_auc,precision,recall,average_precision




if __name__ == '__main__':

    #see = np.average(see, axis=0)
    p = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
    t = np.array([[9.63095963e-01,1.75727922e-02], [9.79060829e-01,3.15358713e-02], [7.85742998e-01,6.52132779e-02],[9.64212358e-01,4.93585095e-02],[9.42467809e-01,3.99554558e-02],[9.91547048e-01,3.48307751e-02],[8.61993790e-01,6.66181326e-01] ,[9.62756276e-01,1.89335495e-02],[9.84570563e-01,8.27044435e-03] ,[9.83874083e-01,1.49833700e-02],])
    # 建立built_ROC函数
    built_ROC(p,t)