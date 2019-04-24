#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The SVM model
"""

import numpy as np

import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
from sklearn.externals import joblib


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()




xte = np.load('./xte.npy')

yte = np.load('./yte.npy')

filename='savedsvms.sav'
loaded_model = joblib.load(filename)

y_pred = loaded_model.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

# Draw the confusion matrix
plot_cmat(yte, y_pred)
