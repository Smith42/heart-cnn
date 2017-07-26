from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import tflearn
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import datetime

# Import and preprocess data

if __name__ == "__main__":
    i = int(sys.argv[1]) # i is current kfold
    k = 3 # k folds

    inData = np.load("./3D-2ch-fakedata/data/inData.npy")
    inLabels = np.load("./3D-2ch-fakedata/data/inLabels.npy")
    inLabelsOH = np.eye(2)[inLabels.astype(int)] # One hot encode

    # k fold the data
    kfoldData = np.array_split(inData, k)
    kfoldLabels = np.array_split(inLabels, k)
    kfoldLabelsOH = np.array_split(inLabelsOH, k)

    try:
        spec, sens, auc, tpr, fpr = np.load("./3D-2ch-fakedata/mess.npy")
    except IOError: # FileNotFoundError in python3
        print("./3D-2ch-fakedata/mess.npy not found. It will be created at the end of this pass")
        pass

    # Does spec, sens, and roc exist?
    try:
        spec
    except NameError:
        spec = []
    try:
        sens
    except NameError:
        sens = []
    try:
        auc
    except NameError:
        auc = []
    try:
        tpr
    except NameError:
        tpr = []
    try:
        fpr
    except NameError:
        fpr = []

    # Neural net (two-channel)

    sess = tf.InteractiveSession()
    tf.reset_default_graph()
    tflearn.initializations.normal()

    # Input layer:
    net = tflearn.layers.core.input_data(shape=[None,34,34,34,2])

    # First layer:
    net = tflearn.layers.conv.conv_3d(net, 32, [10,10,10],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    # Second layer:
    net = tflearn.layers.conv.conv_3d(net, 64, [5,5,5],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    # Fully connected layers
    net = tflearn.layers.core.fully_connected(net, 2048, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 512, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    # Output layer:
    net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

    net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.000001, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)

    # Train the model, leaving out the kfold not being used
    dummyData = np.reshape(np.concatenate(kfoldData[:i] + kfoldData[i+1:], axis=0), [-1,34,34,34,2])
    dummyLabels = np.reshape(np.concatenate(kfoldLabelsOH[:i] + kfoldLabelsOH[i+1:], axis=0), [-1, 2])
    model.fit(dummyData, dummyLabels, batch_size=100, n_epoch=400, show_metric=True)

    # Get sensitivity and specificity
    illTest = []
    healthTest = []
    for index, item in enumerate(kfoldLabels[i]):
        if item == 1:
            illTest.append(kfoldData[i][index])
        if item == 0:
            healthTest.append(kfoldData[i][index])

    healthLabel = np.tile([1,0], (len(healthTest), 1))
    illLabel = np.tile([0,1], (len(illTest), 1))
    sens = np.append(sens, model.evaluate(np.array(healthTest), healthLabel))
    spec = np.append(spec, model.evaluate(np.array(illTest), illLabel))

    # Get roc curve data
    predicted = np.array(model.predict(np.array(kfoldData[i])))
    fprs, tprs, th = roc_curve(kfoldLabels[i], predicted[:,1])
    aucs = roc_auc_score(kfoldLabels[i], predicted[:,1])
    auc = np.append(auc, aucs)
    fpr.append(fprs)
    tpr.append(tprs)

    np.save("./3D-2ch-fakedata/mess", (spec, sens, auc, [tpr], [fpr]))
