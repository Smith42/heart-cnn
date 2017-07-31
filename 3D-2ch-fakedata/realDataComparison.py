from __future__ import print_function
import matplotlib     # These are needed to run
matplotlib.use("Agg") # the code headless.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import datetime
import tensorflow as tf
import tflearn

def modelLoad(modelPath):
    """
    Load model from modelPath.
    The CNN defined below must be the same as that used to generate the *.tflearn file.
    """
    sess = tf.InteractiveSession()
    tf.reset_default_graph()

    # Input layer:
    net = tflearn.layers.core.input_data(shape=[None,34,34,34,2])

    # First layer:
    net = tflearn.layers.conv.conv_3d(net, 32, [10,10,10],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    # Second layer:
    net = tflearn.layers.conv.conv_3d(net, 64, [5,5,5],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    net = tflearn.layers.conv.conv_3d(net, 128, [2,2,2], activation="leaky_relu") # This was added for CNN 2017-07-28

    # Fully connected layers
    net = tflearn.layers.core.fully_connected(net, 2048, activation="leaky_relu") # regularizer="L2", weight_decay=0.01,
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 1024, activation="leaky_relu") # regularizer="L2", weight_decay=0.01,
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 512, activation="leaky_relu") # regularizer="L2", weight_decay=0.01,
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    # Output layer:
    net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

    net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    model.load(modelPath)
    return model

if __name__ == "__main__":
    # inData are heartcubes with same shape as those used in the CNN
    inData = np.load("./data/shufData.npy")
    inLabels= np.load("./data/shufLab.npy")
    modelPaths = ["./models/2017-07-27_22:40:00_3d-2channel-fakedata_0-of-5.tflearn","./models/2017-07-28_03:21:00_3d-2channel-fakedata_1-of-5.tflearn","./models/2017-07-28_08:01:00_3d-2channel-fakedata_2-of-5.tflearn","./models/2017-07-28_12:42:00_3d-2channel-fakedata_3-of-5.tflearn","./models/2017-07-28_17:23:00_3d-2channel-fakedata_4-of-5.tflearn"] # Put model paths here.

    i = int(sys.argv[1]) # i is current kfold
    k = 5

    model = modelLoad(modelPaths[i])

    # Get sensitivity and specificity
    illTest = []
    healthTest = []
    for index, item in enumerate(inLabels):
        if item == 1:
            illTest.append(inData[index])
        if item == 0:
            healthTest.append(inData[index])

    healthLabel = np.tile([1,0], (len(healthTest), 1))
    illLabel = np.tile([0,1], (len(illTest), 1))
    sens = model.evaluate(np.array(healthTest), healthLabel)
    spec = model.evaluate(np.array(illTest), illLabel)

    # Get roc curve data
    predicted = np.array(model.predict(np.array(inData)))
    fpr, tpr, th = roc_curve(inLabels, predicted[:,1])
    auc = roc_auc_score(inLabels, predicted[:,1])

    savefileacc = "./logs/"+dt+"_3d-2channel-realdata-acc_"+str(i)+"-of-"+str(k-1)+".log"
    savefileroc = "./logs/"+dt+"_3d-2channel-realdata-roc_"+str(i)+"-of-"+str(k-1)+".log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
