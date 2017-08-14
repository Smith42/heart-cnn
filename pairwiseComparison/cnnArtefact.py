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
import h5py
import datetime

# Import and preprocess data

if __name__ == "__main__":
    h5f = h5py.File("./data/artefact-healthy.h5", "r")
    h5f_test = h5py.File("./data/artefact-healthy-test.h5", "r")
    inData = h5f["inData"]
    inLabelsOH = h5f["inLabels"]
    inData_test = h5f_test["inData"]
    inLabelsOH_test = h5f_test["inLabels"]

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

    # Third layer:
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
    model = tflearn.DNN(net, tensorboard_verbose=0)

    # Train the model, leaving out the kfold not being used
    model.fit(inData, inLabelsOH, batch_size=100, n_epoch=20, show_metric=True)
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))
    model.save("./models/"+dt+"_3d-2channel-fakedata_artefact.tflearn")

    # Get sensitivity and specificity
    illTest = []
    healthTest = []
    inLabels_test = inLabelsOH_test[:,1]
    for index, item in enumerate(inLabels_test):
        if item == 1:
            illTest.append(inData_test[index])
        if item == 0:
            healthTest.append(inData_test[index])

    healthLabel = np.tile([1,0], (len(healthTest), 1))
    illLabel = np.tile([0,1], (len(illTest), 1))
    sens = model.evaluate(np.array(healthTest), healthLabel)
    spec = model.evaluate(np.array(illTest), illLabel)

    # Get roc curve data
    predicted = model.predict(inData_test[0][np.newaxis,...]) # Dirty hack to save memory..
    for j in np.arange(1, inLabels_test.shape[0]):
        predicted = np.append(predicted, model.predict(inData_test[j][np.newaxis,...]), axis=0)

    fpr, tpr, th = roc_curve(inLabels_test, predicted[:,1])
    auc = roc_auc_score(inLabels_test, predicted[:,1])

    print(spec[0], sens[0], auc)
    savefileacc = "./logs/"+dt+"_3d-2channel-fakedata-acc_artefact.log"
    savefileroc = "./logs/"+dt+"_3d-2channel-fakedata-roc_artefact.log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
    h5f.close()
    h5f_test.close()
