from __future__ import print_function
import matplotlib     # These are needed to run
matplotlib.use("Agg") # the code headless.

import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import datetime, time
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
    model = tflearn.DNN(net)
    model.load(modelPath)
    return model

def getLoss(inData, inLabel, maskWidth, i, j, k):
    """
    Mask an area of a set of ppt arrays with zeros and get the AUC score for the modified arrays.
    Return AUC score.
    """
    mArr = np.zeros(inData.shape)
    ones = np.ones((inData.shape[0], maskWidth, maskWidth, maskWidth, 2))
    mArr[:,i:i+maskWidth,j:j+maskWidth,k:k+maskWidth] = ones # Set mask array for this index
    mInData = np.ma.MaskedArray(inData, mask=mArr)
    mInData = np.ma.MaskedArray.filled(mInData, fill_value=0)
    predicted = np.array(model.predict(mInData))
    loss = abs(predicted[:,1]-inLabel) # Simple loss function so we can see how close we are to being right
    return loss

def normalise(inData):
    """
    Normalise 3D array.
    """
    inDataAbs = np.fabs(inData)
    inDataMax = np.amax(inData)
    normalisedData = inDataAbs/inDataMax
    return normalisedData

if __name__ == "__main__":
    # inData are heartcubes with same shape as those used in the CNN
    ppt = 20
    inData = np.load("./3D-2ch-fakedata/data/inData.npy")[ppt]
    inData = inData[np.newaxis,...]
    inLabels= np.load("./3D-2ch-fakedata/data/inLabels.npy")[ppt]

    model = modelLoad("./models/2017-07-27_22:40:00_3d-2channel-fakedata_0-of-5.tflearn")

    # Does the CNN predict correctly in the first place?
    p = model.predict(inData[0])[:,1]
    if abs(p - inLabels) >= 0.5:
        print("Model predicts", p, "but we want", inLabels, "quitting...")
        exit()

    maskWidth = 6 # Might be more representative to have this as even.
    lossCube = np.zeros(inData.shape[1:4])

    for i in np.arange(inData.shape[1] - maskWidth + 1):
        for j in np.arange(inData.shape[2] - maskWidth + 1):
            for k in np.arange(inData.shape[3] - maskWidth + 1):
                loss = getLoss(inData, inLabels, maskWidth, i, j, k)
                lossCube[i+maskWidth/2,j+maskWidth/2,k+maskWidth/2] = loss
                print(i+maskWidth/2,j+maskWidth/2,k+maskWidth/2,":",loss)

    lossCube = normalise(lossCube)

    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))
    np.save("./logs/lossCubes/"+dt+"_ppt"+str(ppt)+"_lossCube", lossCube)
    np.save("./logs/lossCubes/"+dt+"_ppt"+str(ppt)+"_heartCube", inData[0])
