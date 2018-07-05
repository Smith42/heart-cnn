from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import tflearn
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
from CNN import getCNN
import scipy
import h5py
import datetime

# Import and preprocess data

if __name__ == "__main__":
    h5f = h5py.File("./data/infarction-healthy.h5", "r")
    h5f_test = h5py.File("./data/infarction-healthy-test.h5", "r")
    inData = h5f["inData"]
    inLabelsOH = h5f["inLabels"]
    inData_test = h5f_test["inData"]
    inLabelsOH_test = h5f_test["inLabels"]

    # Neural net (two-channel)
    sess = tf.InteractiveSession()
    model = getCNN(2) # 2 classes: healthy, infarcted

    # Train the model, leaving out the kfold not being used
    model.fit(inData, inLabelsOH, batch_size=100, n_epoch=20, show_metric=True)
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))
    model.save("./models/"+dt+"_3d-2channel-fakedata_infarction.tflearn")

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
    savefileacc = "./logs/"+dt+"_3d-2channel-fakedata-acc_infarction.log"
    savefileroc = "./logs/"+dt+"_3d-2channel-fakedata-roc_infarction.log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
    h5f.close()
    h5f_test.close()
