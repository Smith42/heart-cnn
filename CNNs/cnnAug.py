from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import tflearn
import sklearn
from sklearn.utils import shuffle as mutual_shuf
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
from CNN import getCNN
import scipy
import h5py
import datetime

# Import and preprocess data

if __name__ == "__main__":
    i = int(sys.argv[1]) # current fold
    k = 5 # kfolds
    k_arr = np.arange(k)
    k_arr_m = k_arr[k_arr != i]

    h5_aug = h5py.File("./data/aug_data.h5", "r")
    inData = h5_aug["in_data"]
    inLabelsOH = h5_aug["in_labels"]
    print(inData.shape)

    inData = np.concatenate([h5f["kfold/"+str(k_arr_m[0])], h5f["kfold/"+str(k_arr_m[1])], h5f["kfold/"+str(k_arr_m[2])], h5f["kfold/"+str(k_arr_m[3])]])
    inLabelsOH = np.eye(2)[np.concatenate([h5f["klabel/"+str(k_arr_m[0])], h5f["klabel/"+str(k_arr_m[1])], h5f["klabel/"+str(k_arr_m[2])], h5f["klabel/"+str(k_arr_m[3])]]).astype(int)]
    inData, inLabelsOH = mutual_shuf(inData, inLabelsOH)

    inData_test = np.load("./data/shufData.npy")
    inLabels_test = np.load("./data/shufLab.npy")
    illTest = np.array_split(inData_test[inLabels_test == 1], k)
    healthTest = np.array_split(inData_test[inLabels_test == 0], k)
    illTest = illTest[i]
    healthTest = healthTest[i]
    print(illTest.shape, healthTest.shape)

    # Neural net (two-channel)
    sess = tf.InteractiveSession()
    model = getCNN(2) # 2 classes: healthy, ischaemia

    # Train the model, leaving out the kfold not being used
    model.fit(inData, inLabelsOH, batch_size=100, n_epoch=20, show_metric=True)
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))
    model.save("./models/"+dt+"-augment_data.tflearn")

    # Get sensitivity and specificity
    healthLabel = np.tile([1,0], (len(healthTest), 1))
    illLabel = np.tile([0,1], (len(illTest), 1))
    sens = model.evaluate(np.array(healthTest), healthLabel)
    spec = model.evaluate(np.array(illTest), illLabel)
    inData_test = np.concatenate((healthTest, illTest))
    inLabels_test = np.concatenate((healthLabel, illLabel))[:,1]

    # Get roc curve data
    predicted = model.predict(inData_test[0][np.newaxis,...]) # Dirty hack to save memory..
    for j in np.arange(1, inLabels_test.shape[0]):
        predicted = np.append(predicted, model.predict(inData_test[j][np.newaxis,...]), axis=0)

    fpr, tpr, th = roc_curve(inLabels_test, predicted[:,1])
    auc = roc_auc_score(inLabels_test, predicted[:,1])

    print(spec[0], sens[0], auc)
    savefileacc = "./logs/"+dt+"-"+str(i)+"-augment_data-acc.log"
    savefileroc = "./logs/"+dt+"-"+str(i)+"-augment_data-roc.log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
    h5f.close()
