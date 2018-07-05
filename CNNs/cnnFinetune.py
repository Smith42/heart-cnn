from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import tflearn
import sklearn
from CNN import getCNN
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import datetime

# Import and preprocess data

if __name__ == "__main__":
    # The CNN used here should be identical to the one that the model has originally been trained on,
    # but with all but the final few layers not being trainable.

    # This has been written with healthy/ill pairs in mind, but can be generalise to n classes, as in cnnAll.py

    i = int(sys.argv[1]) # i is current kfold
    k = 5 # k folds

    inData = np.load("./data/shufData.npy")
    inLabels = np.load("./data/shufLab.npy")
    inLabelsOH = np.eye(2)[inLabels.astype(int)] # One hot encode

    # k fold the data
    kfoldData = np.array_split(inData, k)
    kfoldLabels = np.array_split(inLabels, k)
    kfoldLabelsOH = np.array_split(inLabelsOH, k)

    # Neural net (two-channel)
    sess = tf.InteractiveSession()
    model = getCNN(2, finetune=True)

    # Train the model, leaving out the kfold not being used
    dummyData = np.reshape(np.concatenate(kfoldData[:i] + kfoldData[i+1:], axis=0), [-1,34,34,34,2])
    dummyLabels = np.reshape(np.concatenate(kfoldLabelsOH[:i] + kfoldLabelsOH[i+1:], axis=0), [-1, 2])
    model.load("./models/placeholder")
    model.fit(dummyData, dummyLabels, batch_size=5, n_epoch=150, show_metric=True) # In practice learning stops ~150 epochs.
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))
    model.save("./models/"+dt+"_3d-2channel-finetuned_"+str(i)+"-of-"+str(k-1)+".tflearn")

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
    sens = model.evaluate(np.array(healthTest), healthLabel)
    spec = model.evaluate(np.array(illTest), illLabel)

    # Get roc curve data
    predicted = np.array(model.predict(np.array(kfoldData[i])))
    fpr, tpr, th = roc_curve(kfoldLabels[i], predicted[:,1])
    auc = roc_auc_score(kfoldLabels[i], predicted[:,1])

    savefileacc = "./logs/"+dt+"_3d-2channel-finetuned-acc_"+str(i)+"-of-"+str(k-1)+".log"
    savefileroc = "./logs/"+dt+"_3d-2channel-finetuned-roc_"+str(i)+"-of-"+str(k-1)+".log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
