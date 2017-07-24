from __future__ import print_function
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tflearn
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import datetime

# Import and preprocess data

def importHeartData(calmFile, stressFile, resize):
    """
    Import heart data and extract the pixel array.
    Slice halfway along ind axis.
    Concatenate and return stress file and calm file.
    If resize == 1, interpolate data to fit (34,34) arr.
    """
    calmTmp = pyfits.open(calmFile)[0].data
    stressTmp = pyfits.open(stressFile)[0].data

    calmTmp = cropHeart(calmTmp)
    stressTmp = cropHeart(stressTmp)

    # Pad the 3d arrays with zeros so that they are all the same size
    zeroArr0 = np.zeros((34,34,34))
    zeroArr1 = np.zeros((34,34,34))

    if resize == 1:
        # Resize the 2D slices
        calmRatio = 34.0/np.amax(calmTmp.shape)
        stressRatio = 34.0/np.amax(stressTmp.shape)

        calm3d = scipy.ndimage.interpolation.zoom(calmTmp, (calmRatio))
        stress3d = scipy.ndimage.interpolation.zoom(stressTmp, (stressRatio))

        if calm3d.shape[0] != 34:
            startInd = (34 - calm3d.shape[0])/2
            zeroArr0[startInd:calm3d.shape[0]+startInd,:calm3d.shape[1],:calm3d.shape[2]] = calm3d
        if calm3d.shape[1] != 34:
            startInd = (34 - calm3d.shape[1])/2
            zeroArr0[:calm3d.shape[0],startInd:calm3d.shape[1]+startInd,:calm3d.shape[2]] = calm3d
        if calm3d.shape[2] != 34:
            startInd = (34 - calm3d.shape[2])/2
            zeroArr0[:calm3d.shape[0],:calm3d.shape[1],startInd:calm3d.shape[2]+startInd] = calm3d


        if stress3d.shape[0] != 34:
            startInd = (34 - stress3d.shape[0])/2
            zeroArr1[startInd:stress3d.shape[0]+startInd,:stress3d.shape[1],:stress3d.shape[2]] = stress3d
        if stress3d.shape[1] != 34:
            startInd = (34 - stress3d.shape[1])/2
            zeroArr1[:stress3d.shape[0],startInd:stress3d.shape[1]+startInd,:stress3d.shape[2]] = stress3d
        if stress3d.shape[2] != 34:
            startInd = (34 - stress3d.shape[2])/2
            zeroArr1[:stress3d.shape[0],:stress3d.shape[1],startInd:stress3d.shape[2]+startInd] = stress3d

    else:
        zeroArr0[:calm3d.shape[0],:calm3d.shape[1],:calm3d.shape[2]] = calm3d
        zeroArr1[:stress3d.shape[0],:stress3d.shape[1],stress3d.shape[2]] = stress3d

    for i in np.arange(zeroArr0.shape[0]):
        zeroArr0[i] = sklearn.preprocessing.normalize(zeroArr0[i])
        zeroArr1[i] = sklearn.preprocessing.normalize(zeroArr1[i])

    #catOut = [zeroArr0, zeroArr1]
    catOut = np.dot(zeroArr0, zeroArr1) # Encode images together
    return catOut

def importType(pptType, n):
    """
    Get stress and calm scans for n patients with pptType illness.
    Return joined array.
    """
    tmplst = []
    simsDir = "/data/jim/Heart/sims/"
    for i in np.arange(0,n):
        cwdStress = str(simsDir+"stress_"+pptType+"_%0.4d.fits") %i
        cwdCalm = str(simsDir+"rest_"+pptType+"_%0.4d.fits") %i
        # Get zoomed 3d arrays:
        xAx = importHeartData(cwdCalm, cwdStress, 1) # zoom = 1
        tmplst.append(xAx)

    dataFile = np.array(tmplst)
    #print(dataFile.shape)

    return dataFile

def cropHeart(inp):
    """
    Crop the heart so that all the padding is done away with.
    Output cropped heart.
    """
    # argwhere will give you the coordinates of every point above smallest
    true_points = np.argwhere(inp)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = inp[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1]+1,   # inclusive
          top_left[2]:bottom_right[2]+1]
    return out

if __name__ == "__main__":

    # Do data import
    abName = "ischaemia"
    abDat = importType(abName,500)
    abDat = np.moveaxis(abDat,1,-1)

    normName = "healthy"
    normDat = importType(normName,500) # Normal and abnormal data same number of ppts
    normDat = np.moveaxis(normDat,1,-1)

    inData = np.concatenate([normDat, abDat])

    # Do labelling
    normLab = np.zeros(normDat.shape[0])
    abLab = np.ones(abDat.shape[0])
    labels = np.concatenate([normLab, abLab])

    # Mutual shuffle
    shufData, shufLab = sklearn.utils.shuffle(inData, labels, random_state=1)
    shufData = np.reshape(shufData,(-1,34,34,34,1))
    shufLabOH = np.eye(2)[shufLab.astype(int)] # One hot encode

    # k fold the data
    k = 3
    kfoldData = np.array_split(shufData, k)
    kfoldLabels = np.array_split(shufLab, k)
    kfoldLabelsOH = np.array_split(shufLabOH, k)

    # Neural net (two-channel)

    spec = []
    sens = []
    roc = []

    for i in np.arange(0,k,1):
        sess = tf.InteractiveSession()
        tf.reset_default_graph()
        tflearn.initializations.normal()

        # Input layer:
        net = tflearn.layers.core.input_data(shape=[None,34,34,34,1])

        # First layer:
        net = tflearn.layers.conv.conv_3d(net, 32, [10,10,10],  activation="leaky_relu")
        net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

        # Second layer:
        net = tflearn.layers.conv.conv_3d(net, 64, [5,5,5],  activation="leaky_relu")
        net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

        # Fully connected layers
        net = tflearn.layers.core.fully_connected(net, 2048, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
        net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
        net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        net = tflearn.layers.core.fully_connected(net, 512, regularizer="L2", weight_decay=0.01, activation="leaky_relu")
        net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        # Output layer:
        net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

        net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        # Train the model, leaving out the kfold not being used
        dummyData = np.reshape(np.concatenate(kfoldData[:i] + kfoldData[i+1:], axis=0), [-1,34,34,34,1])
        dummyLabels = np.reshape(np.concatenate(kfoldLabelsOH[:i] + kfoldLabelsOH[i+1:], axis=0), [-1, 2])
        model.fit(dummyData, dummyLabels, batch_size=64, n_epoch=100, show_metric=True)

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
        sens.append(model.evaluate(np.array(healthTest), healthLabel))
        spec.append(model.evaluate(np.array(illTest), illLabel))

        # Get roc curve data
        predicted = np.array(model.predict(np.array(kfoldData[i])))
        fpr, tpr, th = roc_curve(kfoldLabels[i], predicted[:,1])
        auc = roc_auc_score(kfoldLabels[i], predicted[:,1])
        roc.append([fpr, tpr, auc])


    # Postprocessing (specificity, sensitivity, roc curves)
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))

    plt.figure(figsize=(5, 5))

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for i in np.arange(k):
        fpr = roc[i][0]
        tpr = roc[i][1]
        plt.plot(fpr, tpr, alpha=0.15, color="darkblue")
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, color="darkblue", label="Average ROC curve")
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='lightblue', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--', label="Random guess")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve '+dt)
    plt.legend(loc=4)
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig("./figures/rocCurves/"+dt+"-3dCNNfakedata.png")

    aucs = []
    for i in np.arange(k):
        aucs.append(roc[i][2])

    log = open("./logs/"+dt+"-3dCNNfakedata.log","w+")
    strOut = str("Specificity: "+str(spec)+"\nAvg: "+str(np.mean(spec))+"\nSensitivity: "+str(sens)+"\nAvg: "+str(np.mean(sens))+"\nROC AUC: "+str(aucs)+"\nAvg: "+str(np.mean(aucs)))
    log.write(strOut)
    log.close()
