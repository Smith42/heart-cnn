from __future__ import print_function
import matplotlib     # These are needed to run
matplotlib.use("Agg") # the code headless.

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import scipy
import sys
import h5py
from CNN import getCNN
import datetime, time
import tensorflow as tf
import tflearn

def getLoss(inData, inLabel, maskWidth, j, k, l):
    """
    Mask an area of a set of ppt arrays with zeros and get the AUC score for the modified arrays.
    Return AUC score.
    """
    mArr = np.zeros(inData.shape)
    ones = np.ones((inData.shape[0], maskWidth, maskWidth, maskWidth, 2))
    mArr[:,j:j+maskWidth,k:k+maskWidth,l:l+maskWidth] = ones # Set mask array for this index
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
    # So far this has only been implemented for healthy/ill pairs. It should be easy enough to generalise it to n classes though.
    # Generalisation to n classes would require grouping of healthy and ill diagnoses as in cnnAll.py, or a rethink of the current loss function.
    ppt = 20
    h5f = h5py.File("./data/twoThousand.h5", "r")
    inData = h5f["inData"][ppt]
    inData = inData[np.newaxis,...]

    model = getCNN(2)
    model.load("./data/placeholderModel") # The model that we want to test goes here

    inLabel = model.predict(inData)[:,1]

    maskWidth = 8 # Might be more representative to have this as even.
    lossCube = np.zeros(inData.shape[1:4])

    for j in np.arange(inData.shape[1] - maskWidth + 1):
        for k in np.arange(inData.shape[2] - maskWidth + 1):
            for l in np.arange(inData.shape[3] - maskWidth + 1):
                loss = getLoss(inData, inLabel, maskWidth, j, k, l)
                lossCube[j+maskWidth/2,k+maskWidth/2,l+maskWidth/2] = loss
                print(j+maskWidth/2,k+maskWidth/2,l+maskWidth/2,":",loss)

    lossCube = normalise(lossCube)

    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))

    np.save("./logs/lossCubes/"+dt+"_ppt"+str(ppt)+"_"+str(maskWidth)+"_heartCube", inData[0])
    np.save("./logs/lossCubes/"+dt+"_ppt"+str(ppt)+"_"+str(maskWidth)+"_lossCube-"+str(i)+"-of-"+str(kfold-1), lossCube)
