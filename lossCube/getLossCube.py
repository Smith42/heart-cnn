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
import argparse
import time
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

def getLossCube(inData, inLabel, maskWidth):
    """
    Apply getLoss for all indices in the heartCube.
    """
    lossCube = np.zeros(inData.shape[1:4])

    for j in np.arange(inData.shape[1] - maskWidth + 1):
        for k in np.arange(inData.shape[2] - maskWidth + 1):
            for l in np.arange(inData.shape[3] - maskWidth + 1):
                loss = getLoss(inData, inLabel, maskWidth, j, k, l)
                lossCube[j+maskWidth//2,k+maskWidth//2,l+maskWidth//2] = loss
                print(j+maskWidth//2,k+maskWidth//2,l+maskWidth//2,":",loss)

    return normalise(lossCube)

def relu(x):
    return np.maximum(x, 0)

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

    # Argument parsing
    parser = argparse.ArgumentParser("Generate losscube for SPECT scan files.")
    parser.add_argument("-o", "--occlusion_map", help="Use occlusion mapping for generation of the loss cube.", dest="occlusion_map", action="store_true")
    parser.add_argument("-c", "--cam", help="Use CAM for generation of the loss cube.", dest="cam", action="store_true")
    parser.add_argument("-w", "--mask_width", help="[n,n,n] mask to be used in the generation of the occlusion losscube.", type=int, default=1)
    parser.add_argument("-p", "--participant", help="Generate the losscube for this participant. If not given, a random ppt will be chosen.", type=int)
    parser.add_argument("-f", "--file_path", help="File path for input .h5 file.", type=argparse.FileType('r'), required=True)
    parser.add_argument("-m", "--model_path", help="Model path (.tflearn).", required=True)
    parser.set_defaults(occlusion_map=False, grad_cam=False)

    args = parser.parse_args()
    dt = str(int(time.time()))

    h5f = h5py.File(args.file_path.name, "r")
    if args.participant is None:
        # Randomly choose ill participant
        inLabels = h5f["inLabels"]
        ppt = np.random.randint(inLabels.shape[0])
        while inLabels[ppt][1] != 1.0:
            ppt = np.random.randint(inLabels.shape[0])
    else:
        # Use user's participant
        ppt = args.participant

    inData = h5f["inData"][ppt]
    inData = inData[np.newaxis,...]

    model, observer = getCNN(2, observe=True)
    model.load(args.model_path)

    predLabel = model.predict(inData)[:,1]

    if abs(inLabels[ppt][1] - predLabel > 0.4):
        # Throw a warning if the CNN misdiagnoses.
        input("ERROR: CNN has misdiagnosed ppt ("+str(inLabels[ppt][1])+" vs "+str(predLabel)+").\n\
    Press enter to continue (^c to exit).")

    if args.occlusion_map:
        maskWidth = args.mask_width # Might be more representative to have this as even. 2018-05-14 -- lower is better!!
        lossCube = getLossCube(inData, predLabel, maskWidth)

        np.save("./logs/lossCubes/"+dt+"_ppt-"+str(ppt)+"_"+str(maskWidth)+"_lossCube_occlusion_map", lossCube)

    if args.cam:
        weights = model.get_weights(tflearn.variables.get_layer_variables_by_name('FullyConnected')[0])
        intLabel = int(np.rint(predLabel))
        weights = relu(weights[:,intLabel])

        observed = relu(observer.predict(inData)[0])
        lossCube = np.pad(np.tensordot(weights, observed, axes=[0,-1]), [1,0], "constant")[:-1,:-1,:-1] # This padding is needed due to the precision loss in convolution
        lossCube = scipy.ndimage.interpolation.zoom(lossCube, 4) # The int here is dependent on the pooling in the CNN
        np.save("./logs/lossCubes/"+dt+"_ppt-"+str(ppt)+"_"+str(maskWidth)+"_lossCube_cam_map", lossCube)

    np.save("./logs/lossCubes/"+dt+"_ppt-"+str(ppt)+"_"+str(maskWidth)+"_heartCube-rest", inData[0][...,0])
    np.save("./logs/lossCubes/"+dt+"_ppt-"+str(ppt)+"_"+str(maskWidth)+"_heartCube-stress", inData[0][...,1])
