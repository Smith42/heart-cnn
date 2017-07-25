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
import datetime

if __name__ == "__main__":
    # Postprocessing (specificity, sensitivity, roc curves)
    dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))

    spec, sens, roc = np.load("./3D-2ch-fakedata/mess.npy")

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
