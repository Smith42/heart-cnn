from __future__ import print_function

import numpy as np
import argparse
import tensorflow as tf
from keras.models import Model
import keras
import os
from keras import backend as K
import horovod.keras as hvd
import sklearn
from sklearn.utils import shuffle as mutual_shuf
from sklearn.metrics import roc_curve, roc_auc_score
from CNN import getCNN
import h5py
import time

def gen_folds(num_ars, i, k):
    """ Generate fold pointer arrays given number of total data cubes """
    k_arr = np.arange(k)

    h_ind = np.arange(num_ars/2)
    np.random.shuffle(h_ind)
    i_ind = np.arange(num_ars/2,num_ars)
    np.random.shuffle(i_ind)

    k_folds_h = np.array_split(h_ind, k)
    k_folds_i = np.array_split(i_ind, k)
    k_folds = [np.concatenate((k_folds_h[j], k_folds_i[j])) for j in k_arr]

    current_fold = np.sort(k_folds[i])
    ro_folds = np.sort(np.concatenate(k_folds[:i]+k_folds[i+1:]))
    # We now have shuffled k folds ready for input
    return list(current_fold), list(ro_folds)

# Import and preprocess data

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Run k-folded CNN on augmented data.")
    parser.add_argument(type=int, dest="i", help="Current testing k-fold.")
    parser.add_argument("-k", "--n-k-folds", nargs="?", type=int, const=5, default=5, dest="k", help="Total number of folds (default 5).")
    parser.add_argument("-s", "--seed", nargs="?", type=int, const=1729, default=1729, dest="SEED", help="Numpy random seed (default 1729).")
    args = parser.parse_args()

    # Initialize Horovod
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    print("Hvd current rank:", str(hvd.local_rank()))
    print("Seed:", str(args.SEED))
    print("Current kfold:", str(args.i), "of", str(args.k-1))
    np.random.seed(args.SEED)

    h5f = h5py.File("./data/temp_data.h5", "r")
    inData = h5f["augs/data"]
    inLabelsOH = h5f["augs/labels"]
    print("Augmented data in:", str(inData.shape), str(inLabelsOH.shape))

    inData_test = h5f["reals/data"][:]
    inLabelsOH_test = h5f["reals/labels"][:]
    inLabels_test = inLabelsOH_test[:,1]
    print("Real (test) data in:", str(inData_test.shape), str(inLabelsOH_test.shape))

    illTest = inData_test[inLabels_test == 1]
    healthTest = inData_test[inLabels_test == 0]

    # Neural net (two-channel)
    model = getCNN(2) # 2 classes: healthy, ischaemia
    opt = keras.optimizers.Adam(lr=0.001*hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    cb = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
        # Reduce the learning rate if training plateaues.
        #keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
    ]
    dt =str(int(time.time()))

    # set up logdir
    filestr = str(dt+"-"+str(args.i))
    logdir = "./logs/s"+str(args.SEED)+"/"
    if hvd.rank() == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        cb.append(keras.callbacks.ModelCheckpoint(filepath=logdir+filestr+".h5", verbose=1, save_best_only=False, period=1))
        cb.append(keras.callbacks.CSVLogger(logdir+filestr+".csv"))


    # Train the model, leaving out the kfold not being used
    n_epochs = int(np.ceil(30 / hvd.size()))
    model.fit(x=inData, y=inLabelsOH, batch_size=100, verbose=2, callbacks=cb, epochs=n_epochs, shuffle='batch')

    # Get sensitivity and specificity
    if hvd.rank() == 0:
        healthLabel = np.tile([1,0], (len(healthTest), 1))
        illLabel = np.tile([0,1], (len(illTest), 1))
        sens = model.evaluate(x=np.array(healthTest), y=healthLabel, verbose=0, batch_size=1)[1] # Get accuracy
        spec = model.evaluate(x=np.array(illTest), y=illLabel, verbose=0, batch_size=1)[1] # Get accuracy
        inData_test = np.concatenate((healthTest, illTest))
        inLabels_test = np.concatenate((healthLabel, illLabel))[:,1]

    # Get roc curve data
        predicted = model.predict(inData_test, verbose=0, batch_size=1)

        fpr, tpr, th = roc_curve(inLabels_test, predicted[:,1])
        auc = roc_auc_score(inLabels_test, predicted[:,1])

        print(spec, sens, auc)

        savefileacc = logdir+filestr+"-acc.log"
        savefileroc = logdir+filestr+"-roc.log"
        np.savetxt(savefileacc, (spec,sens,auc), delimiter=",")
        np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
