from __future__ import print_function

import numpy as np
import argparse
import tensorflow as tf
import tflearn
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

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(args.i)

    print("Seed:", str(args.SEED))
    print("Current kfold:", str(args.i), "of", str(args.k-1))
    np.random.seed(args.SEED)

    h5_aug = h5py.File("./data/aug_data.h5", "r")
    # Fancy indexing is inefficient in H5PY... Look for a nicer way to implement this. (only load current batch into memory?)
    num_ars = h5_aug["in_labels"].shape[0]
    current_fold, ro_folds = gen_folds(num_ars, args.i, args.k)

    inData = h5_aug["in_data"][ro_folds]
    inLabelsOH = h5_aug["in_labels"][ro_folds]
    inLabelsOH = np.repeat(inLabelsOH,inData.shape[1],axis=0)
    inData = inData.reshape([-1,inData.shape[2],inData.shape[3],inData.shape[4],inData.shape[5]])
    h5_aug.close()

    inData, inLabelsOH = mutual_shuf(inData, inLabelsOH)
    print("Augmented data in:", str(inData.shape), str(inLabelsOH.shape))

    h5_real = h5py.File("./data/real_data.h5", "r")
    inData_test = h5_real["in_data"][current_fold]
    inLabelsOH_test = h5_real["in_labels"][current_fold]
    inLabels_test = inLabelsOH_test[:,1]
    h5_real.close()
    print("Real (test) data in:", str(inData_test.shape), str(inLabelsOH_test.shape))

    illTest = inData_test[inLabels_test == 1]
    healthTest = inData_test[inLabels_test == 0]

    # Neural net (two-channel)
    sess = tf.Session()
    model = getCNN(2) # 2 classes: healthy, ischaemia

    # Train the model, leaving out the kfold not being used
    model.fit(inData, inLabelsOH, batch_size=100, n_epoch=30, show_metric=True)
    dt = str(int(time.time()))
    model.save("./models/"+dt+"s"+str(args.SEED)+"-"+str(args.i)+"-augment_data.tflearn")

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
    savefileacc = "./logs/"+dt+"s"+str(args.SEED)+"-"+str(args.i)+"-augment_data-acc.log"
    savefileroc = "./logs/"+dt+"s"+str(args.SEED)+"-"+str(args.i)+"-augment_data-roc.log"
    np.savetxt(savefileacc, (spec[0],sens[0],auc), delimiter=",")
    np.savetxt(savefileroc, (fpr,tpr,th), delimiter=",")
