from __future__ import print_function

import numpy as np
import argparse
import tensorflow as tf
from keras.models import Model
import keras
import os
from keras import backend as K
import sklearn
from sklearn.utils import shuffle as mutual_shuf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from CNN import getCNN
import h5py
import time

class Aug_Generator(keras.utils.Sequence):
    def __init__(self, cubes, labels, indices, batch_size=100, shuffle=False):
        """ Initialization """
        self.batch_size = batch_size
        self.labels = labels
        self.cubes = cubes
        self.indices = indices
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        current_indices = list(np.sort(self.indices[index*self.batch_size:(index+1)*self.batch_size]))

        # Get np array from HDF5 file
        X = self.cubes[current_indices]
        y = self.labels[current_indices]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

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
    parser.add_argument("-b", "--batch_size", nargs="?", type=int, const=248, default=248, dest="batch_size", help="Batch size (small batch sizes throttle speed due to slow h5py data loading).")

    # Initialise data
    args = parser.parse_args()
    np.random.seed(args.SEED)
    h5f_aug = h5py.File("./data/aug_data.h5", "r")
    h5f_real = h5py.File("./data/real_data.h5", "r")

    # Get folding
    num_ars = h5f_real["in_labels"].shape[0]
    current_fold, ro_folds = gen_folds(num_ars, args.i, args.k)

    # Aug data
    inData = h5f_aug["in_data"]
    inLabelsOH = h5f_aug["in_labels"]
    indices = h5f_aug["indices"][:]
    print("Augmented data in:", str(inData.shape), str(inLabelsOH.shape))

    # Get indexes for the augmented array's (k-1) folds
    ro_folds_i = np.squeeze(np.concatenate([np.where(indices == index) for index in ro_folds], axis=-1))
    np.random.shuffle(ro_folds_i)

    # Test data
    inData_test = h5f_real["in_data"][current_fold]
    inLabelsOH_test = h5f_real["in_labels"][current_fold]
    inLabels_test = inLabelsOH_test[:,1]
    print("Real (test) data in:", str(inData_test.shape), str(inLabelsOH_test.shape))

    # Segment out ill and healthy for test data
    illTest = inData_test[inLabels_test == 1]
    healthTest = inData_test[inLabels_test == 0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    print("Seed:", str(args.SEED))
    print("Current kfold:", str(args.i), "of", str(args.k-1))

    # Neural net (two-channel)
    model = getCNN(2) # 2 classes: healthy, ischaemia
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    cb = [
        # Reduce the learning rate if training plateaues.
        keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
    ]
    dt =str(int(time.time()))

    # set up logdir
    filestr = str("k-equals-"+str(args.i))
    logdir = "./logs/s"+str(args.SEED)+"/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    cb.append(keras.callbacks.ModelCheckpoint(filepath=logdir+filestr+".h5", verbose=1, save_best_only=False, period=6))
    cb.append(keras.callbacks.CSVLogger(logdir+filestr+".csv"))

    # Train the model, leaving out the kfold not being used
    train_ind, test_ind = train_test_split(ro_folds_i, test_size=0.1, shuffle=False)
    batch_size = args.batch_size
    epochs = 6
    n_test_batches = len(test_ind) // batch_size
    n_train_batches = len(train_ind) // batch_size

    model.fit_generator(Aug_Generator(inData, inLabelsOH, train_ind, batch_size=batch_size), steps_per_epoch=n_train_batches, validation_data=Aug_Generator(inData, inLabelsOH, test_ind, batch_size=batch_size), validation_steps=n_test_batches, verbose=1, callbacks=cb, epochs=epochs)

    # Get sensitivity and specificity
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
