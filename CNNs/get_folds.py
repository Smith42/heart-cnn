import numpy as np
import h5py
from sklearn.utils import shuffle as mutual_shuf


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

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Run k-fold h5py generation (for memory saving).")
    parser.add_argument(type=int, dest="i", help="Current testing k-fold.")
    parser.add_argument("-k", "--n-k-folds", nargs="?", type=int, const=5, default=5, dest="k", help="Total number of folds (default 5).")
    parser.add_argument("-s", "--seed", nargs="?", type=int, const=1729, default=1729, dest="SEED", help="Numpy random seed (default 1729).")
    args = parser.parse_args()

    print("Seed:", str(args.SEED))
    print("Current kfold:", str(args.i), "of", str(args.k-1))
    np.random.seed(args.SEED)

    h5_aug = h5py.File("./data/aug_data.h5", "r")
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
    h5_real.close()
    print("Real (test) data in:", str(inData_test.shape), str(inLabelsOH_test.shape))

    h5f_temp = h5py.File("./data/temp_data.h5", "w") # This gets clobbered with each run
    aug_grp = h5f_temp.create_group("augs")
    aug_grp.create_dataset("data", inData)
    aug_grp.create_dataset("labels", inLabelsOH)
    real_grp = h5f_temp.create_group("reals")
    real_grp.create_dataset("data", inData_test)
    real_grp.create_dataset("labels", inLabelsOH_test)
    h5_temp.close()
