# TODO implement references as k-folds

import numpy as np
from scipy import ndimage
import scipy.misc as sp
import h5py

def flipz(h_arr):
    """ Flip heart array along z axis """
    return h_arr[:,:,:,::-1,:]

def flipy(h_arr):
    """ Flip heart array along y axis """
    return h_arr[:,:,::-1,:,:]

def gblur(h_arr, sig=0.5):
    """ Gaussian blur scan """
    xaxis = h_arr.shape[1]* 2**3
    twod_heart = h_arr.reshape(-1, xaxis)
    blurred_heart = ndimage.gaussian_filter(twod_heart, sig)
    return np.reshape(blurred_heart, h_arr.shape)

def rotate(h_arr):
    """ Rotate around circular axis """
    r90 = np.rot90(h_arr, 1, axes=(2,3))
    r180 = np.rot90(h_arr, 2, axes=(2,3))
    r270 = np.rot90(h_arr, 3, axes=(2,3))
    return np.concatenate([r90,r180,r270])

def translate(h_arr):
    """ Translate along y axis """
    trans_arr = np.roll(h_arr, 2, axis=2)
    for i in np.arange(4,np.shape(h_arr)[1],2):
        trans_arr = np.append(trans_arr, np.roll(h_arr, i, axis=2), axis=0) 
    return trans_arr

def augment(h_arr):
    """ AUGMENT BOI """
    print("Start shape: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, flipz(h_arr))) # 2*n = 2n
    print("After flip z: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, flipy(h_arr))) # 2*2n = 4n
    print("After flip y: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, flipz(flipy(h_arr)))) # 2*4n = 8n
    print("After flip z y: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, gblur(h_arr))) # 2*8n = 16n
    print("After g blur: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, rotate(h_arr))) # 4*16n = 64n
    print("After rotation: "+str(h_arr.shape))
    h_arr = np.concatenate((h_arr, translate(h_arr))) # 16*64n = 1024n
    print("After translation: "+str(h_arr.shape))
    return h_arr

if __name__ == "__main__":
    hf_real_data = h5py.File("./data/real_data.h5")
    examples = hf_real_data["in_data"]
    labelsOH = hf_real_data["in_labels"]

    aug_arr = np.expand_dims(augment(np.expand_dims(examples[0],0)), 0)
    print("Total array shape: "+str(aug_arr.shape)+"\n")
    for heart in examples[1:]:
        heart = np.expand_dims(heart, 0)
        aug_arr = np.append(aug_arr, np.expand_dims(augment(heart), axis=0), axis=0)
        print("Total array shape: "+str(aug_arr.shape)+"\n")

    with h5py.File("./data/aug_data.h5") as hf:
        hf.create_dataset("in_data", data=aug_arr)
        hf.create_dataset("in_labels", data=labelsOH)
