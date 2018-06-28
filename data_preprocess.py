from __future__ import print_function

import numpy as np
import os
import glob
import h5py
import pydicom
from numpy import interp
import scipy, scipy.ndimage

# Import and preprocess data
# If the heart image stored dir has changed:
#   * Change the directory and regex of the heart images to import in importType.
#   * Change the number of images to import.

def import_heart_data(calm_file, stress_file, resize):
    """
    Import heart data and extract the pixel array.
    Slice halfway along ind axis.
    Concatenate and return stress file and calm file.
    If resize == 1, interpolate data to fit (32,32,32) arr.
    """
    calm_tmp = pydicom.dcmread(calm_file).pixel_array
    stress_tmp = pydicom.dcmread(stress_file).pixel_array

    calm_tmp = crop_heart(calm_tmp)
    stress_tmp = crop_heart(stress_tmp)

    # Pad the 3d arrays with zeros so that they are all the same size
    zero_arr0 = np.zeros((32,32,32))
    zero_arr1 = np.zeros((32,32,32))

    if resize == 1:
        # Resize the 3D slices
        calm_ratio = 32.0/np.amax(calm_tmp.shape)
        stress_ratio = 32.0/np.amax(stress_tmp.shape)

        calm3d = scipy.ndimage.interpolation.zoom(calm_tmp, (calm_ratio))
        stress3d = scipy.ndimage.interpolation.zoom(stress_tmp, (stress_ratio))

        zero_arr0[:calm3d.shape[0],:calm3d.shape[1],:calm3d.shape[2]] = calm3d
        zero_arr1[:stress3d.shape[0],:stress3d.shape[1],:stress3d.shape[2]] = stress3d

    else:
        zero_arr0[:calm3d.shape[0],:calm3d.shape[1],:calm3d.shape[2]] = calm3d
        zero_arr1[:stress3d.shape[0],:stress3d.shape[1],:stress3d.shape[2]] = stress3d

    zero_arr0 = np.expand_dims(normalise(zero_arr0), -1)
    zero_arr1 = np.expand_dims(normalise(zero_arr1), -1)

    cat_out = np.concatenate((zero_arr0, zero_arr1), axis=-1)
    return cat_out

def import_type(ppt_type, n):
    """
    Get stress and calm scans for n patients with pptType illness.
    Return joined array.
    """
    tmp_lst = []
    data_dir = "/home/mike/Documents/hertsDegree/UoH_job/heart_cnn/data/PACIFIC_Leicester/"+ppt_type+"/"

    for path in glob.glob(data_dir+"*/")[:n]:
        cwd_calm = glob.glob(path+"REST*/*.dcm")[0]
        cwd_stress = glob.glob(path+"STRESS*/*.dcm")[0]
        xAx = import_heart_data(cwd_calm, cwd_stress, resize=1)
        tmp_lst.append(xAx)

    data_file = np.array(tmp_lst)
    print(ppt_type, data_file.shape)

    return data_file

def crop_heart(inp):
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
    out = inp[top_left[0]:bottom_right[0]+1,   # plus 1 because slice isn't
              top_left[1]:bottom_right[1]+1,   # inclusive
              top_left[2]:bottom_right[2]+1]
    return out

def normalise(inData):
    """
    Normalise 3D array.
    """
    inDataAbs = np.fabs(inData)
    inDataMax = np.amax(inData)
    normalisedData = inDataAbs/inDataMax
    return normalisedData

if __name__ == "__main__":

    # Do data import
    norm_name = "nlst"
    norm_dat = import_type(norm_name,29) # Normal and abnormal data same number of ppts

    is_name = "rlst"
    is_dat = import_type(is_name,29)

    data = np.concatenate([norm_dat, is_dat])

    # Do labelling
    norm_lab = np.full(norm_dat.shape[0], 0)
    is_lab = np.full(is_dat.shape[0], 1)
    labels = np.eye(2)[np.concatenate([norm_lab, is_lab])]

    print(data.shape)

    # Save data as HDF5 object:
    h5f = h5py.File("./data/real_data.h5", "w")
    h5f.create_dataset("in_data", data=data)
    h5f.create_dataset("in_labels", data=labels)
    h5f.close()
