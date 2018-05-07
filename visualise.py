import numpy as np
import matplotlib.pyplot as plt
import argparse
import imageio
import astropy.io.fits as pyfits, dicom
import scipy, scipy.ndimage
from scipy.interpolate import griddata
import time
import os

## DATA PREPROCESSING FUNCTIONS

def crop_heart(in_data):
    """
    Crop the heart, and interpolate so that all the padding is done away with, and
    the heart cube shape is exactly [34,34,34].
    Output cropped heart.
    """
    # argwhere will give you the coordinates of every point above smallest
    true_points = np.argwhere(in_data)
    # take the smallest points and use them as the top left of the crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of the crop
    bottom_right = true_points.max(axis=0)
    cropped_arr = in_data[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                          top_left[1]:bottom_right[1]+1,  # inclusive
                          top_left[2]:bottom_right[2]+1]

    # Array to be filled
    zero_arr = np.zeros((34,34,34))
    new_ratio = 34.0/np.amax(cropped_arr.shape)
    # Interpolated array
    interp_arr = scipy.ndimage.interpolation.zoom(cropped_arr, (new_ratio))
    # Fill zero_arr with interpolated array
    zero_arr[:interp_arr.shape[0],:interp_arr.shape[1],:interp_arr.shape[2]] = interp_arr

    return normalise(zero_arr)

def normalise(inData):
    """
    Normalise n-D array.
    """
    inDataAbs = np.fabs(inData)
    inDataMax = np.amax(inData)
    normalisedData = inDataAbs/inDataMax
    return normalisedData

## CARTESIAN VISUALISATION FUNCTIONS

def brute_force_plot(heart_array, save_to, log):
    """
    Saves reshaped SPECT scan, with rows showing the scan axes.
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[60,10])

    fig.suptitle("Slices of SPECT scan.", size=20)

    fig.tight_layout()

    ax[0].imshow(heart_array.reshape(heart_array.shape[0], -1))
    ax[0].axis("off")
    ar_z = np.swapaxes(heart_array, 1, 2)
    ax[1].imshow(ar_z.reshape(ar_z.shape[0] ,-1))
    ax[1].axis("off")
    ax[2].imshow(heart_array.reshape(-1, heart_array.shape[2]).T)
    ax[2].axis("off")

    fig.subplots_adjust(top=0.96)

    if log:
        dt = str(time.time())
        plt.savefig(save_to)
    else:
        plt.show()

def unfolded_artefact_plot(heart_array, artefact_site, save_to, log):
    # This will need reworking when I figure out the artefact overlay.
    """
    Plot heart_array at artefact_site.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=[10,10])

    fig.suptitle("Unfolded SPECT scan at simulated artefact site, "+str(artefact_site), size=16)
    fig.subplots_adjust(top=0.92)

    ax[0,1].axis("off")

    ax[0,0].imshow(heart_array[artefact_site[0]])
    ax[0,0].axhline(y=artefact_site[1], linewidth=2, color='red')
    ax[0,0].axvline(x=artefact_site[2], linewidth=2, color='red')
    at = AnchoredText("x = "+str(artefact_site[0]), prop=dict(size=8), frameon=True, loc=4)
    at.patch.set_boxstyle("square")
    ax[0,0].add_artist(at)

    ax[1,0].imshow(heart_array[:,artefact_site[1]])
    ax[1,0].axhline(y=artefact_site[0], linewidth=2, color='red')
    ax[1,0].axvline(x=artefact_site[2], linewidth=2, color='red')
    at = AnchoredText("y = "+str(artefact_site[1]), prop=dict(size=8), frameon=True, loc=4)
    at.patch.set_boxstyle("square")
    ax[1,0].add_artist(at)

    ax[1,1].imshow(heart_array[:,:,artefact_site[2]])
    ax[1,1].axhline(y=artefact_site[0], linewidth=2, color='red')
    ax[1,1].axvline(x=artefact_site[1], linewidth=2, color='red')
    at = AnchoredText("z = "+str(artefact_site[2]), prop=dict(size=8), frameon=True, loc=4)
    at.patch.set_boxstyle("square")
    ax[1,1].add_artist(at)

    fig.subplots_adjust(hspace=0, wspace=0)

    if log:
        plt.savefig(save_to)
    else:
        plt.show()

## POLAR VISUALISATION FUNCTIONS

def circular_mask(a,b, r, n=34):
    """
    Generate a circular mask, with a,b being the midpoint (cartesian) coordinates,
    r being the radius, and n being the background mask side lengths.
    """
    y,x = np.ogrid[-a:n-a, -b:n-b]
    if r != 0:
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n), dtype=bool)
        array[mask] = True
    else:
        array = np.zeros((n, n), dtype=bool)

    return array

def cart2pol(x, y):
    """
    Convert cartesian to polar.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def gen_doughnut_mask(inData, thresh=0.55):
    """
    Generate a 'doughnut' mask for our polar plot.
    Also return detected centrepoints.
    """
    doughnut_mask = []
    centrepoints = []

    # Positional array for where the (normalised) input data is larger than our threshold.
    pos_arr = np.array(np.where(inData>thresh))
    for index in np.arange(inData.shape[0]):
        if index in np.unique(pos_arr[0]) and np.count_nonzero(pos_arr[0] == index) !=1:
            xt = pos_arr[0][pos_arr[0]==index]
            yt = pos_arr[1][pos_arr[0]==index]
            zt = pos_arr[2][pos_arr[0]==index]

            centrepoint = (np.min(yt)+(np.max(yt)-np.min(yt))//2, \
                           np.min(zt)+(np.max(zt)-np.min(zt))//2)

            # Relative to centrepoint
            rel_yt = yt - centrepoint[0]
            rel_zt = zt - centrepoint[1]

            rho, phi = cart2pol(rel_yt, rel_zt)

            inner_rad = int(np.floor(min(rho)))
            outer_rad = int(np.ceil(max(rho)))

            doughnut_mask.append(np.logical_xor(circular_mask(centrepoint[0], centrepoint[1], outer_rad), \
                                                circular_mask(centrepoint[0], centrepoint[1], inner_rad)))
            centrepoints.append(centrepoint)

        else:
            # If there is no mask for that slice, there can't be a centrepoint!
            doughnut_mask.append(np.zeros(inData.shape[1:], dtype=bool))
            centrepoints.append((float('NaN'), float('NaN')))

    return np.array(doughnut_mask), centrepoints

def polar_plot(inData, save_to, log):
    """
    Generate polar plot for inData.
    """
    doughnut_mask, centrepoints = gen_doughnut_mask(inData, 0.55)
    pos_arr = np.array(np.where(doughnut_mask==True))
    prev_max_rho = 0
    rhos = []
    phis = []

    # Can this be done without a loop??
    for index, slice_n in enumerate(np.unique(pos_arr[0])):
        centrepoint = centrepoints[slice_n]
        # Positional array for slice_n
        po = np.array(np.where(doughnut_mask[slice_n]==True))
        # Relative to centrepoint
        rel_y= po[0] - centrepoint[0]
        rel_z = po[1] - centrepoint[1]

        rho, phi = cart2pol(rel_y, rel_z)
        rhos.extend(rho + prev_max_rho)
        prev_max_rho = np.max(rho) + prev_max_rho
        phis.extend(phi)

    # Generate data for plot
    points = np.swapaxes(np.array([rhos,phis]),0,1)
    values = inData[doughnut_mask]

    theta = np.linspace(-np.pi, np.pi, 200)
    r = np.linspace(0, np.max(rhos), 300)
    grid_r, grid_theta = np.meshgrid(r, theta)
    data = griddata(points, values, (grid_r, grid_theta), method='cubic', fill_value=0)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection="polar")
    ax.set_title("Polar plot")
    ax.pcolormesh(theta,r,data.T)

    if log:
        plt.savefig(save_to)
    else:
        plt.show()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Generate visualisations for SPECT scan files.")
    parser.add_argument("-b", "--brute_force", help="Generate 'brute force' (Cartesian) plot.", dest="cartesian", action="store_true")
    parser.add_argument("-u", "--unfolded", help="Generate 'unfolded artefact site' plot.", dest="unfolded", action="store_true")
    parser.add_argument("-p", "--polar", help="Generate polar plot.", dest="polar", action="store_true")
    parser.add_argument("--log", help="Want to log output?", dest="log", action="store_true")
    parser.add_argument("-f", "--file_path", help="File path for input (dicom or fits).", type=argparse.FileType('r'), required=True)
    parser.add_argument("-d", "--diagnostic_file_path", help="File path for diagnostic cube.", type=argparse.FileType('r'))
    parser.set_defaults(cartesian=False, unfolded=False, polar=False, log=False)

    args = parser.parse_args()

    f, file_extension = os.path.splitext(args.file_path.name)
    if file_extension == ".fits":
        raw_array = pyfits.open(args.file_path.name)[0].data
    elif file_extension == ".dcm":
        raw_array = dicom.read_file(args.file_path.name).pixel_array
    elif file_extension == ".npy":
        raw_array = np.load(args.file_path.name)#[...,1]
        print(raw_array.shape)
    else:
        exit("Unknown file name (is the extension '.fits', '.npy', or '.dcm'?)")

    proc_array = crop_heart(raw_array)

    if args.diagnostic_file_path is None:
        dt = str(time.time())
        if args.cartesian:
            save_to = "./figures/visualisations/"+dt+"-cartesian.png"
            brute_force_plot(proc_array, save_to, args.log)
        if args.unfolded:
            print("We need a diagnostic file to show the unfolded visualisation.")
        if args.polar:
            save_to = "./figures/visualisations/"+dt+"-polar.png"
            polar_plot(proc_array, save_to, args.log)

"""
    else:
        df, d_file_extension = os.path.splitext(args.diagnostic_file_path.name)

        if d_file_extension == ".fits":
            d_raw_array = pyfits.open(args.diagnostic_file_path.name)[0].data
        elif d_file_extension == ".dcm":
            d_raw_array = dicom.read_file(args.diagnostic_file_path.name).pixel_array
        elif d_file_extension == ".npy":
            d_raw_array = np.load(args.diagnostic_file_path.name)
        else:
            exit("Unknown diagnostic file name (is the extension '.fits', '.npy', or '.dcm'?)")

        d_proc_array = crop_heart(d_raw_array)

        if args.cartesian:
            save_to = "./figures/visualisations/"+dt+"-cartesian.png"
            brute_force_plot(proc_array, save_to, args.log)
        if args.unfolded:
            # This is not ready
            save_to = "./figures/visualisations/"+dt+"-polar.png"
            unfolded_artefact_plot(proc_array, artefact_site, save_to, args.log)
        if args.polar:
            save_to = "./figures/visualisations/"+dt+"-polar.png"
            polar_plot(proc_array, save_to, args.log)
"""
