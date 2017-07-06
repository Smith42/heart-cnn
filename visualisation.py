#!/usr/bin/python

import scipy.ndimage
from mayavi import mlab
import numpy as np
import dicom

def cropHeart(inp):
    """
    Crop the heart so that all the padding is done away with.
    Output cropped heart.
    """
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(inp)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = inp[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1]+1,   # inclusive
          top_left[2]:bottom_right[2]+1]
    print(out.shape, "cropped from", inp.shape)
    return out

# Import data
ds0 = dicom.read_file("./data/flst/SPECT101/STRESSRECONFBPNOAC/1.3.6.1.4.1.5962.99.1.3005141565.2121075984.1428934316605.2.0.dcm")
px0 = ds0.pixel_array

cropped = cropHeart(px0)
ratio = 34.0/np.amax(cropped.shape)

reshaped = scipy.ndimage.interpolation.zoom(cropped, (ratio))

# Pad the 2d slices with zeros so that they are all the same size
zeroArr = np.zeros((34,34,34))

if reshaped.shape[0] != 34:
   startInd = (34 - reshaped.shape[0])/2
   zeroArr[startInd:reshaped.shape[0]+startInd,:reshaped.shape[1],:reshaped.shape[2]] = reshaped
if reshaped.shape[1] != 34:
   startInd = (34 - reshaped.shape[1])/2
   zeroArr[:reshaped.shape[0],startInd:reshaped.shape[1]+startInd,:reshaped.shape[2]] = reshaped
if reshaped.shape[2] != 34:
   startInd = (34 - reshaped.shape[2])/2
   zeroArr[:reshaped.shape[0],:reshaped.shape[1],startInd:reshaped.shape[2]+startInd] = reshaped


# Visualise data
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(zeroArr),
                            plane_orientation='x_axes',
                            slice_index=zeroArr.shape[0]/2,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(zeroArr),
                            plane_orientation='y_axes',
                            slice_index=zeroArr.shape[1]/2,
                        )

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(zeroArr),
                            plane_orientation='z_axes',
                            slice_index=zeroArr.shape[2]/2,
                        )

mlab.axes()
mlab.show()
