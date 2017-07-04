#!/usr/bin/python

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
ds0 = dicom.read_file("./data/nlst/7/RESTRECONFBPNONAC/1.2.826.0.1.3680043.8.373.1.149242122.1391620200.dcm")
px0 = ds0.pixel_array

cropped = cropHeart(px0)

# Visualise data
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(cropped),
                            plane_orientation='x_axes',
                            slice_index=cropped.shape[0]/2,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(cropped),
                            plane_orientation='y_axes',
                            slice_index=cropped.shape[1]/2,
                        )

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(cropped),
                            plane_orientation='z_axes',
                            slice_index=cropped.shape[2]/2,
                        )
mlab.axes()
mlab.show()
