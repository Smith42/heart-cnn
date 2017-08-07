import numpy as np
from mayavi import mlab

# Show interactive image of the loss of the CNN when an area of the heart cube is obscured.

if __name__ == "__main__":
    heartCube = np.load("../logs/lossCubes/2017-08-04_10:49:00_ppt20_4_heartCube.npy")[...,1] # 1 is stressed, 0 is calm
    lossCube0 = np.load("../logs/lossCubes/2017-08-04_10:49:00_ppt20_4_lossCube-0-of-4.npy")
    lossCube1 = np.load("../logs/lossCubes/2017-08-04_10:59:00_ppt20_4_lossCube-1-of-4.npy")
    lossCube2 = np.load("../logs/lossCubes/2017-08-04_11:09:00_ppt20_4_lossCube-2-of-4.npy")
    lossCube3 = np.load("../logs/lossCubes/2017-08-04_11:19:00_ppt20_4_lossCube-3-of-4.npy")
    lossCube4 = np.load("../logs/lossCubes/2017-08-04_11:29:00_ppt20_4_lossCube-4-of-4.npy")

    lossCube = lossCube0+lossCube1+lossCube2+lossCube3+lossCube4

    mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(lossCube), opacity=0.4)

    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(heartCube),
                            plane_orientation='x_axes',
                            slice_index=heartCube.shape[0]/2,
                        )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(heartCube),
                            plane_orientation='y_axes',
                            slice_index=heartCube.shape[1]/2,
                        )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(heartCube),
                            plane_orientation='z_axes',
                            slice_index=heartCube.shape[2]/2,
                        )
    mlab.axes()
    mlab.show()
