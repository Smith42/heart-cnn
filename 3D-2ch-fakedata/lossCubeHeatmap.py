import numpy as np
from mayavi import mlab

# Show interactive image of the loss of the CNN when an area of the heart cube is obscured.

if __name__ == "__main__":
    heartCube = np.load("../logs/lossCubes/2017-08-03_10:39:00_ppt20_4_heartCube.npy")[...,1] # 1 is stressed, 0 is calm
    lossCube = np.load("../logs/lossCubes/2017-08-03_10:39:00_ppt20_4_lossCube.npy")

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
