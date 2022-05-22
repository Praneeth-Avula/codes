
import numpy as np # load point clouds and more array/vector operations
import cv2 # load images
import plotly.graph_objects as go # visualize point clouds
import pandas as pd
root = "F:\Downloads\Record1"
def inv_rot(rot):
    """
        Calculates for a given 4x4 transformation matrix (R|t) the inverse.
    """
    inv_rot_mat = np.zeros((4,4))
    inv_rot_mat[0:3,0:3] = rot[0:3,0:3].T
    inv_rot_mat[0:3,3] = -np.dot(rot[0:3,0:3].T, rot[0:3,3])
    inv_rot_mat[3,3] = 1
    return inv_rot_mat

def transfer_points(points, rot_t):
    """
        Calculates the transformation of a point cloud for a given transformation matrix.
    """
    points = np.concatenate([points, np.ones([1,points.shape[1]])])
    points[0:3,:] = np.dot(rot_t, points[0:4,:])[0:3, :]
    return points[0:3]

def make_boundingbox(label):
    """
        Calculates the Corners of a bounding box from the parameters.
    """
    corner = np.array([
         [+ label[3]/2, + label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, - label[5]/2],
        [- label[3]/2, + label[4]/2, - label[5]/2]
    ])
    corner = transfer_points(corner.T, rt_matrix(yaw = label[6])).T
    corner = corner + label[0:3]
    return corner

def make_boundingbox_graph(label):
    """
        Calculates the Corners of a bounding box from the parameters.
    """
    corner = np.array([
         [+ label[3]/2, + label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, - label[5]/2],
        [- label[3]/2, + label[4]/2, - label[5]/2],
        [- label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, - label[5]/2],
        [+ label[3]/2, + label[4]/2, - label[5]/2],
    ])
    corner = transfer_points(corner.T, rt_matrix(yaw = label[6])).T
    corner = corner + label[0:3]
    return corner


def rt_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    """
        Calculates a 4x4 Transformation Matrix. Angels in radian!
    """
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    rot = np.dot(np.dot(np.array([[c_y, -s_y, 0],
                                  [s_y, c_y, 0],
                                  [0, 0, 1]]),
                        np.array([[c_p, 0, s_p],
                                  [0, 1, 0],
                                  [-s_p, 0, c_p]])),
                        np.array([[1, 0, 0],
                           [0, c_r, -s_r],
                           [0, s_r, c_r]]))
    matrix = np.array([[0, 0, 0, x],
                       [0, 0, 0, y],
                       [0, 0, 0, z],
                       [0, 0, 0, 1.0]])
    matrix[0:3, 0:3] = rot + matrix[0:3, 0:3]
    #np.add(matrix[0:3, 0:3], rot, out=matrix[0:3, 0:3], casting="unsafe")
    return matrix

rotationmat = np.load(root + "\calibrationMat.npy")
invrotationmat = inv_rot(rotationmat)

ind = 15
blick = np.loadtxt(root + "\Blickfeld/point_cloud/%06d.csv" % ind)
velo = np.loadtxt(root + "\Velodyne/point_cloud/%06d.csv" % ind)
blick = transfer_points(blick.T[0:3], invrotationmat).T
#velo = transfer_points(velo.T[0:3], rotationmat).T

#bb = np.loadtxt(root + "\Velodyne/point_cloud/%06d.csv" % ind)
bb = np.loadtxt(root + "/Blickfeld/bounding_box/%06d.csv" % ind)
bb = make_boundingbox(bb)

data_x = blick[:,0]
data_y = blick[:,1]
data_z = blick[:,2]
data_len = len(data_x)
x_max = (bb[:,0]).max()
x_min = (bb[:,0]).min()
y_max = (bb[:,1]).max()
y_min = (bb[:,1]).min()
z_max = (bb[:,2]).max()
z_min = (bb[:,2]).min()
points = 0
for i in range(data_len):
  if ((data_x[i]<=x_max and data_x[i]>=x_min) and (data_y[i]<=y_max and data_y[i]>=y_min) and (data_z[i]<=z_max and data_z[i]>=z_min)):
    points = points + 1

data = [go.Scatter3d(x = blick[:,0],
                     y = blick[:,1],
                     z = blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = velo[:,0],
                     y = velo[:,1],
                     z = velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                     text=np.arange(8),
                    mode='lines + markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "red",
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-12.5, 12.5], 'rangemode': 'tozero'}
    }
)
fig=go.Figure(data=data, layout = layout)
fig.show()
print(points)
print(bb)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
