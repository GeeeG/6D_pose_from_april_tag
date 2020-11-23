import numpy as np
import os
import transforms3d
import open3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

experiment_trail = "20201123-192923"
file_path = "poses/" + experiment_trail
rotation_mat = np.loadtxt(os.path.join(file_path, "rotation.txt"))
translation = np.loadtxt(os.path.join(file_path, "translation.txt"))

def plot_rotation():
    axag_all = []
    for rot_mat in rotation_mat:
        rot_mat = np.reshape(rot_mat, (3,3))
        try:
            axis, angle = transforms3d.axangles.mat2axangle(rot_mat)
            rotation_axag = axis * angle
            axag_all.append(rotation_axag)
        except:
            pass

    axag_all = np.asarray(axag_all)
    X = axag_all[:,0]
    Y = axag_all[:,1]
    Z = axag_all[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect(1)
    ax.axis('off')

    # Creating plot
    ax.scatter3D(X, Y, Z, color = "blue", s=10);

    # add sphere
    N=50
    stride=1
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))*np.pi
    y = np.outer(np.sin(u), np.sin(v))*np.pi
    z = np.outer(np.ones(np.size(u)), np.cos(v))*np.pi
    ax.plot_surface(x, y, z, alpha=0.05, linewidth=0.0, cstride=stride, rstride=stride, cmap=cm.summer)

    plt.grid()
    plt.show()

def plot_translation():
    trans_all = []
    for trans in translation:
        trans_all.append(trans)

    trans_all = np.asarray(trans_all)
    X = trans_all[:,0]
    Y = trans_all[:,1]
    Z = trans_all[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect(1)
    ax.axis('off')

    # Creating plot
    ax.scatter3D(X, Y, Z, color = "blue", s=10);

    plt.grid()
    plt.show()

plot_translation()
