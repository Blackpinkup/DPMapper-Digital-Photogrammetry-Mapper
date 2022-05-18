from mayavi import mlab
import math
import numpy as np
from time import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import icp
 
 
def ply_read(file_path):
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines
    
    
def load_ply(ply_pth):
    lines = []
    with open(ply_pth, 'r') as f:
        lines = f.readlines()
    points_num = int(lines[3].split(' ')[2])
    points1 = lines[10:(10 + points_num)]
    points1 = np.array(list(map(lambda x: line2np(x), points1)))
    return points1
    
    
def line2np(line):
    line = line.strip().split(' ')
    return list(map(float, line))


def rotate_ply(points, phi, omega, kappa):
    r = cal_rot(phi, omega, kappa)
    points_rot = np.zeros((points.shape[0],points.shape[1]))
    for i in range(points.shape[0]):
        # print(points[i])
        points_rot[i] = np.dot(r, points[i])
        # print(points_rot[i])
        # exit()
    return points_rot
    

def cal_rot(phi, omega, kappa):
    r = np.zeros((3, 3))
    r[0][0] = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
    r[0][1] = -math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
    r[0][2] = -math.sin(phi) * math.cos(omega)
    r[1][0] = math.cos(omega) * math.sin(kappa)
    r[1][1] = math.cos(omega) * math.cos(kappa)
    r[1][2] = -math.sin(omega)
    r[2][0] = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
    r[2][1] = -math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
    r[2][2] = math.cos(phi) * math.cos(omega)
    return r


def o3d_icp(ply_source, ply_target, trans_init, threshold):
    time1 = time()
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(ply_source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(ply_target)
    icp = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(icp)
    source.transform(icp.transformation)
    print(icp.transformation)
    final_points = np.array(source.points)
    time2 = time()
    print('Time cost: {}s'.format(time2 - time1))
    return final_points


def eval(ply_source, ply_target, ply_result):
    match_num = 0
    x_target = ply_target[:, 0]
    y_target = ply_target[:, 1]
    z_target = ply_target[:, 2]
    x_result = ply_result[:, 0]
    y_result = ply_result[:, 1]
    z_result = ply_result[:, 2]
    x_res =  x_result - x_target
    y_res =  y_result - y_target
    z_res =  z_result - z_target
    for i in range(len(x_res)):
        if (np.abs(x_res[i]) < 2.0 and np.abs(y_res[i]) < 2.0 and np.abs(z_res[i]) < 2.0):
            match_num += 1
    dis = np.multiply(x_res, x_res) + np.multiply(y_res, y_res) + np.multiply(z_res, z_res)
    mean_error = np.mean(dis)
    print('Mean Error: {:.3f}'.format(mean_error))
    print('Source points: {} Target points: {}'.format(len(ply_source[:, 0]), len(x_target)))
    print('Result points: {} Matched points {}'.format(len(x_result), match_num))

    
def ply_regis(mode, ply1_pth, ply2_pth):
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    points1 = load_ply(ply1_pth)
    x1 = points1[:, 0]
    y1 = points1[:, 1]
    z1 = points1[:, 2]
    mlab.points3d(x1, y1, z1, mode='point', color=(1, 0, 0), figure=fig)
    
    # points2 = load_ply(ply2_pth)
    points2 = rotate_ply(points1, 0, 0, math.pi / 2)
    x2 = points2[:, 0]
    y2 = points2[:, 1]
    z2 = points2[:, 2]
    mlab.points3d(x2, y2, z2, mode='point', color=(0, 1, 0), figure=fig)
    
    if mode == 'open3d':
        # Open3d ICP 
        threshold = 0.2
        trans_init = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0],
                           [0.0, 0.0, 0.0, 1.0]])
        points_2to1 = o3d_icp(points1, points2, trans_init, threshold)
        x3 = points_2to1[:, 0]
        y3 = points_2to1[:, 1]
        z3 = points_2to1[:, 2]
        mlab.points3d(x3, y3, z3, mode='point', color=(0, 0, 1), figure=fig)
        eval(points2, points1, points_2to1)
    
    elif mode == 'my':
        # My ICP
        points_2to1_my = icp.my_icp(points2, points1, max_iters=100)
        x3_my = points_2to1_my[:, 0]
        y3_my = points_2to1_my[:, 1]
        z3_my = points_2to1_my[:, 2]
        mlab.points3d(x3_my, y3_my, z3_my, mode='point', color=(0, 0, 1), figure=fig)
        eval(points2, points1, points_2to1_my)
    mlab.show()
    