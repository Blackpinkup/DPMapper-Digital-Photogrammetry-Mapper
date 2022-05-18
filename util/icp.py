import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from time import time


def my_icp(ply_source, ply_target, init_pose=None, max_iters=100, threshold=0.005):
    '''
    Iterative Closet Point method
    input:
        ply_a, ply_b: numpy array of points
        init_pose: (dim + 1) * (dim + 1) homogeneous transformation
        max_iters: the maximum of iterations
        threshold: exit criteria
    ouput:
        final_trans: final homogeneous transformation for source to target registration 
        dis: euclidean distances of nearest neighbor
        iter: number of iterations
    '''
    time1 = time()
    assert ply_source.shape == ply_target.shape
    dim = ply_source.shape[1]
    # Make copy of original ply
    source = np.ones((dim + 1, ply_source.shape[0]))
    target = np.ones((dim + 1, ply_target.shape[0]))
    source[:dim, :] = np.copy(ply_source.T)
    target[:dim, :] = np.copy(ply_target.T)
    # Apply initial pose estimation
    if init_pose != None:
        source = np.dot(init_pose, source)
    # Iteration
    final_error = 0
    for i in range(max_iters):
        # Find the nearest neighbors between source and target points
        dis, ind = nearest_neighbors(source[:dim, :].T, target[:dim, :].T)
        # Compute the transformation between source and nearest target points
        final_trans, _, _ = fit_trans(source[:dim, :].T, target[:dim, ind].T)
        # Update source points
        source = np.dot(final_trans, source)
        # Judge if exit by error
        mean_error = np.mean(dis)
        if np.abs(final_error - mean_error) < threshold:
            break
        print('Iteration: {} Error: {:.3f}'.format(i, np.abs(final_error - mean_error)))
        final_error = mean_error

    # Calculate final transformation
    final_trans, _, _ = fit_trans(ply_source, source[:dim, :].T)
    # return final_trans, dis, ind

    # Final transformation apply to source points
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(ply_source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(ply_target)
    print(final_trans)
    source.transform(final_trans)
    final_points = np.array(source.points)
    time2 = time()
    print('Time cost: {}s'.format(time2 - time1))
    return final_points
    
        
def nearest_neighbors(source, target):
    '''
    Find the nearest neighbor between two points cloud
    input:
        source, target: numpy array of points
    output:
        dis: euclidean distances of the nearest neighbor
        ind: dis indices of the neraest neighbor
    '''
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(target)
    dis, ind = nn.kneighbors(source, return_distance=True)
    return dis.ravel(), ind.ravel()


def fit_trans(source, target):
    '''
    Calculate best fit transformation according to least square method
    input:
        source, target: numpy array of points
    output:
        final_trans: (dim + 1) * (dim + 1) homogeneous transformation matrix
        final_r: dim * dim rotation matrix
        final_t: dim * 1 translation vector
    '''
    dim = source.shape[1]
    # Translate points to centroids
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    source_c = source - centroid_source
    target_c = target - centroid_target
    # Calculate rotation matrix by SVD
    H = np.dot(source_c.T, target_c)
    U, S, V = np.linalg.svd(H)
    final_r = np.dot(V.T, U.T)
    # Consider special reflection
    if np.linalg.det(final_r) < 0:
        V[dim-1, :] *= -1
        final_r = np.dot(V.T, U.T)
    # Calculate Translation
    final_t = centroid_target.T - np.dot(final_r, centroid_source.T)
    # Calculate homogeneous transformation
    final_trans = np.identity(dim + 1)
    final_trans[:dim, :dim] = final_r
    final_trans[:dim, dim] = final_t
    return final_trans, final_r, final_t
        