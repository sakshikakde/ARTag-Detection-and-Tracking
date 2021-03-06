import numpy as np
import math

def computeHomography(corners1, corners2):

    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x = corners1[:, 0]
    y = corners1[:, 1]
    xp = corners2[:, 0]
    yp = corners2[:,1]

    nrows = 8
    ncols = 9
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    # print("the Homography matrix is")
    # print(H)
    return H

def applyHomography2Points(points, H):
    Xi = points[:, 0]
    Yi = points[:, 1]
    lin_homg_pts = np.stack((Xi, Yi, np.ones(Xi.size)))

    lin_homg_pts_trans = H.dot(lin_homg_pts)
    lin_homg_pts_trans /= lin_homg_pts_trans[2,:]

    Xt, Yt = lin_homg_pts_trans[:2,:].astype(int)
    points_trans = np.dstack([Xt, Yt])
    return points_trans

def applyHomography2ImageUsingForwardWarping(image, H, size, background_image = None):
    cols, rows = size
    h, w = image.shape[:2] 
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size)))
    trans_lin_homg_pts = H.dot(lin_homg_pts)
    trans_lin_homg_pts /= (trans_lin_homg_pts[2,:] + 1e-7)
    trans_lin_homg_pts = np.round(trans_lin_homg_pts).astype(int)


    if background_image is None:
        image_transformed = np.zeros((rows, cols, 3)) 
    else:
        image_transformed = background_image
    x = trans_lin_homg_pts[0,:]
    y = trans_lin_homg_pts[1,:]
    

    x[x >= cols] = cols - 1
    y[y >= rows] = rows - 1
    x[x < 0] = 0
    y[y < 0] = 0

    image_transformed[y, x] = image[Yi.ravel(), Xi.ravel()]
    image_transformed = np.uint8(image_transformed)
    return image_transformed


def applyHomography2ImageUsingInverseWarping(image, H, size):

    Yt, Xt = np.indices((size[0], size[1]))
    lin_homg_pts_trans = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    lin_homg_pts = H_inv.dot(lin_homg_pts_trans)
    lin_homg_pts /= lin_homg_pts[2,:]

    Xi, Yi = lin_homg_pts[:2,:].astype(int)
    Xi[Xi >=  image.shape[1]] = image.shape[1]
    Xi[Xi < 0] = 0
    Yi[Yi >=  image.shape[0]] = image.shape[0]
    Yi[Yi < 0] = 0

    image_transformed = np.zeros((size[0], size[1], 3))
    image_transformed[Yt.ravel(), Xt.ravel(), :] = image[Yi, Xi, :]
    
    return image_transformed

def computeProjectionMatrix(H, K):
    K_inv = np.linalg.inv(K)

    B_tilda = np.dot(K_inv, H)
    B_tilda_mod = np.linalg.norm(B_tilda)
    if B_tilda_mod < 0:
        B = -1  * B_tilda
    else:
        B =  B_tilda

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    lambda_ = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    lambda_ = 1 / lambda_

    r1 = lambda_ * b1
    r2 = lambda_ * b2
    r3 = np.cross(r1, r2)
    t = lambda_ * b3

    P = np.array([r1,r2, r3, t]).T
    P = np.dot(K, P)
    P = P / P[2,3]
    return P

def applyProjectionMatrix2Points(points, P):
    Xi = points[:, 0]
    Yi = points[:, 1]
    Zi = points[:, 2]

    lin_homg_pts = np.stack((Xi, Yi, Zi, np.ones(Xi.size)))
    lin_homg_pts_trans = P.dot(lin_homg_pts)
    lin_homg_pts_trans /= lin_homg_pts_trans[2,:]
    x = lin_homg_pts_trans[0,:].astype(int)
    y = lin_homg_pts_trans[1,:].astype(int)

    projected_points = np.dstack((x,y)).reshape(4,2)
    return projected_points


