import numpy as np

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

def applyHomography2Image(image, H, size):
    h, w = image.shape[:2] 
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size)))

    trans_lin_homg_pts = H.dot(lin_homg_pts)
    trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

    X_min = np.min(trans_lin_homg_pts[0,:])
    Y_min = np.min(trans_lin_homg_pts[1,:])
    X_max = np.max(trans_lin_homg_pts[0,:])
    Y_max = np.max(trans_lin_homg_pts[1,:])

    image_transformed = np.zeros((int(Y_max - Y_min) + 1, int(X_max - X_min) + 1))
    x, y = trans_lin_homg_pts[:2,:].astype(int)
    image_transformed[y, x] = image[Yi.ravel(), Xi.ravel()]
    image_cropped = image_transformed[0:size[1], 0:size[0]]

    return image_cropped


def applyHomography2ImageUsingInverseWarping(image, H, size):

    Xt, Yt = np.indices((size[0], size[1]))
    lin_homg_pts_trans = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    lin_homg_pts = H_inv.dot(lin_homg_pts_trans)
    lin_homg_pts /= lin_homg_pts[2,:]

    X_min = np.min(lin_homg_pts[0,:]).astype(int)
    Y_min = np.min(lin_homg_pts[1,:]).astype(int)
    X_max = np.max(lin_homg_pts[0,:]).astype(int)
    Y_max = np.max(lin_homg_pts[1,:]).astype(int)
    X_min, X_max, Y_min, Y_max

    #pad image
    max_val = np.max([X_max, image.shape[0], Y_max, image.shape[1]])
    image_i = np.zeros((max_val, max_val, 3))
    print(image_i.shape)
    image_i[0:image.shape[0], 0:image.shape[1], :] = image

    image_transformed = np.zeros((size[0], size[1], 3))
    Xi, Yi = lin_homg_pts[:2,:].astype(int)
    image_transformed[Xt.ravel(), Yt.ravel(), :] = image_i[Yi, Xi, :]
    
    return image_transformed

def createGaussianMask(image_size, sigma_x, sigma_y):
    cols, rows = image_size
    centre_x, centre_y = rows / 2, cols / 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(x,y)
    mask = np.exp(-(np.square((X - centre_x)/sigma_x) + np.square((Y - centre_y)/sigma_y)))
    return mask

def arrangeCorners(corners):
    #leftmost poin
    dist = np.square(corners[:,0]) + np.square(corners[:,1])
    arrnged_corners = []

    for i in range(corners.shape[0]):
        index = np.argmax(dist)
        arrnged_corners.append(corners[index,:])
        dist[index] = -1

    temp = arrnged_corners[2].copy()
    arrnged_corners[2] = arrnged_corners[-1]
    arrnged_corners[3] = temp

    return np.array(arrnged_corners)
