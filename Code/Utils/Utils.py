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

def applyHomography2ImageUsingForwardWarping(image, H, size):
    cols, rows = size
    h, w = image.shape[:2] 
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size)))
    trans_lin_homg_pts = H.dot(lin_homg_pts)
    trans_lin_homg_pts /= (trans_lin_homg_pts[2,:] + 0.01)
    trans_lin_homg_pts = np.round(trans_lin_homg_pts).astype(int)

    X_min = np.min(trans_lin_homg_pts[0,:])
    Y_min = np.min(trans_lin_homg_pts[1,:])
    X_max = np.max(trans_lin_homg_pts[0,:])
    Y_max = np.max(trans_lin_homg_pts[1,:])

    image_transformed = np.zeros((rows, cols, 3)) 
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

    # print(X_min, X_max, Y_min, Y_max)

    #pad image
    max_val = np.max([X_max - X_min, image.shape[0], Y_max - Y_min, image.shape[1]])
    image_i = np.zeros((max_val, max_val, 3))
    # print(image_i.shape)
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

def sortCorners(points):

    x_sorted = points[np.argsort(points[:,0])]

    points_left = x_sorted[0:2, :]
    points_right = x_sorted[2:4, :]

    left_sorted_y = points_left[np.argsort(points_left[:,1])]
    tl, bl = left_sorted_y

    right_sorted_y = points_right[np.argsort(points_right[:,1])]
    tr, br = right_sorted_y
    points_sorted = np.array([tl, bl, br, tr])
    return points_sorted

def extractInfoFromTag(tag):
    tag_size = tag.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(tag_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            grid = tag[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            
            if np.sum(grid) > 100000*0.7 and np.median(grid) == 255:
                print(np.sum(grid))
                info_with_padding[i,j] = 255
    print(info_with_padding)
    info = info_with_padding[2:6, 2:6]
    return info