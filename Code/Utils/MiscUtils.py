import numpy as np

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

def rotatePoints(points):
    point_list = list(points.copy())
    top = point_list.pop(-1)
    point_list.insert(0, top)
    return np.array(point_list)
