import numpy as np
import cv2
from Utils.MathUtils import *

def findContour(image):
    
    ret,thresh = cv2.threshold(np.uint8(image), 200 ,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chosen_contours = []

    previous_area = cv2.contourArea(contours[0])
    for j in range(len(contours)):
        if hierarchy[0, j, 3] == -1:#no parent
            if hierarchy[0, j, 2] !=-1: #child
                #print("no parent, child present")
                area = cv2.contourArea(contours[j])
                if True: #np.abs(area - previous_area) < 1000:
                    chosen_contours.append(contours[j])
                    previous_area = area
    return chosen_contours

def getTagMask(image):
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (21, 21), 0)

    chosen_contours = findContour(image_gray)
    out_mask = np.zeros_like(image_gray)
    corners = []
    for chosen_contour in chosen_contours:
        corner = cv2.approxPolyDP(chosen_contour, 0.009 * cv2.arcLength(chosen_contour, True), True)
        corners.append(corner.reshape(-1,2))
        cv2.drawContours(out_mask, [chosen_contour], -1, 1, cv2.FILLED)  

    out_mask_mul = np.dstack((out_mask, out_mask, out_mask))
    detected_april_tag = image * out_mask_mul
    return detected_april_tag

    
def getTagCorners(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.medianBlur(image_gray, 3)

    (T, thresh) = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    if hierarchy is not None:
        for j, cnt in zip(hierarchy[0], contours):
            cnt_len = cv2.arcLength(cnt,True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len,True)
            # ctr.append(cnt)

            if cv2.isContourConvex(cnt) and len(cnt) == 4 and cv2.contourArea(cnt) > 500 :
                cnt=cnt.reshape(-1,2)
                #if j[0] == -1 and j[1] == -1 and j[3] != -1:
                if j[2] != -1 and j[3] != -1:
                    ctr.append(cnt)
    return ctr

def extractInfoFromTag(tag):
    tag_size = tag.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(tag_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            grid = tag[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            
            if np.sum(grid) > 100000*0.7 and np.median(grid) == 255:
                # print(np.sum(grid))
                info_with_padding[i,j] = 255
    # print(info_with_padding)
    info = info_with_padding[2:6, 2:6]
    return info

def decipherInfoFromTag(info):
    while not info[3,3]:
        info = np.rot90(info, 1)

    # print(info)
    id_info = info[1:3, 1:3]
    id_info_flat = np.array([id_info[0,0], id_info[0,1], id_info[1,1], id_info[1,0]])
    tag_id = 0
    tag_id_bin = []
    for i in range(4):
        if(id_info_flat[i]):
            tag_id = tag_id + 2**(i)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)

    tag_id_bin.reverse()

    return tag_id, tag_id_bin
