import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
from numpy import linalg as la
import math
from PIL import Image
import random
import cv2

def Edgedetection(image,old_ctr):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray,3)
    (T, thresh) = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    for j, cnt in zip(hierarchy[0], contours):
        cnt_len = cv2.arcLength(cnt,True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len,True)
        if cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and len(cnt) == 4  :
            cnt=cnt.reshape(-1,2)
            if j[0] == -1 and j[1] == -1 and j[3] != -1:
                ctr.append(cnt)
        #print(np.shape(ctr))
        old_ctr=ctr
    return ctr

def Imageprocessor(path,src):

    BasePath = '/home/sakshi/courses/ENPM673/sakshi_p1/'
    video_file = BasePath + "Data/multipleTags.mp4"

    vidObj = cv2.VideoCapture(video_file)
    count = 0
    success = 1
    img_array=[]

    while (success):
        if (count==0):
            success, image = vidObj.read()
        height,width,layers= image.shape

        size = (width,height)
        if (count==0):
            old_corners=0
        corners=Edgedetection(image,old_corners)
        img = cv2.drawContours(image, corners,0,(0,255,0),5)
        if(len(corners)==0):
            corners=old_corners


        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # tag_image,Tag=perspective_for_tag(corners,image)
        # new_corners,tag_id=Tag_id_detection(corners,tag_image)


        # image,h=Superimposing(new_corners,image,src)
        # proj_mat=Projection_mat(h)
        # image=Cube3D(proj_mat,image)
        # old_corners=corners
        # count += 1
        # print('Number of frames is',count)
        # cv2.imwrite('%d.jpg' %count,image)
        # img_array.append(image)
        # success, image = vidObj.read()

    # return img_array,size

if __name__ == '__main__':
    src = 0
    Imageprocessor('Tag0.mp4',src)