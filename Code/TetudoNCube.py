
import numpy as np
import cv2
import scipy
import math
import matplotlib.pyplot as plt
import argparse
from Utils.MathUtils import *
from Utils.ImageUtils import *
from Utils.MiscUtils import *
from Utils.KalmanFilter import *
from Utils.MovingAverage import *

import csv 

K = np.array([[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]])
K = np.transpose(K)


def drawCube(image, bottom_points, top_points):

    cv2.drawContours(image, [bottom_points], 0, (255, 0 ,255),3)
    cv2.drawContours(image, [top_points], 0, (255, 0, 255),3)

    for i in range(0, bottom_points.shape[0]):
        color = (int(255/(i+1)), 0, int(255/(i+1)))
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), color, 3)

    return image


def main(): 

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='./', help='Base path of project1, Default:./')
    Parser.add_argument('--VideoFilePath', default='./Data/Tag0.mp4', help='MP4 file name, Default:Tag2.mp4')
    Parser.add_argument('--SaveFileName', default='Results/problem2/testudo/tag0.avi', help='Folder to save graphs, Default:testudo/Tag2_unfiltered.avi')
    Parser.add_argument('--ProjectTestudo', default=True,type=lambda x: bool(int(x)))
    Parser.add_argument('--UseFilter',  default=False,type=lambda x: bool(int(x)))

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    VideoFilePath = Args.VideoFilePath
    SaveFileName = BasePath + Args.SaveFileName
    ProjectTestudo = Args.ProjectTestudo
    UseFilter = Args.UseFilter

    print("ProjectTestudo = ", ProjectTestudo)
    print("UseFilter = ", UseFilter)

    project_testudo = ProjectTestudo
    use_filter = UseFilter

    cap = cv2.VideoCapture(VideoFilePath)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(SaveFileName,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, (frame_width, frame_height)) 

    # f = open("corners.csv", 'w')
    # csvwriter = csv.writer(f)  

    # ff = open("cornersf.csv", 'w')
    # csvwriterf = csv.writer(ff)  


    testudoFileName = BasePath + "Data/testudo.png"
    testudo_image = cv2.imread(testudoFileName)
    if testudo_image is None:
        print("testudo image no found!")


    testudo_x = testudo_image.shape[1]
    testudo_y = testudo_image.shape[0]
    testudo_corners = np.array([[0,0], [0, testudo_y-1], [testudo_x-1, testudo_y-1], [testudo_x-1, 0]])
    testudo_corners = sortCorners(testudo_corners)



    tag_size = 160
    desired_tag_corner = sortCorners(np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]]))

    first_time = True
    window_size_base = 4
    window_size_top = 5
    count = 0
    rotation = 0 

    if use_filter:
        print("APPLYING FILTER TO  POINTS OF CUBE")
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = 1/fps

    while(True):
        
        ret, frame = cap.read()
        if not ret:
            print("Stream ended..")
            break
        
        image_rgb = frame
        rows,cols,ch = image_rgb.shape        
        detected_april_tag = np.uint8(getTagMask(image_rgb)) #without white paper

        if first_time:
            old_corners = getTagCorners(detected_april_tag)
            number_of_tags = len(old_corners)
            maBase = MovingAverage(window_size_base, 10)
            maTop = MovingAverage(window_size_top, 5)


        corners = getTagCorners(detected_april_tag)

        if(len(corners) < 1):
            corners = old_corners
        else:
            old_corners = corners

        image_show = image_rgb.copy()
        for corner in corners:
            set1 = testudo_corners
            set2 = sortCorners(corner)#from video

            if use_filter:
                if maBase.getListLength() <  window_size_base:
                    maBase.addQuadrilateral(set2)
                else:
                    maBase.addQuadrilateral(set2)
                    set2 = maBase.getAverage().astype(int)
                    # cv2.drawContours(image_rgb, [set2_filtered], 0, (255, 255, 0), 5)

            Htd = computeHomography(np.float32(set2), np.float32(desired_tag_corner))

            tag = applyHomography2ImageUsingInverseWarping(image_rgb, Htd, (tag_size, tag_size))
            tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
            ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
            tag_info = extractInfoFromTag(tag)
            ARcorners = np.array([tag_info[0,0], tag_info[0,3], tag_info[3,0], tag_info[3,3]])

            rotation = 0 
            if np.sum(ARcorners) == 255:
                while not tag_info[3,3]:
                    tag_info = np.rot90(tag_info, 1)
                    rotation = rotation + 90
                if first_time:
                    old_rotation = rotation                
                del_rotation = np.abs(old_rotation - rotation)
                if del_rotation == 270:
                    del_rotation = 90
                # np.minimum(np.abs(old_rotation - rotation), np.abs(old_rotation - rotation - 360))
                if (del_rotation > 100): #basically, greater than 90#REVIEW
                    print("del rotation high", del_rotation)
                    # rotation = old_rotation

                old_rotation = rotation
            else:
                print("Tag not detected properly, using old rotation!")
                rotation = old_rotation

            # print("rotation = ", rotation)
            num_rotations = int(rotation/90)
            for n in range(num_rotations):
                set2 = rotatePoints(set2)

            if project_testudo:
                H12 = computeHomography(set1, set2)
                set1_trans = applyHomography2Points(set1, H12)
                cv2.drawContours(detected_april_tag, [set1_trans], 0, (0,255,255),3)
                testudo_transormed = applyHomography2ImageUsingForwardWarping(testudo_image, H12, (cols, rows), background_image = image_show)

            else: #projecting cube
                cube_height = np.array([-(tag_size-1), -(tag_size-1), -(tag_size-1), -(tag_size-1)]).reshape(-1,1)
                cube_corners = np.concatenate((desired_tag_corner, cube_height), axis = 1)
                Hdt = computeHomography(np.float32(desired_tag_corner), np.float32(set2))
                P = computeProjectionMatrix(Hdt, K)
                set2_top = applyProjectionMatrix2Points(cube_corners, P)

                ############### apply Filter to set2_top(top of cube) ##REVIEW
                if use_filter:
                    if maTop.getListLength() <  window_size_top:
                        maTop.addQuadrilateral(set2_top)
                    
                    else:
                        maTop.addQuadrilateral(set2_top)
                        set2_top = maTop.getAverage().astype(int)
                        # cv2.drawContours(image_rgb, [set2_top_filtered], 0, (255, 255, 0), 5)

                    # # X = np.hstack((pos, vel)).T
                    # if first_time:
                    #     pos = np.hstack((set2_top[:,0], set2_top[:,1])).reshape(8,1)
                    #     vel = np.zeros((8,1))
                    #     acc = np.zeros((8,1))
                    #     acc_variance = 100
                    #     kf = KalmanFilter(pos, vel, acc, acc_variance)
                    #     print("KF initialized")

                    # if not first_time:
                    #     # print("predicting...")
                    #     field = np.hstack((set2_top[:,0], set2_top[:,1])).reshape(1,8)
                    #     csvwriter.writerow(field[0]) 

                    #     meas = np.hstack((set2_top[:,0], set2_top[:,1])).reshape(8,1)
                    #     meas_covar = np.eye(8) * 0.001

                    #     kf.predict(dt)
                    #     kf.update(meas, meas_covar, dt)
                    #     # X = kf.state()
                    #     # print(X.shape)

                    #     pos = kf.position()
                    #     vel = kf.velocity()
                    #     set2_pred = np.hstack((pos[0:4], pos[4:8]))
                    #     # print(set2_pred)
                    #     set2_pred = set2_pred.astype(int)

                                    #     # print("predicting...")
                        # field = np.hstack((set2_top[:,0], set2_top[:,1])).reshape(1,8)
                        # csvwriter.writerow(field[0]) 

                        # fieldf = np.hstack((set2_top_filtered[:,0], set2_top_filtered[:,1])).reshape(1,8)
                        # csvwriterf.writerow(fieldf[0]) 
                        # cv2.drawContours(image_rgb, [set2_pred], 0, (255, 255, 0), 5)

                    
                    # set2_filtered = 
            #########################################################

                image_show = drawCube(image_rgb, set2, set2_top)

        count = count + 1
        first_time = False

        cv2.imshow('frame', np.uint8(image_show))
        result.write(np.uint8(image_show)) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


