import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from Utils.Utils import *

def createCircularMask(image_size, radius, high_pass = True):
    rows, cols = image_size
    centre_x, centre_y = int(rows / 2), int(cols / 2)
    center = [centre_x, centre_y]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius*radius

    if high_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask

def createCircularMask(image_size, radius, high_pass = True):
    rows, cols = image_size
    centre_x, centre_y = int(rows / 2), int(cols / 2)
    center = [centre_x, centre_y]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius*radius

    if high_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask

def findContour(image, SavePath, i):
    ret,thresh = cv2.threshold(np.uint8(image), 200 ,255,cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(thresh, cmap = 'gray')
    plt.savefig(SavePath + "thresh" + str(i) + ".jpg")
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chosen_contours = []
    for j in range(len(contours)):
        if hierarchy[0, j, 3] == -1:#no parent
            if hierarchy[0, j, 2] !=-1: #child
                print("no parent, child present")
                chosen_contours.append(contours[j])

    return chosen_contours  

def extractInfoFromTag(tag):

    tag_size = tag.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(tag_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            grid = tag[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            if np.sum(grid) > 100000 and np.median(grid) == 255:
                # print(np.sum(grid))
                info_with_padding[i,j] = 255
    info = info_with_padding[2:6, 2:6]
    return info

def decipherInfoFromTag(info):
    while not info[3,3]:
        info = np.rot90(info, 1)
    id_info = info[1:3, 1:3]
    id_info_flat = np.array([id_info[0,0], id_info[0,1], id_info[1,1], id_info[1,0]])
    tag_id = 0
    for i in range(4):
        if(id_info_flat[i]):
            tag_id = tag_id + 2**i
    return tag_id

def imageBlur(input_image, SavePath, i):

    image = input_image.copy()
    #fft
    fft = scipy.fft.fft2(image, axes = (0,1))
    fft_shifted = scipy.fft.fftshift(fft)
    magnitude_spectrum_fft_shifted = 20*np.log(np.abs(fft_shifted))

    #fft+mask
    fft_masked = fft_shifted * createGaussianMask(image.shape, 30, 30)
    magnitude_spectrum_masked= 20*np.log(np.abs(fft_masked))

    #image back
    img_back = scipy.fft.ifftshift(fft_masked)
    img_back = scipy.fft.ifft2(img_back)
    img_back = np.abs(img_back)

    fx, plts = plt.subplots(2,2,figsize = (10,5))
    plts[0][0].imshow(image, cmap = 'gray')
    plts[0][1].imshow(magnitude_spectrum_fft_shifted, cmap = 'gray')
    plts[1][0].imshow(magnitude_spectrum_masked, cmap = 'gray')
    plts[1][1].imshow(img_back, cmap = 'gray')

    plt.savefig(SavePath + "fft" + str(i) + ".jpg")

    return img_back


def findTag(chosen_frame, SavePath, i):

    image = chosen_frame.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_color = chosen_frame.copy()

    image_blur = imageBlur(image_gray, SavePath, i)
    chosen_contours = findContour(image_blur, SavePath, i)

    out_mask = np.zeros_like(image_gray)
    corners = []
    for chosen_contour in chosen_contours:
        corner = cv2.approxPolyDP(chosen_contours[0], 0.009 * cv2.arcLength(chosen_contours[0], True), True) 
        corners.append(corner.reshape(-1,2))
        cv2.drawContours(image_color, chosen_contour, -1, (0, 0 ,255), 3)
        cv2.drawContours(out_mask, [chosen_contour], -1, 1, cv2.FILLED)    


    out_mask_mul = np.dstack((out_mask, out_mask, out_mask))
    detected_april_tag = image_color * out_mask_mul
    for c in corners:
        for r in range(c.shape[0]):
            x,y = c[r,:]
            cv2.circle(detected_april_tag,(x,y),20,(255, 0, 255),-1) 
    plt.figure()
    plt.imshow(image_color)
    plt.savefig(SavePath + "contour" + str(i) + ".jpg")

    fx, plts = plt.subplots(1,2,figsize = (10,7))
    plts[0].imshow(out_mask, cmap = 'gray')
    plts[1].imshow(detected_april_tag)
    plt.savefig(SavePath + "detect" + str(i) + ".jpg")

    size_x = 400
    size_y = 300

    corner1 = corners[0]
    corner2 = np.array([[0, 0], [0, size_y], [size_x, size_y], [size_x, 0]])

    H = computeHomography(np.float32(corner1), np.float32(corner2))
    image_transformed = applyHomography2ImageUsingInverseWarping(image_gray, H, (size_x, size_y))
    plt.figure()
    plt.imshow(image_transformed, cmap = 'gray')
    plt.savefig(SavePath + "img_trans" + str(i) + ".jpg")

    ret,thresh = cv2.threshold(np.uint8(image_transformed), 200 ,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chosen_contours = []
    for j in range(len(contours)):
        if hierarchy[0, j, 3] == 0:#no parent
            if hierarchy[0, j, 2] !=-1: #child
                print("no parent, child present")
                chosen_contours.append(contours[j])

    tag_size = 160
    tag_corner = cv2.approxPolyDP(chosen_contours[0], 0.009 * cv2.arcLength(chosen_contours[0], True), True) 
    desired_corner = np.array([[0,0], [tag_size, 0], [tag_size, tag_size], [0, tag_size]])

    H = computeHomography(np.float32(tag_corner.reshape(4,2)), np.float32(desired_corner))
    tag = applyHomography2ImageUsingInverseWarping(image_transformed, H, (tag_size, tag_size))
    ret,tag = cv2.threshold(tag, 200 ,255,cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(tag, cmap = 'gray')
    plt.savefig(SavePath + "tag" + str(i) + ".jpg")

    return tag


def main():
    BasePath = '/home/sakshi/courses/ENPM673/sakshi_p1/'
    SavePath = BasePath + "Results/problem1/"
    video_file = BasePath + "Data/Tag2.mp4"
    cap = cv2.VideoCapture(video_file)

    frame_index = 10
    i = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Stream ended..")
            break
        i = i+1
        if i == frame_index:
            chosen_frame = frame
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    tag = findTag(chosen_frame, SavePath, frame_index)
    info = extractInfoFromTag(tag)
    tag_id = decipherInfoFromTag(info)
    print("The tag ID is ", tag_id)

if __name__ == '__main__':
    main()