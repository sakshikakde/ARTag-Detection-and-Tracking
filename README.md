# AR Tag detection and decoding
## Decode reference AR tag
Steps to decode the AR Tag:
- Convert the image to binary and resize it to a size of
160 × 160.
- Divide this 160 × 160 into 8 × 8 grids, i.e. each block
in the grids is of size 20 × 20.
- Take the median for each block and replace the 20 × 20 block with a single value depending on the median. It
will be either 255 or 0 since the image is binary.
- Extract the central 4 × 4 grids which have the rotational
and ID information.
- Check for the values at the four corners. Only one of
them should be high. If values at more than one corner
are high or no values are high, then AR-tag cannot be
detected, and hence shall be discarded.
- The upright position of the tag is determined by the
bottom right grid. Keep rotating the tag till the bottom
right grid is high.
- After the orientation is set, extract the central 2×2 grids
for the tag ID.
- Flatten the 2 × 2 matrix. The leftmost digit is the least
significant bit and the rightmost is the most significant
bit. Convert it to decimal to get the tag ID. Please note
that in figure 3 last image, the values are in reverse
order(left is LSB and right is MSB).
## Decode AR tag from video
### Detect Tag
At multiple times in the video, there are
a few background objects. To avoid false positives, I am first
detecting contours for a tag with white paper and then creating
a mask using the detected contours. I multiply the mask with
the frame and then use the result to detect the actual tag. The
results are shown in figure 4.
For contour detection, I used cv2.findContours() function. To
make the detection stable, I exploited the hierarchy that is a
return from the function. For detecting the contours for a white
paper, I accepted the contours which have no parent and at least one child, hence detecting only the outer boundary of
the tag.
For detecting the actual tag, I again used cv2.findContours(),
but on the masked image. To make detection stable, I used
contour area and contour perimeter as conditions to accept
the detected contours
### Warp tag
Once I get the four corners of the tag, I
sorted the corners in order as top-left, bottom-left, bottom-
right and top-right. These sorted four corners corresponds to
the tag corners as shown in figure 6. Given two sets of points, a homography matrix can be computed which can be used to
warp the image and obtain the AR-tag for decoding. Initially, I
tried warping the image using the forward warping technique.
However, the results were quite noisy. Later, I shifted to
inverse warping as discussed in the supplementary material
provided.
### Decode tag
Once the tag has been obtained, we can use
the same process as described in section I-B. Refer figure 8
for the extracted information. Please note that the leftmost bit
is LSB and the rightmost bit is MSB in figure 8. So the tag
ID will be reverse of [1 0 1 1], which is [1 1 0 1] in binary
and 13 in decimal.

## How to run the code
- Change the location to the root directory 
- Run the following command:
```  
python3 Code/ARDetection.py --BasePath ./ --VideoFilePath ./Data/Tag2.mp4 --SavePath Results/problem1/
```


## Parameters
1) BasePath - BasePath - project folder 
2) VideoFilePath - absolute path of the video file
3) SavePath - Path to the folder where results are saved. Note: This path is relative to BasePath

# Superimposing image onto tag
The first step is to detect the corners of the AR-tag as
discussed in section I-C1. Next, we need to make sure that
the corners are in the correct order, i.e. top-left, bottom-left,
bottom-right, and top-right. We extract the tag from each
frame and compute its orientation as discussed in section I-C3.
Depending on what the orientation is, we rotate our corner
points in that direction. For example, if the white cell is located
at the top-left position, then we need a rotation of 180 degrees
to make the tag upright. So, we will rotate the corner points
twice by 90 degrees. We are trying to superimpose the testudo image on the AR
tag. The corners of the testudo image can be easily found.
However, the order shall remain the same as top-left, bottom-
left, bottom-right, and top-right. Now that we have two sets
of ordered points, we can compute a homography matrix and
warp the testudo image on the video frame.

## How to run the code
- Change the location to the root directory 
- Run the following command: 
```
python3 TetudoNCube.py --BasePath ./ --VideoFilePath ./Data/Tag0.mp4 --SaveFileName Results/problem2/testudo/Tag0.avi --ProjectTestudo 1 --UseFilter 0
```

## Parameters

1) BasePath - project folder 
2) VideoFilePath - absolute path of the video file
3) SaveFileName - file name for the saved video. Note: this path is relative to the BasePath
4) ProjectTestudo - Set as 1 for projecting testudo image, 0 for cube
5) UseFilter - set 1 to use moving average filter. THIS WILL NOT WORK FOR MULTIPLE TAGS VIDEO. Please set it to 0 for multiple tags 

## Results
![alt](https://github.com/sakshikakde/ARTag-Detection-and-Tracking/blob/main/media/testudo.gif)

# Placing a virtual cube onto tag
Since a cube is a three dimensional entity, we need a
3 × 4 projection matrix to project 3D points into the image
plane. The basic assumption while calculating the homography
matrix is that the points are planar, i.e. z = 0 for all the points.
Now, we have z values as well. Thus we need to modify this H
matrix to suit our requirements. To steps to obtain a projection
matrix(P) from a homgraphy matrix(H) and camera calibration
matrix(K) are as follows:
- Lets write the homography matrix as H = K B̃.
- Then, B̃ = K −1 H
- We need to obtain matrix B such that B = λB̃
- Compute mod of B̃. if it is negative, then B̃ = −1 ∗ B̃, where b1 and b2
- We will calculate λ as λ = kb1k+kb2k
- Therefore, B = [λb1, λb2, λb3]
- B can be decomposed and written as [r1, r2, t], where r1
and r2 are rotational components and t is translational
component.
- We know that the rotation matrix is orthonormal; the
last column must be perpendicular to the first two[2].
Hence, r3 = r1 × r2
- The final projection matrix can be written as P =
K[r1, r2, r3, t].

## Results
![alt](https://github.com/sakshikakde/ARTag-Detection-and-Tracking/blob/main/media/cube.gif)

