# File Structure

-sakshi_p1    
--Code  
---ARDetection.py  
---TestudoNCube.py   

--Data   
---Tag0.mp4   
---Tag1.mp4   
---Tag2.mp4   
---multipleTags.mp4   
---testudo.png   
---ref_marker.png   

--Results   
---problem1   
---problem2   
----testudo   
----cube   

--README.md   
--ENPM673_P1_report.pdf   

# Problem 1

## How to run the code
Change the directory where the .py files are present    
/home/sakshi/courses/ENPM673/sakshi_p1/Code

python3 ARDetection.py --BasePath /home/sakshi/courses/ENPM673/sakshi_p1/ --VideoFilePath /home/sakshi/courses/ENPM673/sakshi_p1/Data/Tag2.mp4 --SavePath Results/problem1/

## Parameters
1) BasePath - BasePath - project folder 
2) VideoFilePath - absolute path of the video file
3) SavePath - Path to the folder where results are saved. Note: This path is relative to BasePath

# Problem 2

## How to run the code
Change the directory where the .py files are present    
/home/sakshi/courses/ENPM673/sakshi_p1/Code

 python3 TetudoNCube.py --BasePath /home/sakshi/courses/ENPM673/sakshi_p1/ --VideoFilePath /home/sakshi/courses/ENPM673/sakshi_p1/Data/Tag0.mp4 --SaveFileName Results/problem2/testudo/Tag0.avi --ProjectTestudo 1 --UseFilter 0


## Parameters

1) BasePath - project folder 
2) VideoFilePath - absolute path of the video file
3) SaveFileName - file name for the saved video. Note: this path is relative to the BasePath
4) ProjectTestudo - Set as 1 for projecting testudo image, 0 for cube
5) UseFilter - set 1 to use moving average filter. THIS WILL NOT WORK FOR MULTIPLE TAGS VIDEO. Please set it to 0 for multiple tags 
