import sys
import cv2.data
import numpy as np
import cv2
import time
import imutils
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
import calibration as calib
from matplotlib import pyplot as plt
from noise_player import Noise
from depth_displayer import DepthDisplayer

cam_L = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam_R = cv2.VideoCapture(2, cv2.CAP_DSHOW)

#Frame rate
fr = 60
#Dist between camera in cm, 9.4cm
B = 9.4
#Focal length of camera in mm
f = 3.95
#Angle of view of camera, or FOV
alpha = 78
#camera count
count = -1



max_depth = 500
min_depth = 20
warning1 = 50
warning2 = 100
warning3 = 150
DepthDisplayer = DepthDisplayer(max_depth, min_depth, warning1, warning2, warning3)


while(True):
    count += 1
    
    retR, fR = cam_R.read()
    retL, fL = cam_L.read()
    
    if not retR or not retL:
        print("Error: Cameras not found")
        break
        
    maskR = hsv.add_HSV(fR, 1)
    maskL = hsv.add_HSV(fL, 0)
    
    resR = cv2.bitwise_and(fR, fR, mask=maskR)
    resL = cv2.bitwise_and(fL, fL, mask=maskL)
    
    circleR = shape.find_circles(fR, maskR)
    circleL = shape.find_circles(fL, maskL)
    
    DepthDisplayer.display_depth(circleR, circleL, fR, fL, B, f, alpha, maskR, maskL)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam_L.release()
cam_R.release()
cv2.destroyAllWindows()