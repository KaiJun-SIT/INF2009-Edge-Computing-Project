import sys
import cv2.data
import numpy as np
import cv2
import time
import imutils
from matplotlib import pyplot as plt

import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
import calibration as calib



cam_L = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam_R = cv2.VideoCapture(2, cv2.CAP_DSHOW)

fr = 60

B = 9.4
f = 3.95
alpha = 78

count = -1

while(True):
    count += 1
    
    retR, fR = cam_R.read()
    retL, fL = cam_L.read()
    
#calibration
    #fR, fL = calib.calibrate(fR, fL)


    if cam_L == False or cam_R == False:
        print("Error: Cameras not found")
        break
    else:
        maskR = hsv.add_HSV(fR, 1)
        maskL = hsv.add_HSV(fL, 0)
        
        resR = cv2.bitwise_and(fR, fR, mask=maskR)
        resL = cv2.bitwise_and(fL, fL, mask=maskL)
        
        circleR = shape.find_circles(fR, maskR)
        circleL = shape.find_circles(fL, maskL)
        
        #calculate depth
        
        if np.all(circleR) == None or np.all(circleL) == None:
            cv2.putText(fR, "track lost", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(fL, "track lost", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            depth = tri.find_depth(circleR, circleL, fR, fL, B, f, alpha)
            cv2.putText(fR, "tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(fL, "tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(fR, "Distance: " + str(round(depth,3)) + " cm", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(fL, "Distance: " + str(round(depth,3)) + " cm", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            print("Depth: " + str(depth) + " cm")
           # mulitply com value wirh 205.8 to get cm
           
        cv2.imshow("Right Camera", fR)
        cv2.imshow("Left Camera", fL)
        cv2.imshow("Right Mask", maskR)
        cv2.imshow("Left Mask", maskL)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cam_L.release()
cam_R.release()

cv2.destroyAllWindows()