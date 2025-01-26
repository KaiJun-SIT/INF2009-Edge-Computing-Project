import sys 
import cv2
import numpy as np
import time

def add_HSV(frame, camera):
    
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    lower_rb = np.array([60, 110, 50])
    upper_rb = np.array([255, 255, 255])
    lower_lb = np.array([143, 110, 50])
    upper_lb = np.array([255, 255, 255])
    
    if(camera == 1):
        mask = cv2.inRange(hsv, lower_rb, upper_rb)
    else:
        mask = cv2.inRange(hsv, lower_lb, upper_lb)
        
        
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    
    return mask
