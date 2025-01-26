import sys
import cv2
import numpy as np
import time

def find_depth( circleR, circleL, fR, fL, B, f, alpha):
    
    hR, wR, dR = fR.shape
    hL, wL, dL = fL.shape
    
    if wR == wL:
        f_pixel = (wR * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
        
    else:
        print("Error: Images not the same size")
        return
    
    xR = circleR[0]
    xL= circleL[0]
    
    disparity = xL - xR
    
    zDepth = (B * f_pixel) / disparity
    
    return abs(zDepth)
    
    