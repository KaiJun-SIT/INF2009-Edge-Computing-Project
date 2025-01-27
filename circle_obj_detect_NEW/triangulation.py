import sys
import cv2
import numpy as np
import time

def find_depth(circleR, circleL, fR, fL, B, f, alpha):
    # Check if either circle center is None
    if circleR is None or circleL is None:
        return None
        
    hR, wR, dR = fR.shape
    hL, wL, dL = fL.shape
    
    if wR == wL:
        f_pixel = (wR * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print("Error: Images not the same size")
        return None
    
    try:
        xR = circleR[0]
        xL = circleL[0]
        
        disparity = xL - xR
        
        # Avoid division by zero
        if disparity == 0:
            return None
            
        zDepth = (B * f_pixel) / disparity
        return abs(zDepth)
        
    except (IndexError, TypeError):
        return None
    
    