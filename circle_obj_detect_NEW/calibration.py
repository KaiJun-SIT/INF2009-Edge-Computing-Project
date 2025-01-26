import cv2
import numpy as np

def calibrate(frameR, frameL):
   # Load calibration parameters
   cv_file = cv2.FileStorage("./edge_stereo_cam_embeddedAI/data/calib_generate_params_py.xml", cv2.FILE_STORAGE_READ)
   
   Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
   Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
   Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
   Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
   cv_file.release()

   # Rectify frames
   frame_right = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y,
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
   frame_left = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y,
                         cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                         
   return frame_right, frame_left