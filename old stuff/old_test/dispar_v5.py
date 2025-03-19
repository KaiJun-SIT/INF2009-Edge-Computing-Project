import numpy as np
import cv2
import time
from queue import Queue
from threading import Thread, Lock
import glob

class SynchronizedCamera:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.timestamp_queue = Queue(maxsize=1)
        self.running = True
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
        
        # Start capture thread
        self.thread = Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()
                # Clear queues to keep only latest frame
                while not self.frame_queue.empty():
                    self.frame_queue.get()
                while not self.timestamp_queue.empty():
                    self.timestamp_queue.get()
                self.frame_queue.put(frame)
                self.timestamp_queue.put(timestamp)
            time.sleep(0.001)  # Small delay to prevent busy-waiting
    
    def get_frame(self):
        if self.frame_queue.empty() or self.timestamp_queue.empty():
            return None, None
        return self.frame_queue.get(), self.timestamp_queue.get()
    
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def calibrate_stereo_cameras(left_id, right_id, checkerboard=(9,6)):
    """
    Calibrate stereo cameras using a checkerboard pattern.
    Returns calibration parameters needed for stereo rectification.
    """
    print("Starting stereo calibration...")
    print("Please hold the checkerboard pattern in different positions/orientations.")
    print("Press 'c' to capture a calibration image, 'q' to finish capturing.")
    
    # Initialize synchronized cameras
    left_cam = SynchronizedCamera(left_id)
    right_cam = SynchronizedCamera(right_id)
    time.sleep(1)  # Wait for cameras to initialize
    
    # Prepare object points
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints_left = []  # 2D points in left image plane
    imgpoints_right = []  # 2D points in right image plane
    
    captures = 0
    # In the calibrate_stereo_cameras function, modify the key capture part:
    while True:
        left_frame, left_ts = left_cam.get_frame()
        right_frame, right_ts = right_cam.get_frame()
        
        if left_frame is None or right_frame is None:
            continue
            
        # Ensure frames are synchronized
        if abs(left_ts - right_ts) > 0.033:  # More than one frame difference
            continue
        
        # Show frames
        combined = np.hstack((left_frame, right_frame))
        cv2.putText(combined, f'Captures: {captures}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Stereo Calibration', combined)
        
        # Add a small delay and make sure window is active
        key = cv2.waitKey(30) # Increased delay to 30ms
        
        # Print key value for debugging
        if key != -1:
            print(f"Key pressed: {key}")
        
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('C'):  # Accept both lower and uppercase
            print("Capture initiated...")
            # Rest of the capture code...
            # Convert to grayscale
            gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard, None)
            
            if ret_left and ret_right:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
                
                # Store points
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                
                captures += 1
                print(f"Captured calibration pair {captures}")
    
    # Release everything
    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()
    
    if captures < 10:
        print("Not enough captures for reliable calibration!")
        return None
    
    print("Calibrating cameras...")
    # Calibrate each camera individually
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    
    # Stereo calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        gray_left.shape[::-1], criteria=criteria_stereo, flags=flags)
    
    print("Calibration complete!")
    return mtx_left, dist_left, mtx_right, dist_right, R, T

def main():
    # First test available cameras
    available_cameras = []
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print("Available cameras:", available_cameras)
    if len(available_cameras) < 2:
        print("Not enough cameras found!")
        return
    
    left_id = int(input("Enter left camera ID: "))
    right_id = int(input("Enter right camera ID: "))
    
    # Perform stereo calibration
    calib_params = calibrate_stereo_cameras(left_id, right_id)
    if calib_params is None:
        return
    
    mtx_left, dist_left, mtx_right, dist_right, R, T = calib_params
    
    # Save calibration parameters
    cv_file = cv2.FileStorage("stereo_calibration.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("mtx_left", mtx_left)
    cv_file.write("dist_left", dist_left)
    cv_file.write("mtx_right", mtx_right)
    cv_file.write("dist_right", dist_right)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.release()
    
    print("Calibration parameters saved to stereo_calibration.xml")

if __name__ == "__main__":
    main()