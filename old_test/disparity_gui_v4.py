import numpy as np
import cv2
import time
import math

# Camera setup parameters
BASELINE = 0.095  # Distance between cam
FOV = 78  # Field of view in degrees
FRAME_WIDTH = 640  # Default frame width in pixels
FRAME_HEIGHT = 480  # Default frame height

# Calculate focal length based on FOV and frame width
FOCAL_LENGTH = (FRAME_WIDTH/2) / math.tan(math.radians(FOV/2))

def test_webcams(max_to_test=4):
    """Safely test webcams with proper cleanup"""
    print("Scanning for available cameras...")
    available_cameras = []
    
    for i in range(max_to_test):
        print(f"\nTesting webcam {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret:
                    print(f"Webcam {i} is working")
                    height, width = frame.shape[:2]
                    print(f"Resolution: {width}x{height}")
                    available_cameras.append(i)
                    
                    cv2.imshow(f'Webcam {i}', frame)
                    cv2.waitKey(500)
                    cv2.destroyWindow(f'Webcam {i}')
            except:
                print(f"Error reading from webcam {i}")
            finally:
                cap.release()
        else:
            print(f"Could not open webcam {i}")
        
        time.sleep(0.5)
    
    cv2.destroyAllWindows()
    return available_cameras

def nothing(x):
    pass

def create_minimal_gui():
    """Create minimal GUI with essential parameters"""
    print("Creating GUI...")
    cv2.namedWindow('Stereo Vision Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Vision Controls', 500, 400)
    
    cv2.createTrackbar('Block Size (2n+5)', 'Stereo Vision Controls', 5, 25, nothing)
    cv2.createTrackbar('Num Disparities (16n)', 'Stereo Vision Controls', 2, 10, nothing)
    cv2.createTrackbar('Uniqueness Ratio', 'Stereo Vision Controls', 10, 20, nothing)
    cv2.createTrackbar('Min Disparity', 'Stereo Vision Controls', 0, 10, nothing)
    
    controls_bg = np.zeros((150, 500), dtype=np.uint8)
    cv2.putText(controls_bg, 'Camera Setup:', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, f'Baseline: {BASELINE*100:.1f} cm', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, f'FOV: {FOV} degrees', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, 'Current Depth: waiting...', (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Stereo Vision Controls', controls_bg)

def initialize_cameras(left_id, right_id):
    """Initialize stereo cameras using DirectShow"""
    print(f"Initializing cameras: Left={left_id}, Right={right_id}")
    left = cv2.VideoCapture(left_id, cv2.CAP_DSHOW)
    time.sleep(0.5)
    right = cv2.VideoCapture(right_id, cv2.CAP_DSHOW)
    
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Error: Cannot open one or both cameras")
    
    for cam in [left, right]:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, 30)
    
    return left, right

def disparity_to_depth(disparity):
    """Convert disparity to depth map"""
    disparity[disparity == 0] = 0.1
    depth = (FOCAL_LENGTH * BASELINE) / disparity
    return depth

def update_controls_display(depth_value):
    """Update the controls window with current depth value"""
    controls_bg = np.zeros((150, 500), dtype=np.uint8)
    cv2.putText(controls_bg, 'Camera Setup:', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, f'Baseline: {BASELINE*100:.1f} cm', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, f'FOV: {FOV} degrees', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_bg, f'Current Depth: {depth_value:.2f} m', (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Stereo Vision Controls', controls_bg)

def combine_frames(left_frame, right_frame):
    """Combine left and right frames into a single image"""
    # Add labels to the frames
    cv2.putText(left_frame, 'Left', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(right_frame, 'Right', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Combine frames horizontally
    combined = np.hstack((left_frame, right_frame))
    
    # Add a center line
    height = combined.shape[0]
    center_x = combined.shape[1] // 2
    cv2.line(combined, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    
    return combined

def main():
    try:
        print(f"System Parameters:")
        print(f"Baseline: {BASELINE*100:.1f} cm")
        print(f"Field of View: {FOV} degrees")
        print(f"Calculated Focal Length: {FOCAL_LENGTH:.1f} pixels")
        
        available_cameras = test_webcams()
        
        if len(available_cameras) < 2:
            print(f"Error: Found only {len(available_cameras)} cameras. Need at least 2.")
            return
        
        print("\nAvailable cameras:", available_cameras)
        left_id = int(input("Enter the camera ID for left camera: "))
        right_id = int(input("Enter the camera ID for right camera: "))
        
        left_cam, right_cam = initialize_cameras(left_id, right_id)
        create_minimal_gui()
        stereo = cv2.StereoBM_create()
        
        # Create windows with specific sizes
        cv2.namedWindow('Stereo Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stereo Feed', FRAME_WIDTH*2, FRAME_HEIGHT)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Map', FRAME_WIDTH, FRAME_HEIGHT)
        
        print("\nStarting stereo vision...\nPress 'q' to quit")
        print("Click on the Depth Map window to measure distance at that point")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depth_value = depth_map[y, x]
                update_controls_display(depth_value)
                print(f"Depth at ({x}, {y}): {depth_value:.2f} meters")
        
        cv2.setMouseCallback('Depth Map', mouse_callback)
        
        while True:
            ret_left, left_frame = left_cam.read()
            ret_right, right_frame = right_cam.read()
            
            if not ret_left or not ret_right:
                print("Failed to capture frames, retrying...")
                time.sleep(0.1)
                continue
            
            # Combine color frames
            combined_feed = combine_frames(left_frame.copy(), right_frame.copy())
            
            # Process frames for depth mapping
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            block = cv2.getTrackbarPos('Block Size (2n+5)', 'Stereo Vision Controls') * 2 + 5
            num_disp = cv2.getTrackbarPos('Num Disparities (16n)', 'Stereo Vision Controls') * 16
            uniqueness = cv2.getTrackbarPos('Uniqueness Ratio', 'Stereo Vision Controls')
            min_disp = cv2.getTrackbarPos('Min Disparity', 'Stereo Vision Controls')
            
            stereo.setBlockSize(block)
            stereo.setNumDisparities(num_disp)
            stereo.setUniquenessRatio(uniqueness)
            stereo.setMinDisparity(min_disp)
            
            disparity = stereo.compute(left_gray, right_gray)
            disparity = np.float32(disparity) / 16.0
            
            depth_map = disparity_to_depth(disparity.copy())
            
            depth_map = np.clip(depth_map, 0, 5)
            normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            normalized_depth = np.uint8(normalized_depth)
            
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            
            # Display combined feed and depth map
            cv2.imshow('Stereo Feed', combined_feed)
            cv2.imshow('Depth Map', depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
    
    finally:
        print("Cleaning up...")
        try:
            left_cam.release()
            right_cam.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("Program terminated")

if __name__ == "__main__":
    main()  