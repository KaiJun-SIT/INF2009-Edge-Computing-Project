import cv2
import time

def test_webcams(max_to_test=4):
    """Safely test webcams with proper cleanup"""
    available_cameras = []
    
    for i in range(max_to_test):
        print(f"\nTesting webcam {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Using DirectShow for Windows
        
        if cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret:
                    print(f"Webcam {i} is working")
                    height, width = frame.shape[:2]
                    print(f"Resolution: {width}x{height}")
                    available_cameras.append(i)
                    
                    # Show brief preview
                    cv2.imshow(f'Webcam {i}', frame)
                    cv2.waitKey(500)
                    cv2.destroyWindow(f'Webcam {i}')
            except:
                print(f"Error reading from webcam {i}")
            finally:
                cap.release()
        else:
            print(f"Could not open webcam {i}")
        
        time.sleep(0.5)  # Give time for webcam to release
        
    cv2.destroyAllWindows()
    return available_cameras

if __name__ == "__main__":
    print("Checking available webcams...")
    available = test_webcams()
    print("\nAvailable webcam IDs:", available)
    print("\nTip: Use these IDs for CamL_id and CamR_id in your stereo setup")