import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8-Pose model
model = YOLO("yolov8n-pose.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("YOLOv8-Pose model loaded successfully")

# **Confidence Threshold for Keypoint-Based Detection**
DETECTION_THRESHOLD = 0.5  # Lower to detect more humans
# ANOMALY_THRESHOLD = 0.6  # Adjust based on real-world testing

# **Anomaly Detection Function**
def detect_anomaly(results):
    """ Detects anomaly based on keypoint positions. """

    # Ensure YOLO detected humans
    if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        print("No valid human detected. Ignoring frame.")
        return False  # Ignore frames without valid humans

    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:  # Ensure enough keypoints
                continue

            # Extract keypoints (x, y) positions
            left_wrist, right_wrist = pose[9][:2].cpu().numpy(), pose[10][:2].cpu().numpy()
            left_shoulder, right_shoulder = pose[5][:2].cpu().numpy(), pose[6][:2].cpu().numpy()
            left_knee, right_knee = pose[13][:2].cpu().numpy(), pose[14][:2].cpu().numpy()
            left_hip, right_hip = pose[11][:2].cpu().numpy(), pose[12][:2].cpu().numpy()

            # **Check for Squatting (Possible Urination, Sitting in Restricted Areas)**
            avg_knee_height = (left_knee[1] + right_knee[1]) / 2
            avg_hip_height = (left_hip[1] + right_hip[1]) / 2
            if avg_knee_height > avg_hip_height * 1.2:
                print("Possible anomaly: Squatting")
                return True  

            # **Check for Hand-to-Hand Proximity (Possible Fighting, Theft)**
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            if wrist_distance < shoulder_distance * 0.5:
                print("Possible anomaly: Hand-to-Hand Proximity")
                return True  

    return False  # No anomaly detected

# **Real-time Detection Loop**
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(" Error: Could not read frame.")
            break

    	# **Run YOLO Pose Detection**
        results = model(frame, conf=DETECTION_THRESHOLD)

    	# **Detect anomaly based on keypoints**
        is_anomaly = detect_anomaly(results)

    	# **Display "Normal" or "Anomaly Detected"**
	#status_text = "Anomaly Detected!" if is_anomaly else "Normal"
	#color = (0, 0, 255) if is_anomaly else (0, 255, 0)

    	# **Overlay text on frame**
	#cv2.putText(frame, status_text, (10, 50),
        	#cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    	# **Draw Keypoints**
	#annotated_frame = results[0].plot()

    	# **Show the frame**
	#cv2.imshow("Real-time Anomaly Detection", annotated_frame)

    	#if cv2.waitKey(1) & 0xFF == ord("q"):
        	#break

except KeyboardInterrupt:
        print("Ctrl-C detected. Stopping...")
finally:
        cap.release()
	#cv2.destroyAllWindows()
