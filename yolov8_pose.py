import cv2
from ultralytics import YOLO

# Load the YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")  # Use "yolov8n-pose.pt" for the nano model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run pose estimation on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Get the annotated frame with keypoints and skeletons

    # Display the annotated frame
    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()