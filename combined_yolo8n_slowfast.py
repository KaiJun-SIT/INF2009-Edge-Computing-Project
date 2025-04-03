

import cv2
import torch
from torchvision import transforms
from collections import deque
import numpy as np
from ultralytics import YOLO
from PIL import Image
from slowfast_model.model import resnet50 

# ---- Load Models ---- #
# Load YOLOv8n
yolo_model = YOLO('yolo8n_model/UCF-YOLO-Training/YOLOv8n-finetune10/weights/best.pt')

# Load SlowFast (Anomaly)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
slowfast_model = resnet50(class_num=2)
slowfast_model.load_state_dict(torch.load('slowfast_model/slowfast_video_trained.pth', map_location=DEVICE))
slowfast_model.to(DEVICE).eval()

print("✅ Both models loaded successfully!")

# ---- Configuration ---- #
SEQ_LENGTH = 16
PRED_HISTORY = 10
FRAME_SIZE = (128, 128)
frame_buffer = []
predictions_queue = deque(maxlen=PRED_HISTORY)

transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---- Webcam Setup ---- #
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # --- SlowFast Processing (Temporal) --- #
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_img)
    frame_buffer.append(transformed_frame)

    if len(frame_buffer) > SEQ_LENGTH:
        frame_buffer.pop(0)

    anomaly_label = "Normal"
    anomaly_color = (0, 255, 0)
    smoothed_pred = 0  # Default to Normal

    if len(frame_buffer) == SEQ_LENGTH:
        input_tensor = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = slowfast_model(input_tensor)
            anomaly_prob = torch.softmax(output, dim=1)[0][1].item()
            prediction = 1 if anomaly_prob > 0.7 else 0
            predictions_queue.append(prediction)
            smoothed_pred = 1 if np.mean(predictions_queue) > 0.5 else 0

        if smoothed_pred == 1:
            anomaly_label = "Anomaly Detected!"
            anomaly_color = (0, 0, 255)

        # Overlay Anomaly
        cv2.putText(frame, f"SlowFast: {anomaly_label} ({anomaly_prob:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, anomaly_color, 2, cv2.LINE_AA)

    # --- YOLO Processing (Spatial) --- #
    class_names = [
    "Abuse", "Arrest", "Arson", "Assault", "Explosion",
    "Fighting", "NormalVideos", "Robbery", "Shooting", "Stealing", "Vandalism"
    ]
    results = yolo_model(frame)
    crimes_detected = []
    result = results[0]  # First (and only) result in list
    # Extract parts
    boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
    confs = result.boxes.conf.cpu().numpy()  # confidence
    classes = result.boxes.cls.cpu().numpy()  # class

    # Loop over detections
    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        class_id = int(cls)
        label = f"{class_names[class_id]} ({conf:.2f})"

        # Draw detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # --- Fusion Logic --- #
    if smoothed_pred == 1 and crimes_detected:
        cv2.putText(frame, f"🚨 Crime: {', '.join(crimes_detected)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elif smoothed_pred == 1 and not crimes_detected:
        cv2.putText(frame, "⚠️ Anomaly Detected but No Specific Crime", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    elif smoothed_pred == 0 and crimes_detected:
        cv2.putText(frame, f"✅ Detected (but normal): {', '.join(crimes_detected)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Display --- #
    cv2.imshow('Live Crime Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam released. Combined detection ended.")
