from collections import deque
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from model import resnet50  # SlowFast definition

MODEL_PATH = 'slowfast_crime_detection.pth'
SEQ_LENGTH = 16
FRAME_SIZE = (128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRED_HISTORY = 10  # Smooth over last 10 predictions

print("Loading model...")
model = resnet50(class_num=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)
frame_sequence = []
predictions_queue = deque(maxlen=PRED_HISTORY)

print("i hate this")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Frame transform
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_img)
    frame_sequence.append(transformed_frame)

    # Maintain SEQ_LENGTH
    if len(frame_sequence) > SEQ_LENGTH:
        frame_sequence.pop(0)

    # Predict when enough frames
    if len(frame_sequence) == SEQ_LENGTH:
        input_tensor = torch.stack(frame_sequence, dim=1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            anomaly_prob = torch.softmax(output, dim=1)[0][1].item()  # Anomaly probability

            prediction = 1 if anomaly_prob > 0.7 else 0  
            predictions_queue.append(prediction)

            # Smooth final prediction
            smoothed_pred = 1 if np.mean(predictions_queue) > 0.5 else 0
            label = "Anomaly/Crime" if smoothed_pred == 1 else "Normal"
            color = (0, 0, 255) if smoothed_pred == 1 else (0, 255, 0)

        # Overlay text (without bounding box)
        cv2.putText(frame, f"Prediction: {label} ({anomaly_prob:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show frame
    try:
        cv2.imshow('Crime Detection', frame)
    except:
        pass

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("bye bitch")
        break

# Cleanup
cap.release()
# cv2.destroyAllWindows()
print("Webcam released.")
