from collections import deque
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from model import resnet50  # Your SlowFast model

# --------------------- CONFIG --------------------- #
MODEL_PATH = 'slowfast_video_trained.pth'
SEQ_LENGTH = 16
FRAME_SIZE = (128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRED_HISTORY = 10  # Smoothing over last 10 predictions

# Classes (index to label mapping)
class_labels = ["Explosion", "Fighting", "NormalVideos", "Stealing", "Vandalism"]
NUM_CLASSES = len(class_labels)

# --------------------- Load Model --------------------- #
print("🔹 Loading SlowFast model...")
model = resnet50(class_num=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded with {} classes.".format(NUM_CLASSES))

# --------------------- Frame Preprocessing --------------------- #
transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --------------------- Webcam Init --------------------- #
cap = cv2.VideoCapture(0)  # Webcam input
frame_sequence = []
predictions_queue = deque(maxlen=PRED_HISTORY)

print("✅ Webcam initialized. Press 'q' to quit.")

# --------------------- Main Loop --------------------- #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Frame transform for SlowFast
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_img)
    frame_sequence.append(transformed_frame)

    # Keep only SEQ_LENGTH frames
    if len(frame_sequence) > SEQ_LENGTH:
        frame_sequence.pop(0)

    # Perform prediction when enough frames are collected
    if len(frame_sequence) == SEQ_LENGTH:
        input_tensor = torch.stack(frame_sequence, dim=1).unsqueeze(0).to(DEVICE)  # Shape: [1, C, T, H, W]
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # Class probabilities

            # Find top predicted class
            top_idx = np.argmax(probs)
            top_label = class_labels[top_idx]
            top_conf = probs[top_idx]

            # Append prediction for smoothing
            predictions_queue.append(top_idx)
            smoothed_idx = int(np.round(np.mean(predictions_queue)))  # Majority vote (approximate)
            smoothed_label = class_labels[smoothed_idx]

        # -------- Display Prediction -------- #
        display_text = f"Detected: {smoothed_label} ({top_conf:.2f})"
        color = (0, 0, 255) if smoothed_label != "NormalVideos" else (0, 255, 0)  # Red if crime/anomaly

        # Display on screen
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    # -------- Show Frame -------- #
    cv2.imshow('Real-time Crime Detection (SlowFast)', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# --------------------- Cleanup --------------------- #
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam released.")
