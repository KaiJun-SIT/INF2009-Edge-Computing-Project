import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from slowfast_model.model import resnet50  

class SlowFastPoseLSTM(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden=512, lstm_layers=1, freeze_slowfast=True, keypoints_dim=34):  # 17 keypoints * 2 (x, y)
        super(SlowFastPoseLSTM, self).__init__()
        self.num_classes = num_classes
        self.keypoints_dim = keypoints_dim

        # Load SlowFast backbone
        self.slowfast = resnet50(class_num=num_classes)
        self.slowfast.fc = nn.Identity()  # Remove final classification layer

        if freeze_slowfast:
            for param in self.slowfast.parameters():
                param.requires_grad = False

        # LSTM input: SlowFast feature + keypoints feature
        self.lstm = nn.LSTM(
            input_size=2304 + keypoints_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x, keypoints_seq):  # x: [B, C, T, H, W], keypoints_seq: [B, T, keypoints_dim]
        batch_size, C, T, H, W = x.shape

        # SlowFast visual features
        feat = self.slowfast(x)  # [B, 2304]

        # Repeat features over time dimension
        feat_seq = feat.unsqueeze(1).repeat(1, T, 1)  # [B, T, 2304]

        # Concatenate keypoints
        combined_seq = torch.cat([feat_seq, keypoints_seq], dim=-1)  # [B, T, 2304 + keypoints_dim]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(combined_seq)
        last_time_step = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # Final classification
        output = self.fc(last_time_step)  # [B, num_classes]
        return output


MODEL_PATH = 'slowfast_model/slowfast_pose_lstm_trained2.pth'
YOLO_POSE_MODEL = 'yolov8n-pose.pt'
SEQ_LENGTH = 16
FRAME_SIZE = (128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRED_HISTORY = 10

# Class labels
class_labels = ["Fighting", "NormalVideos", "Stealing", "Vandalism"]
NUM_CLASSES = 4
KEYPOINTS_DIM = 34  # Assuming 17 keypoints (x, y)


print("🔹 Loading trained SlowFast+Pose LSTM model...")
model = SlowFastPoseLSTM(num_classes=NUM_CLASSES, lstm_hidden=512, lstm_layers=1, freeze_slowfast=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("🔹 Loading YOLOv8n-Pose model...")
yolo_pose_model = YOLO(YOLO_POSE_MODEL)
print("Models loaded successfully.")

transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)
frame_sequence = []
keypoints_sequence = []
predictions_queue = deque(maxlen=PRED_HISTORY)

print("Webcam ready. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_img)
    frame_sequence.append(transformed_frame)

    keypoints = np.zeros((KEYPOINTS_DIM,))
    yolo_results = yolo_pose_model(frame)[0]  # Inference

    if yolo_results.keypoints is not None:
        kp = yolo_results.keypoints.xy[0].cpu().numpy()  # First person (assumption)
        kp = kp.flatten()  # Flatten (17 x, y) -> 34
        if len(kp) < KEYPOINTS_DIM:
            kp = np.pad(kp, (0, KEYPOINTS_DIM - len(kp)), 'constant')  # Zero-padding if less
        keypoints = kp

        # Draw keypoints
        for x, y in kp.reshape(-1, 2):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
    keypoints_sequence.append(keypoints_tensor)

    if len(frame_sequence) > SEQ_LENGTH:
        frame_sequence.pop(0)
        keypoints_sequence.pop(0)

    if len(frame_sequence) == SEQ_LENGTH:
        input_tensor = torch.stack(frame_sequence, dim=1).unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
        keypoints_tensor_seq = torch.stack(keypoints_sequence, dim=0).unsqueeze(0).to(DEVICE)  # [1, T, keypoints_dim]

        with torch.no_grad():
            output = model(input_tensor, keypoints_tensor_seq)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            # Top prediction
            top_idx = np.argmax(probs)
            top_label = class_labels[top_idx]
            top_conf = probs[top_idx]

            # Smoothing
            predictions_queue.append(top_idx)
            smoothed_idx = int(np.round(np.mean(predictions_queue)))
            smoothed_label = class_labels[smoothed_idx]

        display_text = f"Action: {smoothed_label} ({top_conf:.2f})"
        color = (0, 255, 0) if smoothed_label == "NormalVideos" else (0, 0, 255)
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    cv2.imshow('🟢 Real-time Action Detection (SlowFast + YOLOv8n-Pose)', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam released.")
