from collections import deque
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import tensorflow as tf

# Path to your TFLite model (update if needed)
TFLITE_MODEL_PATH = "model.tflite"

# Configuration parameters
SEQ_LENGTH = 16
FRAME_SIZE = (128, 128)
PRED_HISTORY = 10  # Smooth over last 10 predictions

# Load TFLite model and allocate tensors
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)
frame_sequence = []
predictions_queue = deque(maxlen=PRED_HISTORY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to PIL image and apply transforms
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_frame = transform(pil_img)
    frame_sequence.append(transformed_frame)

    # Maintain fixed sequence length
    if len(frame_sequence) > SEQ_LENGTH:
        frame_sequence.pop(0)

    # Run inference when enough frames are gathered
    if len(frame_sequence) == SEQ_LENGTH:
        # Stack frames to shape: (1, 3, SEQ_LENGTH, 128, 128)
        # (Each transformed frame is a torch.Tensor of shape (3, 128, 128))
        input_tensor = np.stack([frame.numpy() for frame in frame_sequence], axis=1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        # Set the TFLite model input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compute softmax manually on the output logits
        exp_scores = np.exp(output - np.max(output))
        softmax = exp_scores / np.sum(exp_scores)
        anomaly_prob = softmax[0][1]  # Assuming index 1 is the anomaly class

        # Determine prediction based on a threshold
        prediction = 1 if anomaly_prob > 0.7 else 0  
        predictions_queue.append(prediction)
        smoothed_pred = 1 if np.mean(predictions_queue) > 0.5 else 0
        label = "Anomaly/Crime" if smoothed_pred == 1 else "Normal"
        color = (0, 0, 255) if smoothed_pred == 1 else (0, 255, 0)

        # Overlay prediction text on the frame
        cv2.putText(frame, f"Prediction: {label} ({anomaly_prob:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the frame
    try:
        cv2.imshow('Crime Detection', frame)
    except Exception as e:
        print("Error displaying frame:", e)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam released.")
