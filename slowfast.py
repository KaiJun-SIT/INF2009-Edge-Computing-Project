import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorchvideo.models.hub import slowfast_r50

# Load a pre-trained SlowFast model
model = slowfast_r50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define preprocessing for the frames
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),  # Normalize
])

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Buffer to store frames for SlowFast input
slow_buffer = []  # Slow pathway: low frame rate (e.g., 4 frames)
fast_buffer = []  # Fast pathway: high frame rate (e.g., 32 frames)
slow_stride = 8  # Sample 1 frame every 8 frames for the slow pathway
buffer_size = 32  # Number of frames for the fast pathway

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to a PIL image and preprocess it
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocessed_frame = preprocess(pil_image)

    # Add the preprocessed frame to the fast buffer
    fast_buffer.append(preprocessed_frame)

    # Add frames to the slow buffer based on the stride
    if len(fast_buffer) % slow_stride == 0:
        slow_buffer.append(preprocessed_frame)

    # When the fast buffer is full, run inference
    if len(fast_buffer) == buffer_size:
        # Ensure the slow buffer has exactly 4 frames
        if len(slow_buffer) != 4:
            # If not, sample 4 frames evenly from the fast buffer
            slow_buffer = [fast_buffer[i] for i in range(0, buffer_size, buffer_size // 4)][:4]

        # Convert the buffers to tensors
        slow_clip = torch.stack(slow_buffer)  # Shape: (4, 3, 224, 224)
        fast_clip = torch.stack(fast_buffer)  # Shape: (32, 3, 224, 224)

        # Transpose the tensors to match SlowFast input shape: (batch_size, channels, frames, height, width)
        slow_clip = slow_clip.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, 3, 4, 224, 224)
        fast_clip = fast_clip.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, 3, 32, 224, 224)

        # Prepare the input as a list of tensors for the slow and fast pathways
        input_clip = [slow_clip, fast_clip]

        # Run inference
        with torch.no_grad():
            outputs = model(input_clip)

        # Get the predicted action (example: assume outputs are logits)
        predicted_action = torch.argmax(outputs, dim=1).item()
        print(f"Predicted Action: {predicted_action}")

        # Clear the buffers for the next clip
        slow_buffer = []
        fast_buffer = []

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()