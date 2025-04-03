import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load a pre-trained I3D model (example using PyTorch Hub)
# Note: Replace with an actual I3D implementation if not available on PyTorch Hub
model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define preprocessing for the frames
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Buffer to store frames for I3D input
frame_buffer = []
buffer_size = 16  # I3D typically uses 16-frame clips

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to a PIL image and preprocess it
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocessed_frame = preprocess(pil_image)

    # Add the preprocessed frame to the buffer
    frame_buffer.append(preprocessed_frame)

    # When the buffer is full, run inference
    if len(frame_buffer) == buffer_size:
        # Convert the buffer to a tensor
        input_clip = torch.stack(frame_buffer)  # Shape: (16, 3, 224, 224)

        # Transpose the tensor to match I3D input shape: (batch_size, channels, frames, height, width)
        input_clip = input_clip.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, 3, 16, 224, 224)

        # Run inference
        with torch.no_grad():
            outputs = model(input_clip)

        # Get the predicted action (example: assume outputs are logits)
        predicted_action = torch.argmax(outputs, dim=1).item()
        print(f"Predicted Action: {predicted_action}")

        # Clear the buffer for the next clip
        frame_buffer = []

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()