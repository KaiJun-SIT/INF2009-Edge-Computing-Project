import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

# Load the pretrained model and processor
model_name = "Falconsai/nsfw_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image and run inference
    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_label]

    # Display the predicted label on the frame
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()