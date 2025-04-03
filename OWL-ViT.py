import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import cv2

# Load the model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define custom bad behaviors
bad_behaviors = ["fighting", "spitting", "defecating in a lift", "vandalizing"]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    inputs = processor(images=frame, text=bad_behaviors, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Draw bounding boxes
    scores = outputs["logits"].softmax(-1)[0, :, 1]
    threshold = 0.5  # Only show high-confidence detections

    for i, score in enumerate(scores):
        if score > threshold:
            print(f"Detected {bad_behaviors[i]} with confidence {score.item()}!")

    # Show video feed
    cv2.imshow("Lift Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
