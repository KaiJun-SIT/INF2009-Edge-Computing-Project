import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera (C270)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()