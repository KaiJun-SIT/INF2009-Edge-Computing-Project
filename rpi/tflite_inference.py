import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov8n-pose_saved_model/yolov8n-pose_int8.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

# Define a detection threshold for a valid detection
DETECTION_THRESHOLD = 0.5

def detect_anomaly_tflite(keypoints, scores, detection_threshold=DETECTION_THRESHOLD):
    """
    Detect anomaly based on keypoints from TFLite inference.
    Expects:
      - keypoints: numpy array of shape (N, 17, 3) where each keypoint is [x, y, conf]
      - scores: numpy array of shape (N,) representing overall detection confidence per person
    """
    if keypoints.shape[0] == 0:
        print("No valid human detected. Ignoring frame.")
        return False

    for i in range(keypoints.shape[0]):
        if scores[i] < detection_threshold:
            continue

        # Each detection (pose) is assumed to be an array of shape (17, 3)
        pose = keypoints[i]
        # Extract the (x, y) coordinates from each keypoint
        left_wrist   = pose[9][:2]
        right_wrist  = pose[10][:2]
        left_shoulder  = pose[5][:2]
        right_shoulder = pose[6][:2]
        left_knee    = pose[13][:2]
        right_knee   = pose[14][:2]
        left_hip     = pose[11][:2]
        right_hip    = pose[12][:2]

        # Debug: print keypoint coordinates
        print(f"Pose {i}: Left wrist: {left_wrist}, Right wrist: {right_wrist}")

        # Check for Squatting: average knee y-coordinate should be significantly lower than hips.
        avg_knee_height = (left_knee[1] + right_knee[1]) / 2
        avg_hip_height = (left_hip[1] + right_hip[1]) / 2
        if avg_knee_height > avg_hip_height * 1.2:
            print("Possible anomaly: Squatting")
            return True

        # Check for Hand-to-Hand Proximity: wrists too close relative to shoulder distance.
        wrist_distance = np.linalg.norm(left_wrist - right_wrist)
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        if wrist_distance < shoulder_distance * 0.5:
            print("Possible anomaly: Hand-to-Hand Proximity")
            return True

    return False

# Initialize video capture
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame: resize to model input size, normalize, and add batch dimension.
        input_shape = input_details[0]['shape']  # typically [1, height, width, channels]
        model_height, model_width = input_shape[1], input_shape[2]
        resized_frame = cv2.resize(frame, (model_width, model_height))
        input_data = np.expand_dims(resized_frame, axis=0)
        input_data = (input_data / 255.0).astype(np.float32)

        # Debug: print original and resized frame sizes
        print(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"Resized frame size (model input): {model_width}x{model_height}")

        # Set the input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get and inspect the output tensor.
        output = interpreter.get_tensor(output_details[0]['index'])
        print("Raw output shape:", output.shape)

        # Assume output shape is [1, N, 56] where 56 = 5 (bbox + conf) + 17*3 (keypoints)
        # If the output shape differs (e.g. [1, 56, 8400]), add debugging and adjust accordingly.
        if output.shape[1] == 56:
            detections = output.reshape(-1, 56)
        else:
            # For example, if shape is [1, 56, 8400]:
            total_elements = output.shape[2]
            num_detections = total_elements // 56
            print("Inferred number of detections per row:", num_detections)
            detections = output.reshape(-1, 56)  # Adjust this as needed for your model

        print("Detections shape after reshaping:", detections.shape)

        valid_keypoints = []
        valid_scores = []

        # Loop over each detection and decode the values.
        for detection in detections:
            # Debug: print first 10 values of the detection vector
            print("Detection vector (first 10 values):", detection[:10])
            
            # Assume bounding box and detection confidence are in the first 5 values.
            # Here we take the detection confidence from index 4.
            score = detection[4]
            if score < DETECTION_THRESHOLD:
                continue

            # Assume keypoints start at index 5 and span 17*3 values (i.e. 51 values).
            K = 5  # Offset: first 5 values are bbox (cx, cy, w, h, conf)
            kp_flat = detection[K:K + 17*3]
            print("Raw keypoints slice:", kp_flat)
            if kp_flat.size != 17 * 3:
                print(f"Warning: Unexpected keypoints size. Expected {17*3}, got {kp_flat.size}")
                continue
            keypoints = kp_flat.reshape(17, 3)
            print("Parsed keypoints:", keypoints)
            
            valid_keypoints.append(keypoints)
            valid_scores.append(score)

        print(f"Total valid detections: {len(valid_keypoints)}")

        anomaly_found = False
        for kp, score in zip(valid_keypoints, valid_scores):
            print(f"Processing detection with score: {score}")
            print("Keypoints before anomaly check:", kp)
            kp_expanded = np.expand_dims(kp, axis=0)  # shape becomes (1, 17, 3)
            sc = np.array([score])
            if detect_anomaly_tflite(kp_expanded, sc):
                anomaly_found = True
                break

        if anomaly_found:
            print("Anomaly Detected!")
        else:
            print("Normal")
        
        # Optional: Show the frame with status overlay (for visual debugging)
        status_text = "Anomaly Detected!" if anomaly_found else "Normal"
        color = (0, 0, 255) if anomaly_found else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Real-time Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Ctrl-C detected. Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
