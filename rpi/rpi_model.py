import torch
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import time
import threading
from datetime import datetime
import collections

model = None
_camera_running = False

# Rolling buffer for last 50 frames (pre-anomaly)
FRAME_BUFFER_SIZE = 50
frame_buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)

# Post-anomaly frame logic
collecting_post_frames = False
post_frames_target = 50
post_frames_count = 0
post_anomaly_frames = []

# Store the completed "before + after" frames here once ready
combined_anomaly_clip = None

# anomaly stat tracking
start_time = None
inference_times = []
frame_count = 0
anomaly_count = 0
ANOMALY_COOLDOWN = 10
last_anomaly_time = 0

# global variables to store anomaly state
_anomaly_detected = False
_anomaly_type = None
_anomaly_frame = None
_anomaly_confidence = 0.0

def load_model():
    global model
    # Load YOLO model
    model = YOLO("yolov8n-pose.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"YOLO model loaded on {device}.")

def detect_anomaly(results):
    """Detect anomalies based on keypoint positions and return (is_anomaly, anomaly_type, confidence)."""
    
    anomaly_type = None
    confidence = 0.0
    
    # Check if YOLO found any keypoints
    if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        return False, anomaly_type, confidence
    
    # Overall detection confidence (max among all bounding boxes)
    if len(results[0].boxes.conf) > 0:
        confidence = float(results[0].boxes.conf.max().cpu().numpy())
    
    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:  # Not enough keypoints
                continue
                
            # Extract relevant keypoints
            left_wrist, right_wrist = pose[9][:2].cpu().numpy(), pose[10][:2].cpu().numpy()
            left_shoulder, right_shoulder = pose[5][:2].cpu().numpy(), pose[6][:2].cpu().numpy()
            left_knee, right_knee = pose[13][:2].cpu().numpy(), pose[14][:2].cpu().numpy()
            left_hip, right_hip = pose[11][:2].cpu().numpy(), pose[12][:2].cpu().numpy()
            
            # Check for "Squatting"
            avg_knee_height = (left_knee[1] + right_knee[1]) / 2
            avg_hip_height = (left_hip[1] + right_hip[1]) / 2
            if avg_knee_height > avg_hip_height * 1.2:
                anomaly_type = "Squatting (possible urination or sitting)"
                return True, anomaly_type, confidence
                
            # Check for "Hand-to-Hand Proximity"
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            if wrist_distance < shoulder_distance * 0.5:
                anomaly_type = "Hand-to-Hand Proximity (possible fighting/theft)"
                return True, anomaly_type, confidence
                
    return False, anomaly_type, confidence

async def run_inference_loop():
    """
    Captures frames from the camera, runs YOLO inference,
    maintains a rolling buffer of 50 "pre-anomaly" frames,
    collects 50 "post-anomaly" frames after detection,
    and stores the combined clip for retrieval.
    """
    global _camera_running
    global collecting_post_frames, post_frames_count, post_anomaly_frames
    global combined_anomaly_clip
    global frame_count, anomaly_count, last_anomaly_time, start_time, inference_times
    global _anomaly_detected, _anomaly_type, _anomaly_frame, _anomaly_confidence

    if start_time is None:
        start_time = time.time()
    
    cap = cv2.VideoCapture(0)
    
    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            await asyncio.sleep(1)
            continue
        
        # Store the current frame in rolling buffer
        frame_buffer.append(frame.copy())
        
        # If collecting post-anomaly frames, add them
        if collecting_post_frames:
            post_anomaly_frames.append(frame.copy())
            post_frames_count += 1
            
            if post_frames_count >= post_frames_target:
                # We have the post-anomaly frames we need
                collecting_post_frames = False
                post_frames_count = 0
                
                # Combine the rolling buffer + post-anomaly frames
                all_frames = list(frame_buffer) + post_anomaly_frames
                print(f"Collected {len(all_frames)} frames (50 pre + 50 post).")

                # Store them in the global for retrieval
                combined_anomaly_clip = all_frames

                # Reset post_anomaly_frames for next time
                post_anomaly_frames = []
        
        # Run YOLO inference
        inference_start = time.time()
        results = model(frame)
        inf_time = time.time() - inference_start

        frame_count += 1
        inference_times.append(inf_time)
        
        # Check for anomaly
        _anomaly_detected, _anomaly_type, _anomaly_confidence = detect_anomaly(results)
        
        if _anomaly_detected:
            current_time = time.time()
            anomaly_count += 1
            if (current_time - last_anomaly_time) > ANOMALY_COOLDOWN:
                last_anomaly_time = current_time
                print(f"ANOMALY DETECTED: {_anomaly_type} (conf={_anomaly_confidence:.2f})")
                _anomaly_frame = frame.copy()
                
                # Start collecting the next 50 frames
                collecting_post_frames = True
                post_frames_count = 0
                post_anomaly_frames = []
        
        await asyncio.sleep(0.05)
    
    cap.release()
    print("Camera loop ended.")

def start_inference():
    """Sets _camera_running = True, starts async camera/inference in a thread."""
    global _camera_running
    _camera_running = True
    t = threading.Thread(target=lambda: asyncio.run(run_inference_loop()))
    t.start()

def stop_inference():
    """Stops camera inference loop."""
    global _camera_running
    _camera_running = False

def get_anomaly_detected():
    """Return True if an anomaly was detected since last check."""
    return _anomaly_detected

def get_anomaly_info():
    """
    Returns (anomaly_type, anomaly_frame).
    The frame is a raw OpenCV image (NumPy array).
    """
    return _anomaly_type, _anomaly_frame, _anomaly_confidence

def clear_anomaly():
    """Reset the anomaly flags so we don't trigger repeatedly."""
    global _anomaly_detected, _anomaly_type, _anomaly_frame
    _anomaly_detected = False
    _anomaly_type = None
    _anomaly_frame = None
    _anomaly_confidence = 0.0

def get_anomaly_clip():
    """
    Returns the most recent 50 pre + 50 post frames
    if an anomaly was detected and the clip was stored.
    Then clears it so it won't be returned again.
    If no clip is available, returns None.
    """
    global combined_anomaly_clip
    frames = combined_anomaly_clip
    
    # Clear it to avoid re-sending the same clip
    combined_anomaly_clip = None
    
    return frames

def get_inference_stats():
    """
    Return a final stats string or a data structure
    that main.py can log or send to Telegram.
    """
    global start_time, frame_count, inference_times, anomaly_count
    
    if start_time is None or frame_count == 0:
        return "No inference data collected yet."
    
    elapsed_time = time.time() - start_time
    avg_inference = sum(inference_times) / len(inference_times)
    fps = frame_count / elapsed_time if elapsed_time else 0.0
    
    stats = (
        f"📊 Performance Stats:\n"
        f"Runtime: {elapsed_time:.2f} seconds\n"
        f"Frames processed: {frame_count}\n"
        f"FPS: {fps:.2f}\n"
        f"Avg inference time: {avg_inference*1000:.2f} ms\n"
        f"Anomalies detected: {anomaly_count}"
    )
    return stats