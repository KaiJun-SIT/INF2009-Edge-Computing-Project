import torch
import cv2
import numpy as np
from ultralytics import YOLO
import telegram
import asyncio
import time
from datetime import datetime
import os
import paho.mqtt.client as mqtt

# Telegram bot configuration
bot = telegram.Bot(token = '7782063988:AAGAqTPsiVU_Vqhkkw1KvqjhrdKwONeaB0o')
anomaly_detection_channel_id = -1002589162188
slms_priority_1_channel_id = -1002531314734
slms_priority_2_channel_id = -1002500092362
slms_priority_3_channel_id = -1002647870078

# MQTT SETUP
MQTT_BROKER = "172.20.10.10" # CHANGE TO FOG MACHINE BROKER IP
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,"Trigger")
client.connect(MQTT_BROKER, 1883, 60)

# Load YOLOv8-Pose model
model = YOLO("yolov8n-pose.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"YOLOv8-Pose model loaded successfully on {device}")

# Configuration parameters
DETECTION_THRESHOLD = 0.5  # Confidence threshold for human detection
ANOMALY_COOLDOWN = 30  # Seconds between anomaly notifications to avoid spam
INFERENCE_LOG_INTERVAL = 60  # Log inference stats every X seconds
SAVE_ANOMALY_FRAMES = True  # Save frames with anomalies

# Stats tracking
last_anomaly_time = 0
inference_times = []
start_time = time.time()
frame_count = 0
anomaly_count = 0

# Create output directory for saved anomaly frames
if SAVE_ANOMALY_FRAMES:
    os.makedirs("anomaly_frames", exist_ok=True)

# Telegram bot functions
async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)

async def send_document(document, chat_id):
    async with bot:
        await bot.send_document(document=document, chat_id=chat_id)

async def send_photo(photo, chat_id, caption=None):
    async with bot:
        await bot.send_photo(photo=photo, chat_id=chat_id, caption=caption)

async def send_video(video, chat_id, caption=None):
    async with bot:
        await bot.send_video(video=video, chat_id=chat_id, caption=caption)

# Improved anomaly detection function with details
def detect_anomaly(results):
    """Detects anomaly based on keypoint positions and returns the type of anomaly."""
    
    anomaly_type = None
    confidence = 0.0
    
    # Ensure YOLO detected humans
    if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        print("No valid human detected. Ignoring frame.")
        return False, anomaly_type, confidence
    
    # Get overall detection confidence
    if len(results[0].boxes.conf) > 0:
        confidence = float(results[0].boxes.conf.max().cpu().numpy())
    
    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:  # Ensure enough keypoints
                continue
                
            # Extract keypoints (x, y) positions
            left_wrist, right_wrist = pose[9][:2].cpu().numpy(), pose[10][:2].cpu().numpy()
            left_shoulder, right_shoulder = pose[5][:2].cpu().numpy(), pose[6][:2].cpu().numpy()
            left_knee, right_knee = pose[13][:2].cpu().numpy(), pose[14][:2].cpu().numpy()
            left_hip, right_hip = pose[11][:2].cpu().numpy(), pose[12][:2].cpu().numpy()
            
            # Check for Squatting 
            avg_knee_height = (left_knee[1] + right_knee[1]) / 2
            avg_hip_height = (left_hip[1] + right_hip[1]) / 2
            if avg_knee_height > avg_hip_height * 1.2:
                anomaly_type = "Squatting (possible urination or sitting in restricted area)"
                return True, anomaly_type, confidence
                
            # Check for Hand-to-Hand Proximity
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            if wrist_distance < shoulder_distance * 0.5:
                anomaly_type = "Hand-to-Hand Proximity (possible fighting or theft)"
                return True, anomaly_type, confidence
                
    return False, anomaly_type, confidence

# Function to log inference statistics
def log_inference_stats():
    global inference_times, frame_count, start_time, anomaly_count
    
    if not inference_times:
        return "No inference data available yet."
    
    elapsed_time = time.time() - start_time
    avg_inference = sum(inference_times) / len(inference_times)
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    stats = (
        f"📊 Performance Stats:\n"
        f"Runtime: {elapsed_time:.2f} seconds\n"
        f"Frames processed: {frame_count}\n"
        f"FPS: {fps:.2f}\n"
        f"Avg inference time: {avg_inference*1000:.2f}ms\n"
        f"Anomalies detected: {anomaly_count}\n"
        f"Device: {device}"
    )
    
    # Reset for next interval
    inference_times = []
    
    return stats

# Main detection loop
async def main():
    global last_anomaly_time, inference_times, frame_count, anomaly_count
    
    # Send startup notification
    startup_msg = f"🚀 Anomaly Detection System Started\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💻 Device: {device}"
    await send_message(startup_msg, anomaly_detection_channel_id)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    last_stats_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                await send_message("⚠️ Camera error: Could not read frame", anomaly_detection_channel_id)
                break
                
            # Measure inference time
            inference_start = time.time()
            
            # Run YOLO Pose Detection
            results = model(frame, conf=DETECTION_THRESHOLD)
            
            # Record inference time
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            frame_count += 1
            
            # Detect anomaly
            is_anomaly, anomaly_type, confidence = detect_anomaly(results)
            
            # Handle anomaly detection with cooldown
            current_time = time.time()
            if is_anomaly and (current_time - last_anomaly_time) > ANOMALY_COOLDOWN:
                last_anomaly_time = current_time
                anomaly_count += 1
                
                # Create annotated frame
                annotated_frame = results[0].plot()
                
                # Save the frame with timestamp
                if SAVE_ANOMALY_FRAMES:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_path = f"anomaly_frames/anomaly_{timestamp}.jpg"
                    cv2.imwrite(frame_path, annotated_frame)
                
                # Prepare and send alert message
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert_msg = (
                    f"🚨 ANOMALY DETECTED!\n"
                    f"⏰ Time: {timestamp}\n"
                    f"🔍 Type: {anomaly_type}\n"
                    f"📏 Confidence: {confidence:.2f}\n"
                    f"⚡ Inference time: {inference_time*1000:.2f}ms"
                )
                
                target_channel = anomaly_detection_channel_id
                
                # TRIGGER VIDEO STREAM TO FOG
                client.publish(MQTT_TOPIC, "start")
                
                
                # Send the alert with the image
                await send_photo(open(frame_path, 'rb'), target_channel, caption=alert_msg)
                print(f"Anomaly alert sent to channel: {target_channel}")
            
            # Periodically log inference statistics
            if current_time - last_stats_time > INFERENCE_LOG_INTERVAL:
                stats_msg = log_inference_stats()
                await send_message(stats_msg, anomaly_detection_channel_id)
                last_stats_time = current_time
                
            # Optional: Display frame locally
            # cv2.imshow("Real-time Anomaly Detection", results[0].plot())
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
                
    except KeyboardInterrupt:
        print("Ctrl-C detected. Stopping...")
        await send_message("🛑 System stopped by user", anomaly_detection_channel_id)
    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}"
        print(error_msg)
        await send_message(error_msg, anomaly_detection_channel_id)
    finally:
        # Send final stats before closing
        final_stats = log_inference_stats()
        await send_message(f"📉 System Shutting Down\n{final_stats}", anomaly_detection_channel_id)
        
        cap.release()
        # cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
