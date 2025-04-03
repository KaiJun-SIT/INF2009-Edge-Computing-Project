import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time
import threading
import torch
import os
import asyncio
import telegram
from datetime import datetime
import base64
from PIL import Image
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms
from collections import deque
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fog_telegram.log"),
        logging.StreamHandler()
    ]
)

# Telegram bot configuration
BOT_TOKEN = '7249438415:AAFd9xiNR33qPMUJIfcz91xWuc0eYGMYFts'
ANOMALY_DETECTION_CHANNEL_ID = -1002589162188
SLMS_PRIORITY_1_CHANNEL_ID = -1002531314734  # Fighting
SLMS_PRIORITY_2_CHANNEL_ID = -1002500092362  # Stealing
SLMS_PRIORITY_3_CHANNEL_ID = -1002647870078  # Vandalism

# MQTT settings
MQTT_BROKER = "172.20.10.10"
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"
INFERENCE_TOPIC = "camera/inference"

# Frame storage settings
SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model and detection settings
MODEL_PATH = 'slowfast_model/slowfast_pose_lstm_trained6.pth'
YOLO_POSE_MODEL = 'yolov8n-pose.pt'
SEQ_LENGTH = 16
FRAME_SIZE = (128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRED_HISTORY = 10

# Class labels
class_labels = ["Fighting", "NormalVideos", "Stealing", "Vandalism"]
# Map class labels to priority channels
channel_mapping = {
    "Fighting": SLMS_PRIORITY_1_CHANNEL_ID,
    "Stealing": SLMS_PRIORITY_2_CHANNEL_ID,
    "Vandalism": SLMS_PRIORITY_3_CHANNEL_ID,
    "NormalVideos": None  # No specific channel for normal videos
}

# Time interval between Telegram messages (to avoid spam)
MIN_TELEGRAM_INTERVAL = 30  # seconds
last_telegram_time = {
    "Fighting": 0,
    "Stealing": 0,
    "Vandalism": 0,
    "NormalVideos": 0
}

# Received frame buffer and playback control
frame_buffer = {}
total_frames = 0
fps = 15
playback_active = False
playback_thread = None
analysis_active = False
analysis_thread = None

# Initialize Telegram bot with connection pool settings
# Configure connection pool with larger size and longer timeout
from telegram.request import HTTPXRequest
request = HTTPXRequest(connection_pool_size=8, pool_timeout=30.0)
bot = telegram.Bot(token=BOT_TOKEN, request=request)

# SlowFast+Pose LSTM model definition
class SlowFastPoseLSTM(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden=512, lstm_layers=1, freeze_slowfast=True, keypoints_dim=34):
        super(SlowFastPoseLSTM, self).__init__()
        self.num_classes = num_classes
        self.keypoints_dim = keypoints_dim

        # Import dynamically to avoid import errors if not needed immediately
        from slowfast_model.model import resnet50
        
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

    def forward(self, x, keypoints_seq):
        # SlowFast visual features from full video clip
        feat = self.slowfast(x)  # [B, 2304]

        # Repeat feature for each timestep
        B, T, D = keypoints_seq.shape
        feat_seq = feat.unsqueeze(1).repeat(1, T, 1)  # [B, T, 2304]

        # Concatenate keypoints with features
        combined_seq = torch.cat([feat_seq, keypoints_seq], dim=-1)  # [B, T, 2304 + keypoints_dim]

        # Temporal modeling
        lstm_out, _ = self.lstm(combined_seq)
        last_time_step = lstm_out[:, -1, :]  # [B, lstm_hidden]

        return self.fc(last_time_step)  # [B, num_classes]

# Load models
# print("🔹 Loading models...")
try:
    # Load YOLO model
    yolo_pose_model = YOLO(YOLO_POSE_MODEL)
    
    # Load SlowFast+Pose LSTM model
    model = SlowFastPoseLSTM(num_classes=len(class_labels), lstm_hidden=512, lstm_layers=1, freeze_slowfast=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # Continue without crashing, we'll handle model absence in the analysis function

# Transform for video frames
transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Connect with MQTT
try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FogMachineReceiver")
except:
    # Fallback for older Paho MQTT versions
    client = mqtt.Client("FogMachineReceiver")

# Telegram utilities
async def send_telegram_message(chat_id, text):
    """Send a text message to specified Telegram channel"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await bot.send_message(chat_id=chat_id, text=text)
            print(f"Sent message to channel {chat_id}")
            return True
        except telegram.error.RetryAfter as e:
            # Handle rate limiting
            retry_time = e.retry_after + 1
            print(f"Rate limited. Retrying in {retry_time} seconds...")
            await asyncio.sleep(retry_time)
        except telegram.error.TimedOut:
            # Handle timeouts
            print(f"Timeout error. Retrying... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(2)
        except Exception as e:
            if "Pool timeout" in str(e):
                print(f"Connection pool timeout. Retrying after delay... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(5)  # Longer delay for pool issues
            else:
                print(f"Error sending Telegram message: {e}")
                return False
    
    print(f"Failed to send message after {max_retries} attempts")
    return False

async def send_telegram_photo(chat_id, photo_path, caption=None):
    """Send a photo to specified Telegram channel"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(photo_path, 'rb') as photo:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
            print(f"Sent photo to channel {chat_id}")
            return True
        except telegram.error.RetryAfter as e:
            # Handle rate limiting
            retry_time = e.retry_after + 1
            print(f"Rate limited. Retrying in {retry_time} seconds...")
            await asyncio.sleep(retry_time)
        except telegram.error.TimedOut:
            # Handle timeouts
            print(f"Timeout error. Retrying... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(2)
        except Exception as e:
            if "Pool timeout" in str(e):
                print(f"Connection pool timeout. Retrying after delay... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(5)  # Longer delay for pool issues
            else:
                print(f"Error sending Telegram photo: {e}")
                return False
    
    print(f"Failed to send photo after {max_retries} attempts")
    return False

async def send_telegram_video(chat_id, video_path, caption=None):
    """Send a video to specified Telegram channel"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(video_path, 'rb') as video:
                await bot.send_video(chat_id=chat_id, video=video, caption=caption)
            print(f"Sent video to channel {chat_id}")
            return True
        except telegram.error.RetryAfter as e:
            # Handle rate limiting
            retry_time = e.retry_after + 1
            print(f"Rate limited. Retrying in {retry_time} seconds...")
            await asyncio.sleep(retry_time)
        except telegram.error.TimedOut:
            # Handle timeouts
            print(f"Timeout error. Retrying... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(2)
        except Exception as e:
            if "Pool timeout" in str(e) or "All connections in the connection pool are occupied" in str(e):
                print(f"Connection pool timeout. Retrying after delay... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(5)  # Longer delay for pool issues
            else:
                print(f"Error sending Telegram video: {e}")
                return False
    
    print(f"Failed to send video after {max_retries} attempts")
    return False

def run_telegram_async(coroutine):
    """Run an async coroutine from synchronous code"""
    asyncio.run(coroutine)

def send_inference_to_telegram(inference_type, confidence, video_path=None, frame_path=None):
    """Send inference results to appropriate Telegram channels"""
    global last_telegram_time
    
    # Check cooldown to avoid spam
    current_time = time.time()
    if current_time - last_telegram_time.get(inference_type, 0) < MIN_TELEGRAM_INTERVAL:
        print(f"Telegram cooldown active for {inference_type}, skipping message")
        return
    
    # Update last message time
    last_telegram_time[inference_type] = current_time
    
    # Send to general anomaly detection channel
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚨 ANOMALY DETECTED!\n⏰ Time: {timestamp}\n🔍 Type: {inference_type}\n📏 Confidence: {confidence:.2f}"
    
    # Create a sequence of sending tasks to avoid overwhelming Telegram
    async def send_all_messages():
        # Always send to main anomaly channel first
        if frame_path and os.path.exists(frame_path):
            print(f"Sending photo to anomaly detection channel...")
            success = await send_telegram_photo(ANOMALY_DETECTION_CHANNEL_ID, frame_path, message)
            if not success:
                # Fallback to text if photo fails
                await send_telegram_message(ANOMALY_DETECTION_CHANNEL_ID, message)
        else:
            await send_telegram_message(ANOMALY_DETECTION_CHANNEL_ID, message)
        
        # Wait a bit before sending to priority channel to avoid overloading
        await asyncio.sleep(2)
        
        # If it's not a normal video, also send to the appropriate priority channel
        if inference_type != "NormalVideos" and channel_mapping.get(inference_type):
            priority_channel = channel_mapping[inference_type]
            priority_message = f"PRIORITY ALERT: {inference_type}\n Time: {timestamp}\n📏 Confidence: {confidence:.2f}"
            
            if video_path and os.path.exists(video_path):
                print(f"Sending video to priority channel {priority_channel}...")
                success = await send_telegram_video(priority_channel, video_path, priority_message)
                if not success and frame_path and os.path.exists(frame_path):
                    # If video fails, try photo
                    await send_telegram_photo(priority_channel, frame_path, priority_message)
            elif frame_path and os.path.exists(frame_path):
                await send_telegram_photo(priority_channel, frame_path, priority_message)
            else:
                await send_telegram_message(priority_channel, priority_message)
    
    # Run the async function in a separate thread
    threading.Thread(
        target=run_telegram_async,
        args=(send_all_messages(),)
    ).start()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to all relevant topics
        client.subscribe(MQTT_TOPIC + "/frame/#")
        client.subscribe(MQTT_TOPIC + "/metadata")
        client.subscribe(TRIGGER_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global frame_buffer, total_frames, fps, playback_active
    
    try:
        # Handle trigger message
        if msg.topic == TRIGGER_TOPIC:
            command = msg.payload.decode()
            print(f"Received trigger: {command}")
            
            if command.lower() == "start":
                print("Received start trigger - preparing to receive frames")
                # Reset frame buffer for new sequence
                frame_buffer.clear()
                
            elif command.lower() == "stop":
                print("Received stop trigger")
                if playback_active:
                    playback_active = False
        
        # Handle metadata
        elif msg.topic == MQTT_TOPIC + "/metadata":
            metadata = msg.payload.decode().split(',')
            total_frames = int(metadata[0])
            fps = int(metadata[1])
            print(f"Metadata received: {total_frames} frames at {fps} FPS")
            
            # Reset frame buffer for new sequence
            frame_buffer.clear()
        
        # Handle frame messages
        elif msg.topic.startswith(MQTT_TOPIC + "/frame/"):
            # Extract frame index from topic
            frame_index = int(msg.topic.split('/')[-1])
            
            # Decode base64 image
            jpg_bytes = base64.b64decode(msg.payload)
            img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Add to buffer with index as key
                frame_buffer[frame_index] = frame
                
                # Check if we have all frames or if we have at least 80% of the expected frames
                # and it's been more than 5 seconds since we got the first frame
                received_percentage = (len(frame_buffer) / total_frames) * 100 if total_frames > 0 else 0
                
                if (len(frame_buffer) == total_frames or 
                    (received_percentage >= 80 and time.time() - frame_buffer.get('first_frame_time', 0) > 5)) and not playback_active:
                    if len(frame_buffer) < total_frames:
                        print(f"Received {len(frame_buffer)}/{total_frames} frames ({received_percentage:.1f}%). Starting analysis with partial data.")
                    else:
                        print(f"All {total_frames} frames received. Starting analysis.")
                    
                    # Start analysis in a new thread
                    playback_active = True
                    analysis_thread = threading.Thread(target=analyze_frames)
                    analysis_thread.daemon = True
                    analysis_thread.start()
                
                # Record time of first frame
                if 'first_frame_time' not in frame_buffer and frame_index == 0:
                    frame_buffer['first_frame_time'] = time.time()
    
    except Exception as e:
        print(f"Error processing message: {e}")

def analyze_frames():
    """Analyze received frames with SlowFast+Pose LSTM model and send to Telegram"""
    global frame_buffer, playback_active, model, yolo_pose_model, transform
    logger = logging.getLogger('analyze_frames')
    
    try:
        print("Starting analysis of received frames")
        
        # Sort frames by index, filtering out non-frame entries like 'first_frame_time'
        frame_keys = [k for k in frame_buffer.keys() if isinstance(k, int)]
        sorted_frames = [frame_buffer[i] for i in sorted(frame_keys)]
        logger.info(f"Processing {len(sorted_frames)} frames")
        
        # Save frames to video
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = f"{SAVE_DIR}/received_{timestamp}.mp4"
        height, width = sorted_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in sorted_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved received frames to {video_path}")
        
        # Prepare for analysis
        frame_sequence = []
        keypoints_sequence = []
        predictions_queue = deque(maxlen=PRED_HISTORY)
        keyframe_path = None
        best_keyframe_conf = 0
        
        # Sample frames evenly to create a sequence for analysis
        if len(sorted_frames) > SEQ_LENGTH:
            indices = np.linspace(0, len(sorted_frames) - 1, SEQ_LENGTH, dtype=int)
            analysis_frames = [sorted_frames[i] for i in indices]
        else:
            # If we have fewer frames than needed, duplicate frames
            analysis_frames = sorted_frames
            while len(analysis_frames) < SEQ_LENGTH:
                analysis_frames.append(analysis_frames[-1])
        
        # Create output video with annotations
        analysis_path = f"{SAVE_DIR}/analysis_{timestamp}.mp4"
        analysis_writer = cv2.VideoWriter(analysis_path, fourcc, fps, (width, height))
        
        print("Extracting keypoints and processing frames...")
        
        # First pass: check if any frames have keypoints to avoid wasting time
        has_keypoints = False
        for frame in analysis_frames[:5]:  # Check first few frames
            results = yolo_pose_model(frame)[0]
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                has_keypoints = True
                break
        
        if not has_keypoints:
            print("No human keypoints detected in sample frames. Skipping detailed analysis.")
            keyframe_path = f"{SAVE_DIR}/keyframe_{timestamp}.jpg"
            cv2.imwrite(keyframe_path, sorted_frames[len(sorted_frames)//2])
            
            # Send to Telegram with "No human detected" message
            send_inference_to_telegram(
                "NormalVideos", 
                1.0,  # High confidence of normal activity
                video_path=video_path if os.path.exists(video_path) else None,
                frame_path=keyframe_path
            )
            return
        
        # Process each frame for analysis if we found keypoints
        for i, frame in enumerate(analysis_frames):
            # Extract keypoints with YOLOv8n-pose
            keypoints = np.zeros((34,))  # 17 keypoints * 2 (x, y)
            yolo_results = yolo_pose_model(frame)[0]
            
            # Process keypoints if detected
            person_detected = False
            person_confidence = 0
            
            if yolo_results.keypoints is not None and yolo_results.boxes is not None:
                if len(yolo_results.keypoints.xy) > 0 and len(yolo_results.boxes.conf) > 0:
                    keypoints_list = yolo_results.keypoints.xy
                    confidence = float(yolo_results.boxes.conf[0])
                    person_detected = True
                    person_confidence = confidence
                    
                    # Get the first person's keypoints
                    kp = keypoints_list[0].cpu().numpy()
                    kp = kp.flatten()
                    
                    if len(kp) < 34:
                        kp = np.pad(kp, (0, 34 - len(kp)), 'constant')
                    keypoints = kp
            
            # Convert frame for model input
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transformed_frame = transform(pil_img)
            frame_sequence.append(transformed_frame)
            
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
            keypoints_sequence.append(keypoints_tensor)
            
            # Draw keypoints on frame
            annotated_frame = yolo_results.plot()
            analysis_writer.write(annotated_frame)
            
            # Save best frame for Telegram notification (frame with highest person confidence)
            if person_detected and person_confidence > best_keyframe_conf:
                best_keyframe_conf = person_confidence
                keyframe_path = f"{SAVE_DIR}/keyframe_{timestamp}_{i}.jpg"
                cv2.imwrite(keyframe_path, annotated_frame)
        
        # Use middle frame if no good keyframe was found
        if keyframe_path is None:
            keyframe_path = f"{SAVE_DIR}/keyframe_{timestamp}.jpg"
            middle_idx = len(analysis_frames) // 2
            cv2.imwrite(keyframe_path, analysis_frames[middle_idx])
        
        # Run inference if we have enough frames and keypoints
        if len(frame_sequence) == SEQ_LENGTH and len(keypoints_sequence) == SEQ_LENGTH:
            # print("Running inference with SlowFast+Pose LSTM model...")
            
            input_tensor = torch.stack(frame_sequence, dim=1).unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
            keypoints_tensor_seq = torch.stack(keypoints_sequence, dim=0).unsqueeze(0).to(DEVICE)  # [1, T, keypoints_dim]
            
            with torch.no_grad():
                output = model(input_tensor, keypoints_tensor_seq)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                # Top prediction
                top_idx = np.argmax(probs)
                top_label = class_labels[top_idx]
                top_conf = probs[top_idx]
                
                # Add to prediction history for smoothing
                predictions_queue.append(top_idx)
                smoothed_idx = int(np.round(np.mean(predictions_queue)))
                smoothed_label = class_labels[smoothed_idx]
                smoothed_conf = probs[smoothed_idx]
                
                print(f"Inference result: '{smoothed_label}' (confidence: {smoothed_conf:.2f})")

                # Save analysis results
                analysis_result_path = f"{SAVE_DIR}/result_{timestamp}.txt"
                with open(analysis_result_path, 'w') as f:
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Primary classification: {smoothed_label}\n")
                    f.write(f"Confidence: {smoothed_conf:.4f}\n\n")
                    f.write("All class probabilities:\n")
                    for i, label in enumerate(class_labels):
                        f.write(f"{label}: {probs[i]:.4f}\n")
                
                # Send to appropriate Telegram channel
                send_inference_to_telegram(
                    smoothed_label, 
                    smoothed_conf, 
                    video_path=video_path if os.path.exists(video_path) else None,
                    frame_path=keyframe_path if os.path.exists(keyframe_path) else None
                )
        else:
            print("Not enough frames or keypoints for inference")
        
        # Close video writer
        analysis_writer.release()
        print(f"Analysis complete. Output saved to {analysis_path}")
        
    except Exception as e:
        print(f"Error analyzing frames: {e}")
    finally:
        playback_active = False

# Set up MQTT client
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
def start_mqtt_client():
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        
        # Send startup notification to Telegram
        startup_msg = f"🚀 Fog Machine Analysis System Started\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💻 Device: {DEVICE}"
        threading.Thread(
            target=run_telegram_async,
            args=(send_telegram_message(ANOMALY_DETECTION_CHANNEL_ID, startup_msg),)
        ).start()
        
        return True
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return False

def main():
    """Main function to start the system"""
    if not start_mqtt_client():
        return
    
    print("\nFog Machine with Telegram Integration running.")
    print("Waiting for video frames from Raspberry Pi camera...")
    print("\nAvailable commands:")
    print("  analyze - Re-analyze latest saved video")
    print("  status - Show system status")
    print("  exit - Exit the program")
    
    try:
        while True:
            command = input("> ").strip().lower()
            
            if command == "exit":
                break
            elif command == "analyze":
                # Find the most recent video file
                video_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp4') and not f.endswith('_analyzed.mp4')]
                if video_files:
                    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
                    latest_video = os.path.join(SAVE_DIR, video_files[0])
                    print(f"Re-analyzing latest video: {latest_video}")
                    
                    # Load the video and analyze it
                    cap = cv2.VideoCapture(latest_video)
                    frame_buffer.clear()
                    idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_buffer[idx] = frame
                        idx += 1
                    cap.release()
                    
                    total_frames = len(frame_buffer)
                    analyze_frames()
                else:
                    print("No video files found to analyze")
            elif command == "status":
                print("\n--- System Status ---")
                print(f"MQTT Connected: {client.is_connected()}")
                print(f"Device: {DEVICE}")
                print(f"Frames in buffer: {len(frame_buffer)}")
                print(f"Analysis active: {playback_active}")
                print(f"Total frames expected: {total_frames}")
                print(f"FPS: {fps}")
                print("--------------------\n")
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        playback_active = False
        
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join(timeout=2)
            
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()
        
        # Send shutdown notification
        shutdown_msg = f"Fog Machine Analysis System Shutdown\n {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        run_telegram_async(send_telegram_message(ANOMALY_DETECTION_CHANNEL_ID, shutdown_msg))
        
        print("Resources released, exiting")

if __name__ == "__main__":
    main()