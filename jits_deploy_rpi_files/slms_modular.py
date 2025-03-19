import cv2
import numpy as np
import time
import os
import requests
import json
import RPi.GPIO as GPIO
from collections import deque
from PIL import Image
import config

class UltrasonicSensor:
    """Class to handle the ultrasonic sensor for motion detection"""
    
    def __init__(self, trig_pin, echo_pin, occupied_threshold):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.occupied_threshold = occupied_threshold
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        
        print("✅ Ultrasonic sensor initialized")
        
    def get_distance(self):
        """Read distance from ultrasonic sensor"""
        # Send trigger pulse
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)

        # Get echo time
        start_time = time.time()
        stop_time = time.time()

        # Record time of signal send
        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            start_time = time.time()
            # Add timeout to prevent hanging
            if time.time() - timeout_start > 0.1:
                return 400  # Return a large distance (no object)

        # Record time of signal return
        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            stop_time = time.time()
            # Add timeout to prevent hanging
            if time.time() - timeout_start > 0.1:
                return 400  # Return a large distance (no object)

        # Calculate distance
        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2  # Speed of sound in cm/s
        return distance
        
    def is_occupied(self):
        """Check if the lift is occupied based on ultrasonic sensor"""
        distance = self.get_distance()
        # If distance is less than threshold, something is likely in the lift
        return distance < self.occupied_threshold


class TelegramNotifier:
    """Class to handle Telegram notifications"""
    
    def __init__(self, bot_token, channels):
        self.bot_token = bot_token
        self.channels = channels
        self.last_alerts = {}  # To track cooldown periods
        
        # Validate bot token by sending a test message to yourself
        if self.bot_token == "YOUR_BOT_TOKEN_HERE":
            print("⚠️ Warning: Telegram bot token not configured. Notifications will not work.")
        else:
            print("✅ Telegram notifier initialized")
    
    def send_message(self, channel_key, message):
        """Send a text message to a specific channel"""
        if self.bot_token == "YOUR_BOT_TOKEN_HERE":
            print(f"Would send to {channel_key}: {message}")
            return
            
        chat_id = self.channels.get(channel_key)
        if not chat_id:
            print(f"Error: Channel {channel_key} not found in configuration")
            return
            
        send_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(send_url, json=payload)
            if response.status_code == 200:
                print(f"✅ Message sent to {channel_key} channel")
            else:
                print(f"⚠️ Failed to send message: {response.text}")
        except Exception as e:
            print(f"⚠️ Error sending Telegram message: {e}")
    
    def send_image(self, channel_key, image_path, caption=""):
        """Send an image to a specific channel"""
        if self.bot_token == "YOUR_BOT_TOKEN_HERE":
            print(f"Would send image {image_path} to {channel_key} with caption: {caption}")
            return
            
        chat_id = self.channels.get(channel_key)
        if not chat_id:
            print(f"Error: Channel {channel_key} not found in configuration")
            return
            
        send_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        try:
            with open(image_path, 'rb') as image_file:
                response = requests.post(
                    send_url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": image_file}
                )
                
            if response.status_code == 200:
                print(f"✅ Image sent to {channel_key} channel")
            else:
                print(f"⚠️ Failed to send image: {response.text}")
        except Exception as e:
            print(f"⚠️ Error sending Telegram image: {e}")
    
    def send_alert(self, priority_level, message, image_path=None, cooldown=60):
        """Send alert with appropriate priority to the correct channel"""
        # Map priority level to channel
        channel_map = {
            0: "anomaly",     # Initial detection
            1: "priority1",   # Low priority
            2: "priority2",   # Medium priority
            3: "priority3"    # High priority
        }
        
        channel_key = channel_map.get(priority_level, "anomaly")
        alert_type = f"PRIORITY {priority_level}" if priority_level > 0 else "ANOMALY DETECTED"
        
        # Check cooldown period
        current_time = time.time()
        if channel_key in self.last_alerts and current_time - self.last_alerts[channel_key] < cooldown:
            print(f"⏳ Cooldown active for {channel_key}. Skipping alert.")
            return
            
        # Update last alert time
        self.last_alerts[channel_key] = current_time
        
        # Format message
        formatted_message = f"⚠️ <b>{alert_type}</b> ⚠️\n{message}\n\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send message
        self.send_message(channel_key, formatted_message)
        
        # Send image if provided
        if image_path:
            self.send_image(channel_key, image_path, caption=f"{alert_type} - Visual Evidence")


class MockAnomalyDetector:
    """Simple mock detector for testing without ML model"""
    
    def __init__(self, priority_mapping):
        self.anomaly_types = ["normal"] + list(set([item for sublist in priority_mapping.values() for item in sublist]))
        print(f"✅ Mock detector initialized with {len(self.anomaly_types)} possible anomaly types")
        
    def detect(self, frame):
        """Simulate detection with random anomalies"""
        # For testing: randomly detect an anomaly 10% of the time
        if np.random.random() < 0.1:
            # Choose a random anomaly type (excluding "normal")
            anomaly_idx = np.random.randint(1, len(self.anomaly_types))
            anomaly_type = self.anomaly_types[anomaly_idx]
            
            # Random confidence score between 0.6 and 0.95
            confidence = np.random.uniform(0.6, 0.95)
            
            return anomaly_type, confidence
        
        return "normal", 0.05


class TFLiteAnomalyDetector:
    """TFLite-based anomaly detector"""
    
    def __init__(self, model_path, frame_size):
        self.frame_size = frame_size
        self.frame_sequence = deque(maxlen=16)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model file {model_path} not found. Using mock detector instead.")
            self.model_available = False
            return
            
        try:
            # Import TFLite interpreter
            import tensorflow as tf
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Check input shape - this will depend on your model
            self.input_shape = self.input_details[0]['shape']
            
            self.model_available = True
            print(f"✅ TFLite model loaded successfully from {model_path}")
            print(f"ℹ️ Model expects input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"⚠️ Error loading TFLite model: {e}")
            self.model_available = False
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for model input"""
        # Resize to expected input size
        resized = cv2.resize(frame, self.frame_size)
        
        # Normalize pixel values to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Add to frame sequence
        self.frame_sequence.append(normalized)
        
        return normalized
        
    def detect(self, frame):
        """Detect anomalies in the current frame"""
        if not self.model_available:
            # Fall back to mock detector
            mock = MockAnomalyDetector(config.DETECTION_CONFIG["priority_mapping"])
            return mock.detect(frame)
            
        # Preprocess frame
        preprocessed = self.preprocess_frame(frame)
        
        # If we don't have enough frames yet, return normal
        if len(self.frame_sequence) < 16:
            return "normal", 0.0
            
        try:
            # Prepare input data - this will depend on your specific model
            # For SlowFast or similar models, you might need to process the frame sequence
            # This is a simplified example
            input_data = np.array([list(self.frame_sequence)], dtype=np.float32)
            
            # Reshape if needed to match model's expected input shape
            input_data = np.transpose(input_data, (0, 2, 1, 3, 4))  # Adjust based on your model
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process results - assuming binary classification (normal/anomaly)
            # Adjust this based on your model's output format
            confidence = output_data[0][1]  # Probability of anomaly
            
            # For testing purposes, let's pick a random anomaly type if confidence is high
            if confidence > 0.6:
                anomaly_types = ["fighting", "vandalism", "littering", "assault"]
                anomaly_type = np.random.choice(anomaly_types)
                return anomaly_type, confidence
            
            return "normal", confidence
            
        except Exception as e:
            print(f"⚠️ Error during inference: {e}")
            return "normal", 0.0


class SmartLiftMonitoringSystem:
    """Main SLMS class to coordinate all components"""
    
    def __init__(self):
        # Initialize components based on configuration
        self.ultrasonic = UltrasonicSensor(
            config.HARDWARE_CONFIG["trig_pin"],
            config.HARDWARE_CONFIG["echo_pin"],
            config.HARDWARE_CONFIG["occupied_threshold"]
        )
        
        self.notifier = TelegramNotifier(
            config.TELEGRAM_CONFIG["bot_token"],
            config.TELEGRAM_CONFIG["channels"]
        )
        
        # Choose detector based on configuration
        if config.MODEL_CONFIG["use_model"]:
            self.detector = TFLiteAnomalyDetector(
                config.MODEL_CONFIG["model_path"],
                config.DETECTION_CONFIG["frame_size"]
            )
        else:
            self.detector = MockAnomalyDetector(
                config.DETECTION_CONFIG["priority_mapping"]
            )
        
        # Initialize camera
        self.camera = cv2.VideoCapture(config.HARDWARE_CONFIG["camera_index"])
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.HARDWARE_CONFIG["frame_width"])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HARDWARE_CONFIG["frame_height"])
        
        if not self.camera.isOpened():
            raise RuntimeError("Failed to initialize camera. Check connections.")
        
        # Create evidence directory if it doesn't exist
        os.makedirs(config.STORAGE_CONFIG["evidence_dir"], exist_ok=True)
        
        print("✅ Smart Lift Monitoring System initialized successfully")
    
    def get_priority_level(self, anomaly_type, confidence):
        """Determine priority level based on anomaly type and confidence"""
        # Get mappings from config
        priority_mapping = config.DETECTION_CONFIG["priority_mapping"]
        thresholds = config.DETECTION_CONFIG["confidence_thresholds"]
        
        # Check high priority anomalies
        if anomaly_type in priority_mapping["high_priority"] and confidence > thresholds["medium"]:
            return 3
        # Check medium priority anomalies
        elif anomaly_type in priority_mapping["medium_priority"] and confidence > thresholds["low"]:
            return 2
        # Check low priority anomalies
        elif anomaly_type in priority_mapping["low_priority"] and confidence > thresholds["low"]:
            return 1
        else:
            return 0
    
    def save_evidence(self, frame, anomaly_type, confidence):
        """Save frame as evidence"""
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{config.STORAGE_CONFIG['evidence_dir']}/{timestamp}_{anomaly_type}_{confidence:.2f}.jpg"
        
        # Save the image
        cv2.imwrite(filename, frame)
        print(f"✅ Evidence saved to {filename}")
        
        return filename
    
    def process_frame(self, frame):
        """Process a single frame for anomaly detection"""
        # Make a copy for display and evidence
        display_frame = frame.copy()
        evidence_frame = frame.copy()
        
        # Run detection on the frame
        anomaly_type, confidence = self.detector.detect(frame)
        
        # Add detection info to display frame
        cv2.putText(
            display_frame,
            f"Detection: {anomaly_type} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if anomaly_type != "normal" else (0, 255, 0),
            2
        )
        
        # Handle detection if anomaly found with sufficient confidence
        if anomaly_type != "normal" and confidence > config.DETECTION_CONFIG["confidence_thresholds"]["low"]:
            # Determine priority level
            priority_level = self.get_priority_level(anomaly_type, confidence)
            
            # Save evidence
            evidence_path = self.save_evidence(evidence_frame, anomaly_type, confidence)
            
            # Prepare notification message
            message = f"Detected: {anomaly_type}\nConfidence: {confidence:.2f}"
            
            # First time anomaly detected - always send to anomaly channel
            self.notifier.send_alert(
                0,  # Priority level 0 = anomaly channel
                "Anomaly detected in lift!",
                evidence_path,
                config.DETECTION_CONFIG["cooldown_period"]
            )
            
            # If it has a priority level, send to appropriate channel
            if priority_level > 0:
                self.notifier.send_alert(
                    priority_level,
                    message,
                    evidence_path,
                    config.DETECTION_CONFIG["cooldown_period"]
                )
        
        return display_frame, anomaly_type, confidence
        
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting Smart Lift Monitoring System...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Check if lift is occupied using ultrasonic sensor
                occupied = self.ultrasonic.is_occupied()
                
                if not occupied:
                    # Display "Lift Empty" message
                    empty_frame = np.zeros((
                        config.HARDWARE_CONFIG["frame_height"],
                        config.HARDWARE_CONFIG["frame_width"],
                        3
                    ), dtype=np.uint8)
                    
                    cv2.putText(
                        empty_frame,
                        "Lift Empty - Monitoring",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    
                    cv2.imshow("SLMS Monitor", empty_frame)
                    
                    # Check for quit key and wait
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                        
                    # Short wait before checking sensor again
                    time.sleep(0.5)
                    continue
                
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️ Error: Could not read frame from camera")
                    time.sleep(1)
                    continue
                
                # Process the frame
                display_frame, anomaly_type, confidence = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("SLMS Monitor", display_frame)
                
                # Check for quit key (set waitKey to 1ms for better responsiveness)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Release resources"""
        # Release camera
        self.camera.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Cleanup GPIO
        GPIO.cleanup()
        
        print("✅ Resources released and GPIO cleaned up")


# Main entry point
if __name__ == "__main__":
    try:
        # Create and run the system
        slms = SmartLiftMonitoringSystem()
        slms.run_monitoring_loop()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # Make sure we clean up GPIO even if there's an error
        GPIO.cleanup()