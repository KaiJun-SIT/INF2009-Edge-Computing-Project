import cv2
import numpy as np
import time
import os
import requests
import json
import RPi.GPIO as GPIO
from collections import deque
from PIL import Image

# Telegram Bot Setup
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHANNELS = {
    "anomaly": "CHANNEL_ID_1",  # Channel for initial anomaly detection
    "priority1": "CHANNEL_ID_2", # Low priority issues (littering, minor vandalism)
    "priority2": "CHANNEL_ID_3", # Medium priority (vandalism)
    "priority3": "CHANNEL_ID_4"  # High priority (fighting, assault)
}

# Ultrasonic Sensor Setup
GPIO.setmode(GPIO.BCM)
TRIG_PIN = 23
ECHO_PIN = 24
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Global Variables
frame_sequence = deque(maxlen=16)  # Store last 16 frames
CONFIDENCE_THRESHOLDS = {
    "low": 0.6,      # Log only
    "medium": 0.7,   # Alert town council/security
    "high": 0.8      # Alert police/immediate response
}
FRAME_SIZE = (128, 128)
last_detection_time = 0
DETECTION_COOLDOWN = 60  # seconds between notifications for same event type

# TFLite model setup - simplified for testing
# In a real deployment, you would load your converted TFLite model
use_tflite_model = False  # Set to True when you have the model ready

class AnomalyDetector:
    def __init__(self):
        self.last_alerts = {}  # Track last alert time per category
        
        # For testing without the model
        self.anomaly_types = [
            "normal", "fighting", "vandalism", "littering", 
            "assault", "arson", "abuse", "robbery"
        ]
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Check connections.")
            
        print("✅ Camera initialized successfully")

    def detect_distance(self):
        """Read distance from ultrasonic sensor"""
        # Send trigger pulse
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        # Get echo time
        start_time = time.time()
        stop_time = time.time()

        # Record time of signal send
        while GPIO.input(ECHO_PIN) == 0:
            start_time = time.time()
            # Add timeout to prevent hanging
            if time.time() - stop_time > 0.1:
                return 400  # Return a large distance (no object)

        # Record time of signal return
        while GPIO.input(ECHO_PIN) == 1:
            stop_time = time.time()
            # Add timeout to prevent hanging
            if stop_time - start_time > 0.1:
                return 400  # Return a large distance (no object)

        # Calculate distance
        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2  # Speed of sound in cm/s
        return distance

    def is_lift_occupied(self):
        """Check if the lift is occupied based on ultrasonic sensor"""
        distance = self.detect_distance()
        # Assuming lift has a max width of 200cm
        # If distance is less than 150cm, something is likely in the lift
        return distance < 150
        
    def mock_anomaly_detection(self, frame):
        """Simulate anomaly detection for testing without ML model"""
        # This is just a placeholder that randomly "detects" anomalies
        # In production, you would use your actual ML model
        
        # For testing: randomly detect an anomaly 10% of the time
        if np.random.random() < 0.1:
            # Choose a random anomaly type (excluding "normal")
            anomaly_idx = np.random.randint(1, len(self.anomaly_types))
            anomaly_type = self.anomaly_types[anomaly_idx]
            
            # Random confidence score between 0.6 and 0.95
            confidence = np.random.uniform(0.6, 0.95)
            
            return anomaly_type, confidence
        
        return "normal", 0.05
    
    def get_priority_level(self, anomaly_type, confidence):
        """Determine the priority level based on anomaly type and confidence"""
        high_priority = ["fighting", "assault", "arson", "robbery"]
        medium_priority = ["vandalism", "abuse"]
        low_priority = ["littering"]
        
        if anomaly_type in high_priority and confidence > CONFIDENCE_THRESHOLDS["medium"]:
            return 3
        elif anomaly_type in medium_priority and confidence > CONFIDENCE_THRESHOLDS["low"]:
            return 2
        elif anomaly_type in low_priority and confidence > CONFIDENCE_THRESHOLDS["low"]:
            return 1
        else:
            return 0
    
    def send_telegram_alert(self, message, priority_level, image_path=None):
        """Send alert with optional image to appropriate Telegram channel"""
        global last_detection_time
        
        # First, determine which channel to use based on priority
        if priority_level == 3:
            chat_id = TELEGRAM_CHANNELS["priority3"]
            alert_type = "HIGH PRIORITY"
        elif priority_level == 2:
            chat_id = TELEGRAM_CHANNELS["priority2"]
            alert_type = "MEDIUM PRIORITY"
        elif priority_level == 1:
            chat_id = TELEGRAM_CHANNELS["priority1"]
            alert_type = "LOW PRIORITY"
        else:
            chat_id = TELEGRAM_CHANNELS["anomaly"]
            alert_type = "ANOMALY DETECTED"
            
        # Check cooldown period to avoid spam
        current_time = time.time()
        if alert_type in self.last_alerts and current_time - self.last_alerts[alert_type] < DETECTION_COOLDOWN:
            print(f"Cooldown active for {alert_type}. Skipping alert.")
            return
            
        # Update last alert time
        self.last_alerts[alert_type] = current_time
        
        # Format the message
        formatted_message = f"⚠️ {alert_type} ⚠️\n{message}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
        # Send message
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": formatted_message
        }
        
        try:
            response = requests.post(send_url, json=payload)
            print(f"Message sent to Telegram. Response: {response.status_code}")
            
            # If we have an image to send, send it as a second message
            if image_path:
                self.send_telegram_image(chat_id, image_path, f"{alert_type} - Visual Evidence")
                
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
    
    def send_telegram_image(self, chat_id, image_path, caption=""):
        """Send an image to a Telegram channel"""
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        try:
            with open(image_path, 'rb') as image_file:
                response = requests.post(
                    send_url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": image_file}
                )
            print(f"Image sent to Telegram. Response: {response.status_code}")
        except Exception as e:
            print(f"Error sending Telegram image: {e}")
            
    def save_evidence(self, frame, anomaly_type, confidence):
        """Save a frame as evidence when an anomaly is detected"""
        # Create directory if it doesn't exist
        evidence_dir = "evidence"
        os.makedirs(evidence_dir, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{evidence_dir}/{timestamp}_{anomaly_type}_{confidence:.2f}.jpg"
        
        # Save the image
        cv2.imwrite(filename, frame)
        print(f"Evidence saved to {filename}")
        
        return filename
        
    def process_frame(self, frame):
        """Process a single frame"""
        # Resize frame for display and potential model input
        display_frame = cv2.resize(frame, (640, 480))
        
        # Make a copy for potential evidence saving
        evidence_frame = display_frame.copy()
        
        # Add frame to our deque for sequence-based detection
        # (when your TFLite model is ready)
        small_frame = cv2.resize(frame, FRAME_SIZE)
        frame_sequence.append(small_frame)
        
        # Run anomaly detection (mock for now)
        anomaly_type, confidence = self.mock_anomaly_detection(small_frame)
        
        # Add text to display frame
        cv2.putText(
            display_frame,
            f"Detection: {anomaly_type} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if anomaly_type != "normal" else (0, 255, 0),
            2
        )
        
        # Handle detection result
        if anomaly_type != "normal" and confidence > CONFIDENCE_THRESHOLDS["low"]:
            # Determine priority
            priority_level = self.get_priority_level(anomaly_type, confidence)
            
            # Save evidence
            evidence_path = self.save_evidence(evidence_frame, anomaly_type, confidence)
            
            # Send alerts based on priority
            message = f"Detected: {anomaly_type}\nConfidence: {confidence:.2f}"
            
            # First time anomaly detected - send to anomaly channel
            self.send_telegram_alert("Anomaly detected in lift!", 0, evidence_path)
            
            # If it has a priority level, send to appropriate channel
            if priority_level > 0:
                self.send_telegram_alert(message, priority_level, evidence_path)
        
        return display_frame, anomaly_type, confidence
        
    def run_detection_loop(self):
        """Main detection loop"""
        try:
            print("Starting detection loop. Press 'q' to quit.")
            while True:
                # Check if lift is occupied using ultrasonic sensor
                occupied = self.is_lift_occupied()
                
                if not occupied:
                    # Display "Lift Empty" message and wait
                    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
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
                    
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    time.sleep(1)
                    continue
                
                # Process frame
                display_frame, anomaly_type, confidence = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow("SLMS Monitor", display_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Detection loop stopped by user.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released and GPIO cleanup complete.")

if __name__ == "__main__":
    print("Initializing Smart Lift Monitoring System (SLMS)...")
    detector = AnomalyDetector()
    detector.run_detection_loop()
