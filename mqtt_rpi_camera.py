import paho.mqtt.client as mqtt
import cv2
import time
import os
import numpy as np
import base64

# MQTT Broker settings
MQTT_BROKER = "192.168.50.209"  # Change to your broker IP
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"

# Camera settings
CAMERA_ID = 0  # Usually 0 for the primary camera
FPS = 10  # Frames per second to capture
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Create directory for storing images if it doesn't exist
SAVE_DIR = "captured_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Connect to MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to trigger topic
        client.subscribe(TRIGGER_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

# Handle trigger messages
def on_message(client, userdata, msg):
    if msg.topic == TRIGGER_TOPIC:
        trigger_value = msg.payload.decode()
        if trigger_value.lower() == "start":
            print("Received trigger to start streaming")
            userdata["streaming"] = True
        elif trigger_value.lower() == "stop":
            print("Received trigger to stop streaming")
            userdata["streaming"] = False

# Initialize MQTT client
userdata = {"streaming": False}
client = mqtt.Client(client_id="RpiCameraPublisher", userdata=userdata)
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    # Start network loop in background thread
    client.loop_start()
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit(1)

# Initialize camera
camera = cv2.VideoCapture(CAMERA_ID)
if not camera.isOpened():
    print("Error: Could not open camera")
    client.loop_stop()
    exit(1)

# Set camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

print("Camera initialized. Waiting for trigger...")

try:
    while True:
        # Check if streaming is enabled
        if userdata["streaming"]:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Option 1: Save frame as file and send filename
            timestamp = int(time.time() * 1000)
            filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Publish filename to MQTT
            client.publish(MQTT_TOPIC + "/filename", filename)
            print(f"Published frame filename: {filename}")
            
            # Option 2: Encode frame as base64 and send directly
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer)
            
            # Split into chunks if needed (for large frames)
            chunk_size = 16384  # Adjust based on your MQTT broker's max message size
            chunks = [jpg_as_text[i:i+chunk_size] for i in range(0, len(jpg_as_text), chunk_size)]
            
            # Send number of chunks first
            client.publish(MQTT_TOPIC + "/chunks", str(len(chunks)))
            
            # Send each chunk
            for i, chunk in enumerate(chunks):
                client.publish(f"{MQTT_TOPIC}/chunk/{i}", chunk)
            
            print(f"Published frame as {len(chunks)} chunks")
            
            # Control frame rate
            time.sleep(1/FPS)
        else:
            # Just wait a bit and check again
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up
    camera.release()
    client.loop_stop()
    client.disconnect()
    print("Resources released, exiting")