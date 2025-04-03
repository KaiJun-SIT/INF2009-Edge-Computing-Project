import paho.mqtt.client as mqtt
import cv2
import time
import os
import numpy as np

import base64
import threading

# MQTT Broker settings - Change these to match your setup
MQTT_BROKER = "172.20.10.10"
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"

# Frame storage settings
SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# For reconstructing chunked frames
current_frame_chunks = {}
expected_chunks = 0
frame_lock = threading.Lock()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to all relevant topics
        client.subscribe(f"{MQTT_TOPIC}/filename")
        client.subscribe(f"{MQTT_TOPIC}/chunks")
        client.subscribe(f"{MQTT_TOPIC}/chunk/#")
        
        # Subscribe to trigger topic - useful for debugging
        client.subscribe(TRIGGER_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global current_frame_chunks, expected_chunks
    
    try:
        topic = msg.topic
        
        # Handle trigger messages
        if topic == TRIGGER_TOPIC:
            trigger_value = msg.payload.decode()
            print(f"Received trigger: {trigger_value}")
            
            # Add your custom trigger handling code here
            # ...
        
        # Handle filename messages (Option 1)
        elif topic == f"{MQTT_TOPIC}/filename":
            filename = msg.payload.decode()
            print(f"Received image filename: {filename}")
            
            # Check if file exists (for local file system)
            if os.path.exists(filename):
                frame = cv2.imread(filename)
                if frame is not None:
                    # Save a copy in our directory
                    timestamp = int(time.time() * 1000)
                    local_filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
                    cv2.imwrite(local_filename, frame)
                    
                    # This is where you can add your custom frame processing
                    process_frame(frame, local_filename)
                else:
                    print(f"Error: Could not read image from {filename}")
            else:
                print(f"Error: File not found at {filename}")
        
        # Handle chunked base64 image (Option 2)
        elif topic == f"{MQTT_TOPIC}/chunks":
            with frame_lock:
                # Reset frame chunks
                current_frame_chunks = {}
                expected_chunks = int(msg.payload.decode())
                print(f"Expecting {expected_chunks} chunks for next frame")
        
        elif topic.startswith(f"{MQTT_TOPIC}/chunk/"):
            chunk_index = int(topic.split('/')[-1])
            
            with frame_lock:
                # Store the chunk
                current_frame_chunks[chunk_index] = msg.payload
                
                # Check if we have all chunks
                if len(current_frame_chunks) == expected_chunks and expected_chunks > 0:
                    # Reconstruct the frame
                    chunks = [current_frame_chunks[i] for i in range(expected_chunks)]
                    jpg_as_text = b''.join(chunks)
                    
                    # Decode base64 to bytes
                    jpg_bytes = base64.b64decode(jpg_as_text)
                    
                    # Convert to numpy array
                    np_arr = np.frombuffer(jpg_bytes, np.uint8)
                    
                    # Decode the JPEG image
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Save frame
                        timestamp = int(time.time() * 1000)
                        filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        # This is where you can add your custom frame processing
                        process_frame(frame, filename)
                        
                        print(f"Processed complete frame from {expected_chunks} chunks")
                    else:
                        print("Error: Could not decode image from chunks")
                    
                    # Reset for next frame
                    current_frame_chunks = {}
                    expected_chunks = 0
    
    except Exception as e:
        print(f"Error processing message: {e}")

def process_frame(frame, filename):
    """
    Custom frame processing function - replace with your own logic
    
    Args:
        frame: The OpenCV image frame (numpy array)
        filename: The filename where the frame was saved
    """
    # PLACEHOLDER: This is where you would implement your custom processing
    # For example:
    # - Analyze the frame for motion
    # - Detect specific objects
    # - Apply filters or transformations
    # - Control the fog machine based on analysis results
    
    # Example: Just print frame dimensions as a placeholder
    height, width, channels = frame.shape
    print(f"Received frame: {width}x{height}, saved to {filename}")
    
    # =====================================================
    # ADD YOUR CUSTOM FOG MACHINE CONTROL LOGIC HERE
    # =====================================================
    
    # Example placeholder for fog machine activation:
    # if detect_condition(frame):
    #     activate_fog_machine()

def send_trigger(client, command):
    """Utility function to send trigger commands"""
    client.publish(TRIGGER_TOPIC, command)
    print(f"Sent trigger: {command}")

# Initialize MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,"SimplifiedSubscriber", protocol=mqtt.MQTTv311)
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

try:
    print("Simplified MQTT video receiver running. Press Ctrl+C to exit.")
    print("Available commands:")
    print("  start - Request video streaming from RPi")
    print("  stop - Request to stop video streaming")
    print("  exit - Exit the program")
    
    while True:
        command = input("> ").strip().lower()
        
        if command == "exit":
            break
        elif command in ["start", "stop"]:
            send_trigger(client, command)
        else:
            print("Unknown command")

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up
    client.loop_stop()
    client.disconnect()
    print("Resources released, exiting")