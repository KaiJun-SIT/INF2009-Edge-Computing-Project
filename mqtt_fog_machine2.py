import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time
import base64
import os
import threading
import collections
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
from PIL import Image
from ultralytics import YOLO
from slowfast_model.model import resnet50

# MQTT settings
MQTT_BROKER = "172.20.10.10"  # Use same IP as publisher
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"

# Frame storage settings
SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Received frame buffer
frame_buffer = collections.OrderedDict()
total_frames = 0
fps = 15  # Will be updated from metadata
playback_active = False
playback_thread = None

# Connect with MQTT V5 protocol (or fall back to V3.1.1)
try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FogMachineReceiver")
except:
    # Fallback for older Paho MQTT versions
    client = mqtt.Client("FogMachineReceiver")

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
                
                # Save frame (optional)
                # cv2.imwrite(f"{SAVE_DIR}/frame_{frame_index}.jpg", frame)
                
                # Check if we have all frames
                if len(frame_buffer) == total_frames and not playback_active:
                    print(f"All {total_frames} frames received. Starting playback.")
                    
                    # Start playback in a new thread
                    playback_active = True
                    playback_thread = threading.Thread(target=play_frames)
                    playback_thread.daemon = True
                    playback_thread.start()
    
    except Exception as e:
        print(f"Error processing message: {e}")

def play_frames():
    """Play received frames and control fog machine"""
    global playback_active, frame_buffer
    
    try:
        # Sort frames by index
        sorted_frames = [frame_buffer[i] for i in sorted(frame_buffer.keys())]
        
        # Set up window for display
        cv2.namedWindow("Anomaly Video", cv2.WINDOW_NORMAL)
        
        # Calculate frame delay based on FPS
        frame_delay = 1.0 / fps
        
        # Create video writer for saving
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = f"{SAVE_DIR}/anomaly_{timestamp}.mp4"
        height, width = sorted_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        
        # Activate fog machine when starting playback
        activate_fog_machine(True)
        print("🌫️ FOG MACHINE ACTIVATED 🌫️")
        
        # Play all frames
        for i, frame in enumerate(sorted_frames):
            if not playback_active:
                break
                
            # Display frame
            cv2.imshow("Anomaly Video", frame)
            
            # Write frame to video
            out.write(frame)
            
            # Wait for key press (q to quit) or delay based on FPS
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep to maintain correct playback speed
            time.sleep(frame_delay)
            
            # Show progress
            if i % 10 == 0:
                print(f"Playing frame {i+1}/{len(sorted_frames)}")
        
        # Deactivate fog machine after playback
        activate_fog_machine(False)
        print("🚫 FOG MACHINE DEACTIVATED 🚫")
        
        # Release resources
        out.release()
        cv2.destroyAllWindows()
        playback_active = False
        
        print(f"Playback complete. Video saved to {out_file}")
        
    except Exception as e:
        print(f"Error during playback: {e}")
        playback_active = False
        activate_fog_machine(False)

def activate_fog_machine(activate=True):
    """
    Function to control the fog machine
    Replace with your actual fog machine control code
    """
    # PLACEHOLDER: Add your fog machine control logic here
    # Examples:
    # - GPIO control for Raspberry Pi
    # - Serial communication with Arduino/microcontroller
    # - Network commands to smart plug or controller
    
    status = "ACTIVATED" if activate else "DEACTIVATED"
    print(f"Fog machine {status}")
    
    # Example GPIO control:
    """
    import RPi.GPIO as GPIO
    FOG_CONTROL_PIN = 17
    
    # Set up GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FOG_CONTROL_PIN, GPIO.OUT)
    
    # Control fog machine
    GPIO.output(FOG_CONTROL_PIN, GPIO.HIGH if activate else GPIO.LOW)
    """

def send_trigger(command):
    """Send a trigger command"""
    try:
        trigger_client = mqtt.Client("FogTriggerSender")
        trigger_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        trigger_client.publish(TRIGGER_TOPIC, command)
        print(f"Sent trigger: {command}")
        trigger_client.disconnect()
    except Exception as e:
        print(f"Failed to send trigger: {e}")

# Set up MQTT client
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit(1)

# Main loop
try:
    print("Fog machine receiver running.")
    print("Available commands:")
    print("  start - Request video streaming from RPi")
    print("  stop - Stop video streaming/playback")
    print("  exit - Exit the program")
    
    while True:
        command = input("> ").strip().lower()
        
        if command == "exit":
            break
        elif command in ["start", "stop"]:
            send_trigger(command)
            if command == "stop" and playback_active:
                playback_active = False
        else:
            print("Unknown command")
            
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Clean up
    playback_active = False
    if playback_thread and playback_thread.is_alive():
        playback_thread.join(timeout=2)
        
    client.loop_stop()
    client.disconnect()
    cv2.destroyAllWindows()
    
    print("Resources released, exiting")