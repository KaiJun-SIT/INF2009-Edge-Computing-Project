import paho.mqtt.client as mqtt
import time
import base64

# MQTT SETUP
MQTT_BROKER = "172.20.10.10"
MQTT_PORT = 1883
MQTT_TOPIC = "camera/stream"
TRIGGER_TOPIC = "camera/trigger"

def connect_mqtt(broker=MQTT_BROKER, port=MQTT_PORT, client_name="Trigger"):
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_name)
    except:
        # Fallback for older Paho MQTT versions
        client = mqtt.Client(client_name)
    client.connect(broker, port, 60)
    print(f"Connected to MQTT broker at {broker}: {port}")
    return client

# Function to send frames via MQTT
def send_frames_via_mqtt(frames, client, fps=15):
    """Send frames to the fog machine via MQTT."""

    if client is None:
        print("MQTT client is not connected.")
        return

    # First send a start signal
    client.publish(TRIGGER_TOPIC, "START\n")
    
    # Send number of frames
    client.publish(MQTT_TOPIC + "/metadata", f"{len(frames)}, {fps}")
    
    # Send each frame
    for i, frame in enumerate(frames):
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        # Encode as base64 to avoid binary transmission issues
        jpg_as_text = base64.b64encode(buffer)
        # Send the frame
        client.publish(MQTT_TOPIC + f"/frame/{i}", jpg_as_text)
        # Short sleep to avoid overwhelming the broker
        time.sleep(0.01)
    
    print("Finished sending frames.")

def publish_message(topic, message):
    """
    Simple function to publish a text message to a given MQTT topic.
    """
    global client
    if client:
        client.publish(topic, message)