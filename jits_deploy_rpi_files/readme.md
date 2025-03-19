# Smart Lift Monitoring System (SLMS)

## Overview

The Smart Lift Monitoring System (SLMS) is an edge-based surveillance solution designed for residential and business lifts. It can detect and report anomalous behavior including vandalism, littering, fighting, and other unwanted activities.

This system uses computer vision and motion sensing to detect when the lift is occupied and analyze behavior, sending alerts to different Telegram channels based on the type and severity of detected anomalies.

## Hardware Requirements

- Raspberry Pi 400 (or any Raspberry Pi)
- Webcam (USB camera)
- Ultrasonic sensor (HC-SR04)
- Jumper wires
- Internet connection for sending Telegram alerts

## Installation

1. Clone or download this repository to your Raspberry Pi

2. Run the setup script to install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. The setup script will guide you through configuring your Telegram bot and channels.

## Hardware Setup

### Ultrasonic Sensor (HC-SR04)

Connect the ultrasonic sensor to your Raspberry Pi as follows:

- VCC → 5V pin
- GND → Ground pin
- TRIG → GPIO 23 (configurable in config.py)
- ECHO → GPIO 24 (configurable in config.py)

### Camera

Connect a USB webcam to one of the USB ports on your Raspberry Pi.

## Configuration

The system behavior can be customized by editing the `config.py` file:

- **Telegram Configuration**: Set up bot token and channel IDs
- **Hardware Configuration**: Adjust pins, camera settings, and distance thresholds
- **Detection Configuration**: Modify confidence thresholds and alert priorities
- **Model Configuration**: Enable/disable ML model and set model path
- **Storage Configuration**: Change where evidence images are stored

## Using the System

### Starting the System

```
python3 slms_modular.py
```

### Operation Modes

1. **Monitoring Mode**: When the lift is empty (detected by ultrasonic sensor), the system will display "Lift Empty - Monitoring" and minimize resource usage.

2. **Detection Mode**: When someone enters the lift, the system begins analyzing behavior through the camera feed.

### Alert Levels

The system has 4 alert channels:

1. **Anomaly Channel**: Initial notification when any anomaly is detected
2. **Priority 1 (Low)**: Minor issues like littering
3. **Priority 2 (Medium)**: Issues like vandalism
4. **Priority 3 (High)**: Serious issues like fighting or assault

### Display

The system shows a live view with detection results overlaid. The text will be green for normal behavior and red for detected anomalies.

### Stopping the System

Press 'q' on the keyboard while the display window is in focus to gracefully stop the system.

## Testing Without ML Model

The default configuration uses a mock detector that randomly generates anomalies for testing purposes. This lets you verify the camera, ultrasonic sensor, and Telegram notification functionality without requiring GPU acceleration.

To test:

1. Make sure your ultrasonic sensor is properly connected
2. Run the system and create activity in front of the ultrasonic sensor
3. The system will occasionally generate random "anomalies" to test the notification pipeline

## Troubleshooting

### Camera Not Working

- Check if the camera is properly connected
- Try a different USB port
- Check permissions: `sudo usermod -a -G video $USER`

### Ultrasonic Sensor Issues

- Verify wiring connections
- Check GPIO pin configuration in config.py
- Test the sensor with a simple test script

### Telegram Notifications

- Verify internet connectivity
- Check that your bot token is correct
- Make sure the bot has been added to all channels
- Verify channel IDs are correctly configured

## Security and Privacy

The SLMS is designed with privacy in mind - all processing is done locally on the edge device, and no continuous data streaming to the cloud is required. Only when anomalies are detected are images saved and notifications sent.

## Contributors

Team 15:
- Abdul Haliq (2302747)
- Chan Jit Lin (2302683)
- Fun Kai Jun (2303556)
- Leena Soo (2302724)
- Muhammad Akid Nufairi Bin Nashily (2302868)
