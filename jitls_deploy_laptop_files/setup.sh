#!/bin/bash

echo "Setting up Smart Lift Monitoring System (SLMS)..."

# Make sure we're up to date
echo "Updating package lists..."
sudo apt-get update

# Install required system packages
echo "Installing required system packages..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev

# Install required Python packages
echo "Installing required Python packages..."
pip3 install -U \
    numpy \
    opencv-python \
    RPi.GPIO \
    requests \
    Pillow

# Optional: Only install TensorFlow Lite if going to use model
read -p "Do you want to install TensorFlow Lite for ML model support? (y/n) " INSTALL_TF
if [ "$INSTALL_TF" = "y" ]; then
    echo "Installing TensorFlow Lite..."
    pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl
fi

# Create directories
echo "Creating required directories..."
mkdir -p evidence

# Set up configuration
echo "Setting up configuration..."
if [ ! -f config.py ]; then
    echo "Config file not found. Using the template configuration."
    # The config file will be created by the installer
else
    echo "Existing config.py file found. Keeping it."
fi

# Set up Telegram bot token
echo ""
echo "To enable Telegram notifications, you need to create a bot using BotFather."
echo "Instructions: https://core.telegram.org/bots#6-botfather"
echo ""
read -p "Do you want to set up Telegram notifications now? (y/n) " SETUP_TELEGRAM
if [ "$SETUP_TELEGRAM" = "y" ]; then
    read -p "Enter your Telegram bot token: " BOT_TOKEN
    # Use sed to replace the token placeholder in config.py
    sed -i "s/YOUR_BOT_TOKEN_HERE/$BOT_TOKEN/g" config.py
    
    echo ""
    echo "Next, you need to set up channels for different priority notifications."
    echo "Create up to 4 different channels in Telegram and add your bot to each."
    echo ""
    read -p "Enter channel ID for anomaly notifications: " CHANNEL1
    read -p "Enter channel ID for priority 1 (low) notifications: " CHANNEL2
    read -p "Enter channel ID for priority 2 (medium) notifications: " CHANNEL3
    read -p "Enter channel ID for priority 3 (high) notifications: " CHANNEL4
    
    # Update the channel IDs in config.py
    sed -i "s/CHANNEL_ID_1/$CHANNEL1/g" config.py
    sed -i "s/CHANNEL_ID_2/$CHANNEL2/g" config.py
    sed -i "s/CHANNEL_ID_3/$CHANNEL3/g" config.py
    sed -i "s/CHANNEL_ID_4/$CHANNEL4/g" config.py
    
    echo "Telegram configuration updated."
fi

echo ""
echo "Setup complete! You can now run the SLMS system with:"
echo "python3 slms_modular.py"
echo ""
