# Configuration for Smart Lift Monitoring System (SLMS)

# Telegram Configuration
TELEGRAM_CONFIG = {
    "bot_token": "YOUR_BOT_TOKEN_HERE",  # Get this from BotFather on Telegram
    "channels": {
        "anomaly": "CHANNEL_ID_1",    # Primary notification channel
        "priority1": "CHANNEL_ID_2",  # Low priority issues
        "priority2": "CHANNEL_ID_3",  # Medium priority issues
        "priority3": "CHANNEL_ID_4"   # High priority/emergency issues
    }
}

# Hardware Configuration
HARDWARE_CONFIG = {
    # Ultrasonic sensor pins (BCM mode)
    "trig_pin": 23,
    "echo_pin": 24,
    
    # Camera settings
    "camera_index": 0,  # Use 0 for the first camera
    "frame_width": 640,
    "frame_height": 480,
    
    # Distance thresholds for ultrasonic sensor (in cm)
    "occupied_threshold": 150  # If distance is less than this, lift is considered occupied
}

# Detection Configuration
DETECTION_CONFIG = {
    # Confidence thresholds for different alert levels
    "confidence_thresholds": {
        "low": 0.6,      # Log only
        "medium": 0.7,   # Alert town council/security
        "high": 0.8      # Alert police/immediate response
    },
    
    # Time between notifications of the same type (seconds)
    "cooldown_period": 60,
    
    # Mapping of anomaly types to priority levels
    "priority_mapping": {
        "high_priority": ["fighting", "assault", "arson", "robbery"],
        "medium_priority": ["vandalism", "abuse"],
        "low_priority": ["littering"]
    },
    
    # Frame sequence length for video-based detection
    "frame_sequence_length": 16,
    
    # Size to resize frames to before processing
    "frame_size": (128, 128)
}

# Model Configuration
MODEL_CONFIG = {
    # Path to TFLite model file
    "model_path": "quantized_model.tflite",
    
    # Whether to use the ML model for detection
    # If False, will use mock detection for testing
    "use_model": False
}

# Storage Configuration
STORAGE_CONFIG = {
    # Directory to save evidence
    "evidence_dir": "evidence"
}
