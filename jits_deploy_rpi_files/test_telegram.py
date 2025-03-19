import requests
import time
import cv2
import numpy as np

def test_telegram_bot(bot_token, chat_id):
    """Test sending messages and images to Telegram"""
    
    # Test parameters
    print("Testing Telegram bot functionality...")
    print(f"Bot Token: {bot_token}")
    print(f"Chat ID: {chat_id}")
    
    # Test sending a simple text message
    print("\nTesting text message...")
    message_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message_payload = {
        "chat_id": chat_id,
        "text": "🔧 SLMS Bot Test - This is a test message",
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(message_url, json=message_payload)
        if response.status_code == 200:
            print("✅ Text message sent successfully")
        else:
            print(f"❌ Failed to send text message: {response.text}")
    except Exception as e:
        print(f"❌ Error sending text message: {e}")
    
    # Test sending a test image
    print("\nTesting image message...")
    
    # Create a test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White background
    # Add text to image
    cv2.putText(
        test_image,
        "SLMS Test Image",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),  # Red text
        2
    )
    
    # Save test image
    test_image_path = "telegram_test_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"Created test image: {test_image_path}")
    
    # Send the image
    photo_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with open(test_image_path, 'rb') as photo:
            response = requests.post(
                photo_url,
                data={"chat_id": chat_id, "caption": "SLMS Test Image"},
                files={"photo": photo}
            )
        
        if response.status_code == 200:
            print("✅ Image sent successfully")
        else:
            print(f"❌ Failed to send image: {response.text}")
    except Exception as e:
        print(f"❌ Error sending image: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    print("Telegram Bot Tester")
    print("-----------------")
    
    # Get bot token and chat id from user
    bot_token = input("Enter your Telegram bot token: ")
    chat_id = input("Enter the chat ID to test: ")
    
    test_telegram_bot(bot_token, chat_id)
