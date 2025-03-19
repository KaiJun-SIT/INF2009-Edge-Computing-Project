import RPi.GPIO as GPIO
import time

# Configuration
TRIG_PIN = 23
ECHO_PIN = 24

def setup():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    
    # Ensure trigger is low
    GPIO.output(TRIG_PIN, False)
    print("Waiting for sensor to settle...")
    time.sleep(2)
    print("Sensor ready!")

def get_distance():
    """Measure distance from ultrasonic sensor"""
    # Send 10us pulse to trigger
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    # Wait for echo to start (pulse sent)
    timeout_start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if time.time() - timeout_start > 0.5:
            return -1  # Timeout

    # Wait for echo to end (pulse received)
    timeout_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if time.time() - timeout_start > 0.5:
            return -1  # Timeout
    
    # Calculate pulse duration and convert to distance
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound ~343m/s, /2 for echo return
    distance = round(distance, 2)
    
    return distance

def main():
    """Main test loop"""
    setup()
    
    try:
        print("Testing ultrasonic sensor. Press CTRL+C to exit.")
        while True:
            distance = get_distance()
            
            if distance < 0:
                print("Error: Sensor timeout")
            else:
                print(f"Distance: {distance} cm")
                
                # Classification for testing
                if distance < 100:
                    print("STATUS: OCCUPIED (CLOSE)")
                elif distance < 150:
                    print("STATUS: OCCUPIED (MEDIUM)")
                else:
                    print("STATUS: EMPTY")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        GPIO.cleanup()
        print("GPIO cleanup complete")

if __name__ == "__main__":
    main()
