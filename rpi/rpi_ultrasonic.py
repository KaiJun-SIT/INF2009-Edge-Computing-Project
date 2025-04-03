import threading
from gpiozero import DistanceSensor
from time import time, sleep

# Create two ultrasonic sensor objects
sensor_left = DistanceSensor(echo=24, trigger=23)
sensor_right = DistanceSensor(echo=22, trigger=27)

DIST_THRESHOLD = 70.0  # trigger threshold
TIMEOUT = 5.0   # seconds to wait for second sensor

# GLobal tracking
people_count   = 0

# Track whether sensor was "detected" last loop
left_last_detected  = False
right_last_detected = False

# Time at which each sensor last triggered a "leading edge"
left_trigger_time  = None
right_trigger_time = None

# Internal control
_run_flag = False   # Will be True while the monitoring thread runs
_sensor_thread = None

def _is_detected(dist_cm):
    """Return True if the ultrasonic distance is below threshold."""
    return dist_cm < DIST_THRESHOLD

def _sensor_loop():
    """
    This function runs in a background thread to continuously
    read the ultrasonic sensors and update `people_count`.
    """
    global people_count
    global left_last_detected, right_last_detected
    global left_trigger_time, right_trigger_time
    
    try:
        while _run_flag:
            # Measure distances sequentially to reduce crosstalk
            left_dist = sensor_left.distance * 100
            sleep(0.03)
            right_dist = sensor_right.distance * 100
            sleep(0.03)
            
            now = time()
            
            # Determine if each sensor sees an object
            left_current_detected = _is_detected(left_dist)
            right_current_detected = _is_detected(right_dist)
            
            # 1) Leading-edge detection (detect transition from not-detected to detected)
            if (not left_last_detected) and left_current_detected:
                left_trigger_time = now
            if (not right_last_detected) and right_current_detected:
                right_trigger_time = now
            
            # 2) If both triggers exist, compare order & time difference
            if left_trigger_time is not None and right_trigger_time is not None:
                if abs(left_trigger_time - right_trigger_time) <= TIMEOUT:
                    # Left triggered first => EXIT
                    if left_trigger_time < right_trigger_time:
                        people_count = max(people_count - 1, 0)
                        print(f"EXIT detected → People in lift: {people_count}")
                    # Right triggered first => ENTER
                    else:
                        people_count += 1
                        print(f"ENTER detected → People in lift: {people_count}")
                
                # Reset triggers after processing
                left_trigger_time = None
                right_trigger_time = None
            
            # Trigger timeout
            if left_trigger_time is not None and (now - left_trigger_time > TIMEOUT):
                left_trigger_time = None
            if right_trigger_time is not None and (now - right_trigger_time > TIMEOUT):
                right_trigger_time = None
            
            # Update last_detected states
            left_last_detected = left_current_detected
            right_last_detected = right_current_detected
            
            sleep(0.05)
            
    except KeyboardInterrupt:
        print("Ultrasonic sensor loop interrupted by user.")
    finally:
        print("Ultrasonic monitoring thread ended.")

def start_ultrasonic_monitoring():
    """
    Starts a background thread to continuously update `people_count`.
    Call this once at the beginning of your program.
    """
    global _run_flag, _sensor_thread
    if not _run_flag:
        _run_flag = True
        _sensor_thread = threading.Thread(target=_sensor_loop, daemon=True)
        _sensor_thread.start()
        print("Ultrasonic monitoring started.")

def stop_ultrasonic_monitoring():
    """
    Stops the background thread. Call this before exiting your program
    to cleanly stop sensor monitoring.
    """
    global _run_flag, _sensor_thread
    _run_flag = False
    if _sensor_thread is not None:
        _sensor_thread.join()
    print("Ultrasonic monitoring stopped.")

def get_people_count():
    """
    Returns the current number of people in the lift, as maintained by
    the background thread's sensor logic.
    """
    return people_count