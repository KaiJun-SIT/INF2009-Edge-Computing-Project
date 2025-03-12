import torch
import cv2
import numpy as np
import pyaudio
import wave
from scipy.io import wavfile
from scipy.signal import spectrogram
from ultralytics import YOLO

# Load YOLOv8-Pose model
model = YOLO("yolov8n-pose.pt")  # Using a lightweight version for real-time detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ YOLOv8-Pose model loaded successfully!")

# Audio recording setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
p = pyaudio.PyAudio()

# detection functions
def detect_squatting(results):
    """ Detects squatting (knees significantly lower than hips). """
    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:
                continue

            left_knee, right_knee = pose[13][:2].cpu().numpy(), pose[14][:2].cpu().numpy()
            left_hip, right_hip = pose[11][:2].cpu().numpy(), pose[12][:2].cpu().numpy()

            avg_knee_height = (left_knee[1] + right_knee[1]) / 2
            avg_hip_height = (left_hip[1] + right_hip[1]) / 2

            if avg_knee_height > avg_hip_height * 1.2:
                return "Squatting Detected!"
    return "Normal"

def detect_audio_urination():
    """ Detects urination sound using microphone input. """
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    stream.stop_stream()
    stream.close()

    wave_file = wave.open("temp_audio.wav", "wb")
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(p.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    sample_rate, data = wavfile.read("temp_audio.wav")
    freqs, times, Sxx = spectrogram(data, sample_rate)

    low_freq_energy = np.mean(Sxx[freqs < 1000])

    return "Urination Sound Detected!" if low_freq_energy > 10 else "Normal"

def detect_fighting(results):
    """ Detects fighting by checking if hands are too close to each other. """
    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:
                continue

            left_wrist, right_wrist = pose[9][:2].cpu().numpy(), pose[10][:2].cpu().numpy()
            left_shoulder, right_shoulder = pose[5][:2].cpu().numpy(), pose[6][:2].cpu().numpy()

            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

            if wrist_distance < shoulder_distance * 0.5:
                return "Fighting Detected!"
    return "Normal"

def detect_assault(results):
    """ Detects molestation/harassment by tracking if hands are near another person's face, torso, or private area. """
    poses = [pose for result in results for pose in result.keypoints.data if len(pose) >= 17]

    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            person1, person2 = poses[i], poses[j]

            p1_hand = person1[9][:2].cpu().numpy()  # Left wrist of person 1
            p2_torso = person2[11][:2].cpu().numpy()  # Torso of person 2
            p2_head = person2[0][:2].cpu().numpy()  # Head of person 2
            p2_private_area = person2[13][:2].cpu().numpy()  # Hip area of person 2

            if (
                np.linalg.norm(p1_hand - p2_torso) < 50  # Hand too close to torso
                or np.linalg.norm(p1_hand - p2_head) < 50  # Hand too close to face
                or np.linalg.norm(p1_hand - p2_private_area) < 50  # Hand near private area
            ):
                return "Assault Detected!"

    return "Normal"

def detect_robbery(results):
    """ Detects robbery (hand close to another's torso). """
    return detect_assault(results) # Robbery is a subset of assault

def detect_vandalism(results):
    """ Detects vandalism (Person's hands near walls/buttons for too long). """
    for result in results:
        for pose in result.keypoints.data:
            if len(pose) < 17:
                continue

            right_hand, left_hand = pose[10][:2].cpu().numpy(), pose[9][:2].cpu().numpy()

            if right_hand[0] > 600 or left_hand[0] < 50:
                return "Vandalism Detected!"
    return "Normal"

def detect_arson(frame):
    """ Detects arson (Fire/sudden brightness increase). """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    return "Arson Detected!" if brightness > 180 else "Normal"

def detect_abuse(results):
    """ Detects urination/defecation using pose & audio. """
    squat_status = detect_squatting(results)
    urination_status = detect_audio_urination()

    return "Urinating/Defecating Detected!" if squat_status != "Normal" and urination_status != "Normal" else "Normal"

# Real-time detection loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame)

    # **Detect various behaviors**
    fight_alert = detect_fighting(results)
    assault_alert = detect_assault(results)
    robbery_alert = detect_robbery(results)
    vandal_alert = detect_vandalism(results)
    arson_alert = detect_arson(frame)
    abuse_alert = detect_abuse(results)

    # **Choose the most severe alert**
    alerts = [fight_alert, assault_alert, robbery_alert, vandal_alert, arson_alert, abuse_alert]
    final_alert = next((alert for alert in alerts if alert != "Normal"), "Normal")

    # Draw keypoints and bounding boxes
    annotated_frame = results[0].plot()

    # Display alert message
    cv2.putText(annotated_frame, final_alert, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Lift Behavior Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
p.terminate()
