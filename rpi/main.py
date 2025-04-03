import asyncio
from datetime import datetime
from rpi_ultrasonic import start_ultrasonic_monitoring, stop_ultrasonic_monitoring, get_people_count
import rpi_model as model
from rpi_telegram import send_photo, send_start_msg, send_shutdown_msg, send_anomaly_msg
from rpi_mqtt import send_frames_via_mqtt, connect_mqtt

async def main():
    start_ultrasonic_monitoring()
    model.load_model()
    inference_running = False
    client = connect_mqtt()
    
    try:
        while True:
            count = get_people_count()
            if not inference_running:
                print(f"People in lift: {count}")
            
            if count > 0 and not inference_running:
                print("Starting video recording.")
                model.start_inference()
                inference_running = True

            if count > 0 and inference_running:
                # if anomaly is detected get anomaly info
                    # send anomaly info to telegram (photo and anomaly type)
                    # send anomaly info to MQTT (video frames and anomaly type)
                if model.get_anomaly_detected():
                    print(f"\nAnomaly detected IN MAIN YOOOOOOOOOOOOOOOOOOOOOOOO!\n")
                    anomaly_type, anomaly_frame, anomaly_confidence = model.get_anomaly_info()
                    
                    # Send photo to Telegram
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert_msg = (
                        f"🚨 POTENTIAL ANOMALY DETECTED!\n"
                        f"⏰ Time: {timestamp}\n"
                        f"🔍 Type: {anomaly_type}\n"
                        f"📏 Confidence: {anomaly_confidence:.2f}"
                        # f"⚡ Inference time: {inference_time*1000:.2f}ms"
                    )
                    print(f"SENDING TELEGRAM NOTI")
                    send_anomaly_msg(alert_msg)
                    
                    # Send frames via MQTT
                    frames = model.get_anomaly_clip()
                    send_frames_via_mqtt(frames, client)
                    
                    # Clear anomaly info
                    model.clear_anomaly()
            
            if count == 0 and inference_running:
                print("Stopping recording because no passengers in the lift.")
                model.stop_inference()
                inference_running = False
            
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl-C detected.")
    finally:
        if inference_running:
            model.stop_inference()
        stop_ultrasonic_monitoring()

if __name__ == "__main__":
    print("Starting main loop...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"Ctrl-C in main.py.")
    finally:
        stats_msg = model.get_inference_stats()
        send_shutdown_msg("📉 System Shutting Down\n" + stats_msg)
