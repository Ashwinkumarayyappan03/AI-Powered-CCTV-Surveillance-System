# main.py
import cv2
import time
import threading
from config import Config
from utils.video_stream import VideoStream
from detection.yolo_detector import YOLODetector
from detection.behavior_cnn import BehaviorAnalyzer
from alerts.notifier import AlertNotifier
from dashboard import app as dashboard

def main():
    config = Config()
    
    # Initialize components
    print("Initializing video stream...")
    video_stream = VideoStream(
        source=config.VIDEO_SOURCE,
        target_fps=config.TARGET_FPS
    ).start()
    
    print("Loading YOLOv8 model...")
    detector = YOLODetector(
        model_path=config.YOLO_MODEL,
        confidence=config.CONFIDENCE_THRESHOLD
    )
    
    print("Initializing behavior analyzer...")
    behavior_analyzer = BehaviorAnalyzer(sequence_length=16)
    
    print("Setting up alert system...")
    notifier = AlertNotifier(config)
    
    # Set up dashboard
    dashboard.app.video_stream = video_stream
    dashboard.app.detector = detector
    
    # Start dashboard in background thread
    dashboard_thread = threading.Thread(
        target=lambda: dashboard.socketio.run(
            dashboard.app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
            use_reloader=False
        ),
        daemon=True
    )
    dashboard_thread.start()
    print(f"Dashboard running at http://localhost:{config.DASHBOARD_PORT}")
    
    # Main processing loop
    prev_frame = None
    frame_count = 0
    start_time = time.time()
    
    print("Starting surveillance system...")
    
    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Run YOLO detection
            detections = detector.detect(frame)
            people_count = detector.count_people(detections)
            
            # Check for weapons
            for det in detections:
                if det['class_name'] in ['knife', 'scissors']:
                    notifier.send_alert(
                        'WEAPON_DETECTED',
                        f"Potential weapon detected: {det['class_name']} "
                        f"(confidence: {det['confidence']:.2f})",
                        frame
                    )
                    dashboard.add_alert('weapon', f"{det['class_name']} detected")
            
            # Add frame to behavior analyzer
            behavior_analyzer.add_frame(frame)
            
            # Analyze behavior every 8 frames
            if frame_count % 8 == 0:
                behavior, confidence = behavior_analyzer.analyze()
                if behavior and behavior != 'normal' and confidence > 0.7:
                    notifier.send_alert(
                        behavior.upper(),
                        f"Suspicious behavior detected: {behavior} "
                        f"(confidence: {confidence:.2f})",
                        frame
                    )
                    dashboard.add_alert(behavior, f"Detected with {confidence:.0%} confidence")
            
            # Motion anomaly detection
            is_anomaly, motion_score = behavior_analyzer.detect_motion_anomaly(
                prev_frame, frame, threshold=8000
            )
            if is_anomaly:
                notifier.send_alert(
                    'MOTION_ANOMALY',
                    f"Unusual motion detected (score: {motion_score:.0f})",
                    frame
                )
            
            prev_frame = frame.copy()
            
            # Print FPS every 5 seconds
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = frame_count / elapsed
                print(f"Processing at {fps:.1f} FPS | People: {people_count}")
                frame_count = 0
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
