# dashboard/app.py
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO
import cv2
import json
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# These will be set by main.py
video_stream = None
detector = None
stats = {
    'fps': 0,
    'people_count': 0,
    'alerts': [],
    'status': 'running'
}

def generate_frames():
    """Generate annotated frames for streaming."""
    global stats
    
    while True:
        frame = video_stream.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Run detection
        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame, detections)
        
        # Update stats
        stats['fps'] = round(video_stream.fps, 1)
        stats['people_count'] = detector.count_people(detections)
        
        # Add overlay info
        cv2.putText(annotated, f"FPS: {stats['fps']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"People: {stats['people_count']}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

def add_alert(alert_type, message):
    """Add alert to dashboard."""
    stats['alerts'].insert(0, {
        'type': alert_type,
        'message': message,
        'time': time.strftime("%H:%M:%S")
    })
    stats['alerts'] = stats['alerts'][:50]  # Keep last 50
    socketio.emit('new_alert', stats['alerts'][0])
