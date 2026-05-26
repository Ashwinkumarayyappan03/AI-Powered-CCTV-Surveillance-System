# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Video source (0 for webcam, or RTSP URL)
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", 0)
    
    # Detection settings
    YOLO_MODEL = "yolov8s.pt"  # s for speed/accuracy balance
    CONFIDENCE_THRESHOLD = 0.5
    TARGET_FPS = 30
    
    # Alert settings
    ALERT_COOLDOWN = 30  # seconds between alerts for same event
    
    # Email
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "").split(",")
    
    # Twilio SMS
    TWILIO_SID = os.getenv("TWILIO_SID")
    TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
    TWILIO_FROM = os.getenv("TWILIO_FROM")
    SMS_RECIPIENTS = os.getenv("SMS_RECIPIENTS", "").split(",")
    
    # Dashboard
    DASHBOARD_HOST = "0.0.0.0"
    DASHBOARD_PORT = 5000
