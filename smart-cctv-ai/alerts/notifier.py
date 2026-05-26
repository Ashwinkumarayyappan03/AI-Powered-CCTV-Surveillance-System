# alerts/notifier.py
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import io

class AlertNotifier:
    def __init__(self, config):
        self.config = config
        self.last_alert_time = {}
        self.cooldown = config.ALERT_COOLDOWN
        
        # Try importing Twilio
        try:
            from twilio.rest import Client
            self.twilio_client = Client(config.TWILIO_SID, config.TWILIO_TOKEN)
            self.twilio_enabled = True
        except:
            self.twilio_enabled = False
            print("Twilio not configured - SMS alerts disabled")
    
    def _can_alert(self, event_type):
        """Check if enough time has passed since last alert of this type."""
        now = time.time()
        if event_type in self.last_alert_time:
            if now - self.last_alert_time[event_type] < self.cooldown:
                return False
        self.last_alert_time[event_type] = now
        return True
    
    def send_alert(self, event_type, message, frame=None):
        """Send alert via email and SMS (non-blocking)."""
        if not self._can_alert(event_type):
            return False
        
        # Send in background thread
        thread = threading.Thread(
            target=self._send_alert_async,
            args=(event_type, message, frame),
            daemon=True
        )
        thread.start()
        return True
    
    def _send_alert_async(self, event_type, message, frame):
        """Actual alert sending logic."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] ALERT: {event_type}\n\n{message}"
        
        # Send email
        self._send_email(event_type, full_message, frame)
        
        # Send SMS
        if self.twilio_enabled:
            self._send_sms(f"SECURITY ALERT: {event_type} - {message[:100]}")
    
    def _send_email(self, subject, body, frame=None):
        """Send email with optional image attachment."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.EMAIL_SENDER
            msg['To'] = ", ".join(self.config.EMAIL_RECIPIENTS)
            msg['Subject'] = f"🚨 Security Alert: {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach frame if provided
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                image = MIMEImage(buffer.tobytes(), name='alert_capture.jpg')
                msg.attach(image)
            
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.EMAIL_SENDER, self.config.EMAIL_PASSWORD)
                server.send_message(msg)
                
            print(f"Email alert sent: {subject}")
        except Exception as e:
            print(f"Email send failed: {e}")
    
    def _send_sms(self, message):
        """Send SMS via Twilio."""
        try:
            for recipient in self.config.SMS_RECIPIENTS:
                if recipient.strip():
                    self.twilio_client.messages.create(
                        body=message,
                        from_=self.config.TWILIO_FROM,
                        to=recipient.strip()
                    )
            print("SMS alert sent")
        except Exception as e:
            print(f"SMS send failed: {e}")
