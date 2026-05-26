# detection/yolo_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLODetector:
    # Classes relevant to surveillance
    THREAT_CLASSES = {
        'person': 0,
        'knife': 43,
        'scissors': 76,
        'cell phone': 67,  # potential recording device
    }
    
    def __init__(self, model_path="yolov8s.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = self.model.names
        
    def detect(self, frame):
        """Run detection and return results."""
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Color based on threat level
            color = (0, 255, 0)  # green default
            if det['class_name'] in ['knife', 'scissors']:
                color = (0, 0, 255)  # red for weapons
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def count_people(self, detections):
        """Count number of people detected."""
        return sum(1 for d in detections if d['class_name'] == 'person')
