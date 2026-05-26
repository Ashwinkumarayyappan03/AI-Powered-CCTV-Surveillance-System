# detection/behavior_cnn.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import deque

class BehaviorCNN(nn.Module):
    """CNN for detecting fights, theft, abnormal behavior from frame sequences."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        # Input: 16 frames × 3 channels × 64 × 64
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.classes = ['normal', 'fight', 'theft', 'loitering']
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class BehaviorAnalyzer:
    """Analyzes sequences of frames for abnormal behavior."""
    
    def __init__(self, model_path=None, sequence_length=16, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        
        self.model = BehaviorCNN(num_classes=4).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['normal', 'fight', 'theft', 'loitering']
    
    def add_frame(self, frame):
        """Add frame to buffer."""
        processed = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.frame_buffer.append(processed)
    
    def analyze(self):
        """Analyze current frame buffer for behavior."""
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0
        
        # Stack frames: (C, T, H, W)
        frames = torch.stack(list(self.frame_buffer), dim=1)
        frames = frames.unsqueeze(0).to(self.device)  # Add batch dim
        
        with torch.no_grad():
            outputs = self.model(frames)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        behavior = self.classes[predicted.item()]
        conf = confidence.item()
        
        return behavior, conf
    
    def detect_motion_anomaly(self, prev_frame, curr_frame, threshold=5000):
        """Simple motion-based anomaly detection."""
        if prev_frame is None:
            return False, 0
        
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_score = np.sum(thresh) / 255
        
        return motion_score > threshold, motion_score
