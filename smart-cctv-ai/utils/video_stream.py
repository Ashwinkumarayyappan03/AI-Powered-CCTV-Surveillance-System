# utils/video_stream.py
import cv2
import threading
import time

class VideoStream:
    def __init__(self, source=0, target_fps=30):
        self.source = source
        self.target_fps = target_fps
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.fps = 0
        self._fps_counter = 0
        self._fps_time = time.time()
        
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")
        
        # Optimize capture
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self._update_fps()
            else:
                time.sleep(0.01)
    
    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = time.time()
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
