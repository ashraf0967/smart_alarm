import cv2
import numpy as np
from typing import Tuple, Optional
import threading
import time
import os

class SmileDetector:
    def __init__(self):
        # تحميل مصنفات Haar Cascades
        # OpenCV provides these XML files within the package data
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        if self.face_cascade.empty() or self.smile_cascade.empty():
            print("Warning: Haar cascade files not found or failed to load!")
            
        self.is_smiling = False
        self.smile_threshold = 1.5 # Boolean indicator for simple detection
        
    def detect_smile(self, frame) -> Tuple[bool, float, Optional[np.ndarray]]:
        """الكشف عن الابتسامة في الإطار باستخدام Haar Cascades"""
        try:
            if frame is None:
                return False, 0.0, None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.is_smiling = False
            smile_ratio = 0.0
            annotated_frame = frame.copy()
            
            for (x, y, w, h) in faces:
                # رسم مربع حول الوجه
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = annotated_frame[y:y+h, x:x+w]
                
                # البحث عن الابتسامة داخل منطقة الوجه (في النصف السفلي عادة)
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
                
                if len(smiles) > 0:
                    self.is_smiling = True
                    smile_ratio = 1.0 
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                
                # إضافة معلومات على الإطار
                status_text = "SMILING! 😊" if self.is_smiling else "Detecting..."
                color = (0, 255, 0) if self.is_smiling else (0, 0, 255)
                cv2.putText(annotated_frame, status_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return self.is_smiling, smile_ratio, annotated_frame
            
        except Exception as e:
            print(f"Error in Haar smile detection: {e}")
            return False, 0.0, frame
    
    def release(self):
        """تحرير الموارد"""
        pass

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_camera_active = False
        self.current_frame = None
        
    def start_camera(self, camera_index=0):
        """تشغيل الكاميرا"""
        try:
            # Use DirectShow on Windows for better stability
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.camera = cv2.VideoCapture(camera_index, backend)
            
            if not self.camera.isOpened():
                # تجربة كاميرا افتراضية بدون backend محدد
                self.camera = cv2.VideoCapture(0)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_camera_active = True
            self.current_frame = None 
            
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def _read_frames(self):
        """قراءة الإطارات من الكاميرا"""
        while self.is_camera_active:
            if self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.current_frame = frame
                    else:
                        time.sleep(1) 
                except Exception as e:
                    print(f"Camera read error: {e}")
                    time.sleep(1)
            else:
                time.sleep(1)
            time.sleep(0.03)  
    
    def get_frame(self):
        """الحصول على الإطار الحالي"""
        return self.current_frame
    
    def stop_camera(self):
        """إيقاف الكاميرا"""
        self.is_camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()