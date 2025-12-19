import cv2
import numpy as np
from typing import Tuple, Optional
import threading
import time
import os

class SmileDetector:
    def __init__(self):
        # تحميل مصنفات Haar Cascades
        self.face_cascade = cv2.CascadeClassifier()
        self.smile_cascade = cv2.CascadeClassifier()
        
        # Paths to try (Standard + Local Fallback)
        face_names = ['haarcascade_frontalface_default.xml']
        smile_names = ['haarcascade_smile.xml']
        
        search_dirs = [
            cv2.data.haarcascades,
            "assets/cascades",
            "cascades",
            "."
        ]
        
        # Load Face
        loaded_face = False
        for d in search_dirs:
            p = os.path.join(d, face_names[0])
            if os.path.exists(p) and self.face_cascade.load(p):
                print(f"Loaded face cascade from: {p}")
                loaded_face = True
                break
        
        # Load Smile
        loaded_smile = False
        for d in search_dirs:
            p = os.path.join(d, smile_names[0])
            if os.path.exists(p) and self.smile_cascade.load(p):
                print(f"Loaded smile cascade from: {p}")
                loaded_smile = True
                break

        if not loaded_face or not loaded_smile:
            print("WARNING: Cascades failed to load. Smile detection will not work.")
            
        self.is_smiling = False
        self.smile_count = 0 
        
    def detect_smile(self, frame) -> Tuple[bool, float, Optional[np.ndarray]]:
        """الكشف عن الابتسامة في الإطار باستخدام Haar Cascades مع دعم للمحاولات المتعددة"""
        try:
            if frame is None:
                return False, 0.0, None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scaleFactor=1.2 for faster/more tolerant face detection
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            self.is_smiling = False
            smile_ratio = 0.0
            annotated_frame = frame.copy()
            
            for (x, y, w, h) in faces:
                # رسم مربع الوجه
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # منطقة البحث عن الفم (65% من أسفل الوجه وموسعة عرضياً)
                roi_y = y + int(h * 0.5)
                roi_h = int(h * 0.5)
                roi_x = x + int(w * 0.1)
                roi_w = int(w * 0.8)
                
                roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                roi_color = annotated_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # رسم مربع منطقة الفم (Cyan) للمساعدة البصرية
                cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 0), 1)

                # محاولة 1: معاملات عادية (Medium)
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.4, 10, minSize=(20, 20))
                
                # محاولة 2: إذا لم يجد شيئاً، نجرب معاملات أكثر مرونة (Lenient)
                if len(smiles) == 0:
                    smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.2, 3, minSize=(15, 15))

                if len(smiles) > 0:
                    self.is_smiling = True
                    smile_ratio = 2.0 # قيمة كافية لتجاوز عتبة الـ UI
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                
                status_text = "FACE FOUND! SMILE!" if not self.is_smiling else "SMILE DETECTED! 😊"
                color = (0, 255, 0) if self.is_smiling else (255, 255, 0)
                cv2.putText(annotated_frame, status_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return self.is_smiling, smile_ratio, annotated_frame
            
        except Exception as e:
            print(f"Error in Multi-Pass smile detection: {e}")
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