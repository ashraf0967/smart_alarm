import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
import threading
import time
import os

class SmileDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # نقاط الوجه للابتسامة
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.is_smiling = False
        self.smile_threshold = 0.3
        
    def calculate_smile_ratio(self, landmarks, image_width, image_height) -> float:
        """حساب نسبة الابتسامة بناءً على نقاط الشفاه"""
        try:
            # نقاط الشفاه العلوية والسفلية
            upper_lip = [landmarks[13], landmarks[14]]
            lower_lip = [landmarks[17], landmarks[18]]
            lip_corners = [landmarks[61], landmarks[291]]
            
            # حساب المسافات
            lip_height = abs(upper_lip[0].y * image_height - lower_lip[0].y * image_height)
            lip_width = abs(lip_corners[0].x * image_width - lip_corners[1].x * image_width)
            
            if lip_width == 0:
                return 0.0
                
            smile_ratio = lip_height / lip_width
            return smile_ratio
            
        except Exception as e:
            print(f"Error calculating smile ratio: {e}")
            return 0.0
    
    def detect_smile(self, frame) -> Tuple[bool, float, Optional[np.ndarray]]:
        """الكشف عن الابتسامة في الإطار"""
        try:
            # تحويل BGR إلى RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # معالجة الإطار
            results = self.face_mesh.process(rgb_frame)
            
            # إعادة تمكين الكتابة للرسم
            rgb_frame.flags.writeable = True
            
            smile_ratio = 0.0
            annotated_frame = None
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # رسم نقاط الوجه
                    annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    self.mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec
                    )
                    
                    # حساب نسبة الابتسامة
                    smile_ratio = self.calculate_smile_ratio(
                        face_landmarks.landmark,
                        annotated_frame.shape[1],
                        annotated_frame.shape[0]
                    )
                    
                    # التحقق من الابتسامة
                    self.is_smiling = smile_ratio > self.smile_threshold
                    
                    # إضافة معلومات على الإطار
                    cv2.putText(annotated_frame, 
                              f"Smile Ratio: {smile_ratio:.2f}", 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, 
                              (0, 255, 0) if self.is_smiling else (0, 0, 255), 
                              2)
                    
                    if self.is_smiling:
                        cv2.putText(annotated_frame, 
                                  "SMILING! 😊", 
                                  (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, 
                                  (0, 255, 0), 
                                  2)
            
            return self.is_smiling, smile_ratio, annotated_frame
            
        except Exception as e:
            print(f"Error in smile detection: {e}")
            return False, 0.0, None
    
    def release(self):
        """تحرير الموارد"""
        self.face_mesh.close()

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
            self.current_frame = None # Reset frame
            
            # خيط لقراءة الإطارات
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
                        print("Failed to grab frame")
                        time.sleep(1) # Wait before retry
                except Exception as e:
                    print(f"Camera read error: {e}")
                    time.sleep(1)
            else:
                time.sleep(1)
            time.sleep(0.03)  # ~30 إطار في الثانية
    
    def get_frame(self):
        """الحصول على الإطار الحالي"""
        return self.current_frame
    
    def stop_camera(self):
        """إيقاف الكاميرا"""
        self.is_camera_active = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()