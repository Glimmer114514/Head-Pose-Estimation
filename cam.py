import cv2
import mediapipe as mp
import numpy as np

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        gpu_options = {}
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_faces=1,
                static_image_mode=False,
                gpu_options=mp.GpuOptions()
            )
            print("MediaPipe GPU acceleration enabled")
        except:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_faces=1,
                static_image_mode=False
            )
            print("MediaPipe using CPU (GPU not available)")
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))
        self.translation_scale = 0.1
        self.gpu_enabled = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("OpenCV CUDA GPU acceleration enabled")
                self.gpu_enabled = True
            else:
                print("OpenCV using CPU (CUDA not available)")
        except:
            print("OpenCV GPU check failed, using CPU")
    
    def estimate_pose(self, frame):
        height, width, _ = frame.shape
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype=np.float64
        )
        if self.gpu_enabled:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                rgb_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = rgb_gpu.download()
            except Exception as e:
                print(f"GPU processing failed: {e}")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_tip = 1
            chin = 199
            left_eye = 33
            right_eye = 263
            left_mouth = 61
            right_mouth = 291
            image_points = np.array([
                self.get_landmark_point(face_landmarks, nose_tip, width, height),
                self.get_landmark_point(face_landmarks, chin, width, height),
                self.get_landmark_point(face_landmarks, left_eye, width, height),
                self.get_landmark_point(face_landmarks, right_eye, width, height),
                self.get_landmark_point(face_landmarks, left_mouth, width, height),
                self.get_landmark_point(face_landmarks, right_mouth, width, height)
            ], dtype=np.float64)
            _, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            scaled_translation = translation_vector * self.translation_scale
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
            return euler_angles, scaled_translation.flatten()
        return None, None
    
    def get_landmark_point(self, landmarks, index, width, height):
        landmark = landmarks.landmark[index]
        return (int(landmark.x * width), int(landmark.y * height))
    
    def rotation_matrix_to_euler_angles(self, R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])
    
    def release(self):
        self.face_mesh.close()
