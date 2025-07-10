import sys
import cv2
import time
import threading
from PyQt5 import QtWidgets, QtCore, QtGui
from cam import HeadPoseDetector
from show import HeadModelViewer
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("头部追踪预览")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                background-color: #2d2d2d;
                color: white;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                min-width: 100px;
                font-family: Arial;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """)
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("background-color: black; border-radius: 8px;")
        self.viewer = HeadModelViewer()
        btn_layout = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("暂停")
        self.pause_btn.setIcon(QtGui.QIcon.fromTheme("media-playback-pause"))
        self.resume_btn = QtWidgets.QPushButton("继续")
        self.resume_btn.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        self.exit_btn = QtWidgets.QPushButton("退出")
        self.exit_btn.setIcon(QtGui.QIcon.fromTheme("application-exit"))
        self.switch_cam_btn = QtWidgets.QPushButton("切换摄像头")
        self.switch_cam_btn.setIcon(QtGui.QIcon.fromTheme("camera-switch"))
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.switch_cam_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addStretch()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.camera_label, stretch=1)
        layout.addWidget(self.viewer, stretch=2)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(btn_layout)
        main_widget.setLayout(main_layout)
        self.exit_btn.clicked.connect(self.close)
        self.pause_btn.clicked.connect(self.pause_camera)
        self.resume_btn.clicked.connect(self.resume_camera)
        self.switch_cam_btn.clicked.connect(self.switch_camera)
        self.camera_index = 1
        self.running = True
        self.paused = False
        self.detector = HeadPoseDetector()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        cap = None
        target_fps = 60
        min_frame_interval = 1.0 / target_fps
        last_time = time.time()
        while self.running:
            if getattr(self, 'cap_released', False) or cap is None:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(self.camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap_released = False
                last_time = time.time()
                min_frame_interval = 1.0 / target_fps
            
            if not self.paused:
                current_time = time.time()
                if current_time - last_time < min_frame_interval:
                    time.sleep(0.001)
                    continue
                
                ret, frame = cap.read()
                if ret:
                    estimated_angles, translation = self.detector.estimate_pose(frame)
                    if estimated_angles is not None and translation is not None:
                        try:
                            self.viewer.update_pose(
                                estimated_angles[0],
                                estimated_angles[1],
                                estimated_angles[2],
                                translation[0],
                                translation[1],
                                translation[2]
                            )
                        except Exception as e:
                            print(f"Update error: {e}")
                    self.update_camera_frame(cv2.resize(frame, (320, 240)))
                    last_time = current_time
            
            time.sleep(0.001)
        cap.release()
    
    def pause_camera(self):
        self.paused = True
    
    def resume_camera(self):
        self.paused = False
    
    def switch_camera(self):
        self.camera_index = 0 if self.camera_index == 1 else 1
        print(f"切换到摄像头 {self.camera_index}")
        self.cap_released = True

    def closeEvent(self, event):
        self.running = False
        self.detector.release()
        event.accept()
    
    def update_camera_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())