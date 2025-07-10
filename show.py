import threading
import queue

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PyQt5 import QtWidgets, QtCore

class HeadModelViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = pv.read('HEAD.obj')
        self.mesh.rotate_y(90, inplace=True)
        self.mesh.rotate_z(180, inplace=True)
        self.mesh.rotate_x(90, inplace=True)
        self.figure = Figure(figsize=(8, 6))
        self.figure.set_facecolor('black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.ax.xaxis.pane.set_facecolor((0, 0, 0))
        self.ax.yaxis.pane.set_facecolor((0, 0, 0))
        self.ax.zaxis.pane.set_facecolor((0, 0, 0))
        self.ax.grid(True, color='gray')
        self.init_ui()
    
    def init_ui(self):
        text_layout = QtWidgets.QVBoxLayout()
        
        self.angle_label = QtWidgets.QLabel("角度:\nX: 0.0°\nY: 0.0°\nZ: 0.0°")
        self.angle_label.setStyleSheet("color: green; font-family: Arial; font-size: 20px;")
        self.angle_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        self.translation_label = QtWidgets.QLabel("位置:\nX: 0.0\nY: 0.0\nZ: 0.0")
        self.translation_label.setStyleSheet("color: green; font-family: Arial; font-size: 20px;")
        self.translation_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        text_layout.addWidget(self.angle_label)
        text_layout.addWidget(self.translation_label)
        text_layout.addStretch()
        
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(text_layout)
        self.setLayout(main_layout)
        
        points = self.mesh.points
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        scale = np.max(max_coords - min_coords)
        self.normalized_points = (points - min_coords) / scale - 0.5
        
        self.original_vertices = self.normalized_points.copy()
        
        triangles = []
        faces = self.mesh.faces
        index = 0
        while index < len(faces):
            n_vertices = faces[index]
            index += 1
            if n_vertices == 3:
                triangles.append(faces[index:index+3])
                index += 3
            elif n_vertices == 4:
                v0, v1, v2, v3 = faces[index:index+4]
                triangles.append([v0, v1, v2])
                triangles.append([v0, v2, v3])
                index += 4
            else:
                triangles.append(faces[index:index+3])
                index += n_vertices
        
        self.triangles = np.array(triangles)
        
        self.verts = self.original_vertices[self.triangles]
        self.collection = Poly3DCollection(
            self.verts,
            facecolors='black',
            edgecolors='green',
            linewidths=0.3,
            alpha=0.9
        )
        self.ax.add_collection3d(self.collection)
        
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        
        self.tx = 0
        self.ty = 0
        self.tz = 0
        
        self.lock = threading.Lock()
        
        self.running = True
        
        self.update_queue = queue.Queue()

    @QtCore.pyqtSlot(float, float, float, float, float, float)
    def update_pose(self, pitch, yaw, roll, tx, ty, tz):
        self.update_queue.put((
            pitch, yaw, roll,
            tx * 0.1,
            -ty * 0.1,
            -tz * 0.1
        ))
        self.update_display()
    
    def update_display(self):
        try:
            while True:
                pitch, yaw, roll, tx, ty, tz = self.update_queue.get_nowait()
                with self.lock:
                    self.pitch = pitch
                    self.yaw = yaw
                    self.roll = roll
                    self.tx = tx
                    self.ty = ty
                    self.tz = tz
        except queue.Empty:
            with self.lock:
                pitch = self.pitch
                yaw = self.yaw
                roll = self.roll
                tx = self.tx
                ty = self.ty
                tz = self.tz

        self.angle_label.setText(
            f"角度:\nX: {pitch:.1f}°\nY: {yaw:.1f}°\nZ: {roll:.1f}°"
        )

        self.translation_label.setText(
            f"位置:\nX: {tx:.3f}\nY: {ty:.3f}\nZ: {tz:.3f}"
        )

        moved_vertices = self.original_vertices.copy()
        moved_vertices += np.array([tx, ty, tz])

        new_verts = moved_vertices[self.triangles]
        self.collection.set_verts(new_verts)

        self.ax.view_init(elev=pitch, azim=-yaw)

        padding = 0.3
        self.ax.set_xlim(tx - 0.8, tx + 0.8)
        self.ax.set_ylim(ty - 0.8, ty + 0.8)
        self.ax.set_zlim(tz - 0.8, tz + 0.8)

        self.canvas.draw_idle()
    
    def start(self):
        self.timer = self.figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self.update_display)
        self.timer.start()
        
    def close(self):
        self.running = False