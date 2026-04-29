import sys
import cv2
import json
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QLabel, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

class CameraRecorder(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.setWindowTitle("RTSP Robust Recorder + Cleanup")
        self.resize(1300, 750)
        
        self.config_path = Path(config_path)
        self.cameras = self.load_cameras()
        self.current_idx = 0
        
        # State
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.consecutive_failures = 0
        self.recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
        self.recordings_dir.mkdir(exist_ok=True)
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        if self.cameras:
            self.switch_camera(0)
        else:
            self.display_label.setText("No cameras found in config.\nRun scanner.py or check cameras.json")

    def load_cameras(self):
        if not self.config_path.exists(): return []
        try:
            with open(self.config_path, 'r') as f: return json.load(f)
        except: return []

    def save_cameras(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.cameras, f, indent=4)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        
        header = QHBoxLayout()
        self.cam_info = QLabel("Ready")
        self.cam_info.setStyleSheet("color: #60a5fa; font-size: 16px; font-weight: bold;")
        header.addWidget(self.cam_info)
        header.addStretch()
        self.rec_status = QLabel("● IDLE")
        self.rec_status.setStyleSheet("color: #94a3b8; font-size: 14px;")
        header.addWidget(self.rec_status)
        layout.addLayout(header)
        
        self.display_label = QLabel("Initializing...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background: #000000; border: 2px solid #1e293b; border-radius: 4px;")
        layout.addWidget(self.display_label)
        
        footer = QHBoxLayout()
        self.hint_label = QLabel("[A/D] Switch  |  [E] REC  |  [DEL] Delete Invalid URL  |  [Esc] Exit")
        self.hint_label.setStyleSheet("color: #475569; font-size: 11px;")
        footer.addWidget(self.hint_label)
        footer.addStretch()
        self.count_label = QLabel(f"Streams: {len(self.cameras)}")
        self.count_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        footer.addWidget(self.count_label)
        layout.addLayout(footer)
        
        central.setLayout(layout)
        self.setStyleSheet("background-color: #020617;")

    def switch_camera(self, idx):
        if not self.cameras:
            self.display_label.setText("No cameras remaining.")
            self.cam_info.setText("Empty Config")
            return
            
        self.stop_recording()
        if self.cap: self.cap.release()
        
        self.current_idx = idx % len(self.cameras)
        cam = self.cameras[self.current_idx]
        
        self.cap = cv2.VideoCapture(cam['url'])
        self.consecutive_failures = 0
        self.count_label.setText(f"Index: {self.current_idx+1}/{len(self.cameras)} | Total Streams: {len(self.cameras)}")
        
        if not self.cap.isOpened():
            self.cam_info.setText(f"OFFLINE: {cam['ip']} ({cam['alias']})")
            self.cam_info.setStyleSheet("color: #ef4444;")
        else:
            self.cam_info.setText(f"LIVE: {cam['ip']} [{cam['alias']}]")
            self.cam_info.setStyleSheet("color: #10b981;")
            self.timer.start(30)

    def delete_current_camera(self):
        if not self.cameras: return
        removed = self.cameras.pop(self.current_idx)
        print(f"Removed invalid URL: {removed['url']}")
        self.save_cameras()
        
        if self.cameras:
            self.switch_camera(self.current_idx) # Switch to same index (which is now next item)
        else:
            self.display_label.setText("All streams deleted.")
            self.cam_info.setText("Config Empty")
            self.count_label.setText("Streams: 0")

    def toggle_recording(self):
        if not self.is_recording: self.start_recording()
        else: self.stop_recording()

    def start_recording(self):
        if not self.cap or not self.cap.isOpened(): return
        cam = self.cameras[self.current_idx]
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"rec_{cam['ip'].replace('.', '_')}_{ts}.mp4"
        sp = self.recordings_dir / fn
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        self.writer = cv2.VideoWriter(str(sp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.is_recording = True
        self.rec_status.setText("● RECORDING")
        self.rec_status.setStyleSheet("color: #ef4444; font-weight: bold;")

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        if self.writer: self.writer.release(); self.writer = None
        self.rec_status.setText("● IDLE")
        self.rec_status.setStyleSheet("color: #94a3b8;")

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            self._draw_status("NO SIGNAL")
            return
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.consecutive_failures = 0
            if self.is_recording and self.writer: self.writer.write(frame)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.display_label.width()-4, self.display_label.height()-4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.display_label.setPixmap(pix)
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures > 5: self._draw_status("SIGNAL LOST")

    def _draw_status(self, text):
        pix = QPixmap(self.display_label.size()); pix.fill(QColor("#000000"))
        p = QPainter(pix); p.setPen(QColor("#ef4444")); p.setFont(QFont("Arial", 20, QFont.Bold))
        p.drawText(pix.rect(), Qt.AlignCenter, text); p.end()
        self.display_label.setPixmap(pix)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape: self.stop_recording(); self.close()
        elif key == Qt.Key_D: self.switch_camera(self.current_idx + 1)
        elif key == Qt.Key_A: self.switch_camera(self.current_idx - 1)
        elif key == Qt.Key_E: self.toggle_recording()
        elif key == Qt.Key_Delete: self.delete_current_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cfg = Path(__file__).resolve().parent / "cameras.json"
    tool = CameraRecorder(cfg)
    tool.show()
    sys.exit(app.exec_())
