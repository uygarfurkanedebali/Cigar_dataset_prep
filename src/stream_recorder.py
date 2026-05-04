import sys
import cv2
import time
import uuid
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

class StreamRecorder(QMainWindow):
    def __init__(self, stream_url):
        super().__init__()
        self.setWindowTitle(f"Stream Recorder - {stream_url}")
        self.resize(1280, 720)

        self.stream_url = stream_url
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.consecutive_failures = 0

        self.recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
        self.recordings_dir.mkdir(exist_ok=True)

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.connect_stream()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        # Header
        header = QHBoxLayout()
        self.stream_info = QLabel("Connecting...")
        self.stream_info.setStyleSheet("color: #60a5fa; font-size: 14px; font-weight: bold;")
        header.addWidget(self.stream_info)
        header.addStretch()
        self.rec_status = QLabel("● IDLE")
        self.rec_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        header.addWidget(self.rec_status)
        layout.addLayout(header)

        # Video display
        self.display_label = QLabel("Connecting to stream...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background: #000000; border: 2px solid #1e293b; border-radius: 4px;")
        layout.addWidget(self.display_label)

        # Footer
        footer = QHBoxLayout()
        self.hint_label = QLabel("[E] Record/Stop  |  [Esc] Exit")
        self.hint_label.setStyleSheet("color: #475569; font-size: 11px;")
        footer.addWidget(self.hint_label)
        footer.addStretch()
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        footer.addWidget(self.stats_label)
        layout.addLayout(footer)

        central.setLayout(layout)
        self.setStyleSheet("background-color: #020617;")

    def connect_stream(self):
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            self.stream_info.setText(f"❌ Cannot connect to stream")
            self.stream_info.setStyleSheet("color: #ef4444;")
            self._draw_status("STREAM UNAVAILABLE")
            return

        # Get stream properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Default if not available
        if fps <= 0:
            fps = 30

        self.fps = fps
        self.width = width
        self.height = height

        self.stream_info.setText(f"🟢 LIVE | {width}x{height} @ {fps:.1f}fps")
        self.stream_info.setStyleSheet("color: #10b981;")
        self.stats_label.setText(f"Resolution: {width}x{height} | FPS: {fps:.1f}")

        self.consecutive_failures = 0
        self.timer.start(int(1000 / fps))  # Update at stream's FPS

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.cap or not self.cap.isOpened():
            return

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{ts}_{unique_id}.mp4"
        filepath = self.recordings_dir / filename

        # Create video writer with stream properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (self.width, self.height))

        self.is_recording = True
        self.current_file = filename
        self.rec_status.setText("● RECORDING")
        self.rec_status.setStyleSheet("color: #ef4444; font-weight: bold;")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None

        self.rec_status.setText("● IDLE")
        self.rec_status.setStyleSheet("color: #94a3b8;")
        print(f"✓ Saved: {self.current_file}")

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            self._draw_status("NO SIGNAL")
            return

        ret, frame = self.cap.read()

        if ret and frame is not None:
            self.consecutive_failures = 0

            # Write to file if recording
            if self.is_recording and self.writer:
                self.writer.write(frame)

            # Display on screen
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.display_label.width() - 4,
                self.display_label.height() - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.display_label.setPixmap(pix)
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures > 5:
                self._draw_status("SIGNAL LOST")

    def _draw_status(self, text):
        pix = QPixmap(self.display_label.size())
        pix.fill(QColor("#000000"))
        p = QPainter(pix)
        p.setPen(QColor("#ef4444"))
        p.setFont(QFont("Arial", 20, QFont.Bold))
        p.drawText(pix.rect(), Qt.AlignCenter, text)
        p.end()
        self.display_label.setPixmap(pix)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.stop_recording()
            self.close()
        elif key == Qt.Key_E:
            self.toggle_recording()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stream_recorder.py <RTSP_URL>")
        print("Example: python stream_recorder.py rtsp://192.168.1.100:554/stream")
        sys.exit(1)

    stream_url = sys.argv[1]
    app = QApplication(sys.argv)
    recorder = StreamRecorder(stream_url)
    recorder.show()
    sys.exit(app.exec_())
