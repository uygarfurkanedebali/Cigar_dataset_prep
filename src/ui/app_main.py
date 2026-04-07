import sys
import os
import cv2
import numpy as np
import csv
import shutil
from pathlib import Path
try:
    from .utils import PoseNormalizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import PoseNormalizer

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QDesktopWidget, QSizePolicy, QDialog, QTextEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class SettingsDialog(QDialog):
    """A dialog to edit labels.txt line by line."""
    def __init__(self, labels_file, parent=None):
        super().__init__(parent)
        self.labels_file = labels_file
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Label Settings")
        self.setMinimumSize(300, 400)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("Enter labels (one per line):")
        layout.addWidget(self.label)
        
        self.text_edit = QTextEdit()
        # Load current labels
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                self.text_edit.setPlainText(f.read())
        else:
            self.text_edit.setPlainText("1\n2\n3\n4")
            
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_and_close)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)

    def save_and_close(self):
        content = self.text_edit.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Warning", "Labels cannot be empty!")
            return
            
        try:
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save labels: {e}")


# COCO 17 keypoint skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6),                              # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]


class AnnotationWindow(QMainWindow):
    """Main Application Window for Annotation Phase 2."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Dataset Annotation Tool")
        self.resize(1200, 800)
        
        self.image_pairs = []
        self.current_index = -1
        self.labels_state = {}
        self.annotations_file = None
        self.category_file = Path("labels.txt").resolve()
        self.category_names = []
        
        self.cached_main_pixmap = None
        self.cached_head_pixmap = None
        
        # 0. Anchor Paths (Absolute Project Root)
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = (self.base_dir / "data" / "raw").resolve()
        self.processed_dir = (self.base_dir / "data" / "processed").resolve()
        self.txt_dir = (self.base_dir / "data" / "txt").resolve()
        
        # Ensure base directories exist immediately
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.txt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Folder Watcher
        self.watcher = QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self.load_dataset)
        
        self.init_ui()
        self.center_window()
        self.load_categories()
        self.load_dataset()

    def center_window(self):
        """Center the main window on the screen."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        """Initialize user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        # 1. Main Image Label (Replaces QFrame)
        self.main_image_label = QLabel("Main Image (Resized)")
        self.main_image_label.setAlignment(Qt.AlignCenter)
        self.main_image_label.setStyleSheet("background-color: #e0e0e0; border: 2px solid #555;")
        self.main_image_label.setMinimumSize(1, 1)
        self.main_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        top_layout.addWidget(self.main_image_label, stretch=2)
        
        right_layout = QVBoxLayout()
        
        # 2. Head Crop Label
        self.head_crop_label = QLabel("Head Crop")
        self.head_crop_label.setAlignment(Qt.AlignCenter)
        self.head_crop_label.setStyleSheet("background-color: #e0e0e0; border: 2px solid #555;")
        self.head_crop_label.setMinimumSize(1, 1)
        self.head_crop_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_layout.addWidget(self.head_crop_label, stretch=1)
        
        # 3. Keypoint Plot Canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.canvas, stretch=1)
        
        top_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(top_layout, stretch=1)
        
        # Bottom Status Label
        self.status_label = QLabel("File: None | Label: None")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px; color: #333;")
        main_layout.addWidget(self.status_label)
        
        central_widget.setLayout(main_layout)

    def load_dataset(self):
        """Find images in raw and processed subfolders using absolute anchors."""
        # Watch raw directory if not already
        if self.raw_dir.as_posix() not in self.watcher.directories():
            self.watcher.addPath(str(self.raw_dir))

        # Store current file to try and preserve index
        current_file = None
        if 0 <= self.current_index < len(self.image_pairs):
            current_file = self.image_pairs[self.current_index][0].name

        # 1. Reset Internal State (Load CSV only if labels_state is empty)
        if not self.labels_state:
            self.labels_state = {}
            self.annotations_file = self.processed_dir / "labels.csv"
            if self.annotations_file.exists():
                try:
                    with open(self.annotations_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            img_name = row.get('image_path') or row.get('image_name')
                            if img_name:
                                self.labels_state[img_name] = row
                except Exception as e:
                    print(f"Error loading annotations: {e}")

        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        # 2. Collect Images from multiple sources
        found_images = {}
        
        # Scan raw (unlabeled)
        if self.raw_dir.exists():
            for img_path in self.raw_dir.iterdir():
                if img_path.suffix.lower() in supported_exts:
                    found_images[img_path.name] = img_path

        # Scan processed subfolders (labeled)
        if self.processed_dir.exists():
            for sub in self.processed_dir.iterdir():
                # SAFETY: explicitly skip any subfolder named 'processed' to recover from prev bug
                if sub.is_dir() and sub.name.lower() != "processed":
                    for img_path in sub.iterdir():
                        if img_path.suffix.lower() in supported_exts:
                            # Prefer current location found (avoids dupes during move)
                            found_images[img_path.name] = img_path

        self.image_pairs = []
        for name, img_path in found_images.items():
            # YOLO txt files are now kept in data/txt/ as per pose_inference.py
            txt_path = self.txt_dir / f"{Path(name).stem}.txt"
            self.image_pairs.append((img_path, txt_path))
        
        # Sort to keep order consistent
        self.image_pairs.sort(key=lambda x: x[0].name)

        if not self.image_pairs:
            print("No matching image/label pairs found. Waiting for images...")
            self.current_index = -1
            self.status_label.setText("No images found in data/raw or data/processed. Add images to start.")
            return

        # Restore index if possible, otherwise default to 0 or preserve current index
        new_index = 0
        if current_file:
            for i, (path, _) in enumerate(self.image_pairs):
                if path.name == current_file:
                    new_index = i
                    break
        
        self.current_index = new_index
        print(f"Dataset loaded. {len(self.image_pairs)} images total in unified queue.")
        self.update_ui()

    def load_categories(self):
        """Load label names from labels.txt."""
        if not self.category_file.exists():
            # Create default labels.txt
            with open(self.category_file, 'w', encoding='utf-8') as f:
                f.write("1\n2\n3\n4")
        
        with open(self.category_file, 'r', encoding='utf-8') as f:
            self.category_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded categories: {self.category_names}")

    def open_settings(self):
        """Open the label settings dialog."""
        dialog = SettingsDialog(self.category_file, self)
        if dialog.exec_() == QDialog.Accepted:
            self.load_categories()
            self.update_ui() # Refresh status bar with new names

    def save_annotations(self):
        """Save the in-memory labels state to the CSV file with normalized coordinates."""
        if not self.annotations_file:
            return
            
        fieldnames = ['label']
        for name in PoseNormalizer.kp_names:
            fieldnames.extend([f"{name}_x", f"{name}_y"])
        fieldnames.append('image_path')
        
        try:
            with open(self.annotations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for img_name, data in self.labels_state.items():
                    if isinstance(data, str):
                        # Handle migration from old format
                        row = {'label': data, 'image_path': img_name}
                        for fn in fieldnames:
                            if fn not in row:
                                row[fn] = 0.0
                    else:
                        row = data
                        # Ensure image_path is set correctly if it was image_name
                        if 'image_name' in row and 'image_path' not in row:
                            row['image_path'] = row['image_name']
                        
                        # Ensure all fieldnames exist in row
                        for fn in fieldnames:
                            if fn not in row:
                                row[fn] = 0.0
                    
                    # Filter keys to match fieldnames exactly
                    filtered_row = {k: row[k] for k in fieldnames if k in row}
                    writer.writerow(filtered_row)
        except Exception as e:
            print(f"Error saving annotations: {e}")

    def parse_yolo_txt(self, txt_path, img_width, img_height):
        """Parse the first YOLO label in a .txt file."""
        if not txt_path or not txt_path.exists():
            return None
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            parts = lines[0].strip().split()
            
        if len(parts) < 5:
            return None
            
        kps = []
        # Keypoints start at index 5 in YOLOv8 Pose format
        for i in range(5, len(parts), 3):
            if i + 2 < len(parts):
                px = float(parts[i]) * img_width
                py = float(parts[i+1]) * img_height
                conf = float(parts[i+2])
                kps.append((px, py, conf))
        return kps

    def cvimg_to_qpixmap(self, cv_img):
        """Convert an OpenCV image to a QPixmap."""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def update_ui(self):
        """Load image logic, compute caching, and draw keypoint plotting."""
        if self.current_index < 0 or self.current_index >= len(self.image_pairs):
            return
            
        img_path, txt_path = self.image_pairs[self.current_index]
        # Map label_id (string or dict) to descriptive name if available
        current_data = self.labels_state.get(img_path.name, None)
        label_id = current_data.get('label') if isinstance(current_data, dict) else current_data
        
        current_display = "None"
        if label_id:
            try:
                idx = int(label_id) - 1
                if 0 <= idx < len(self.category_names):
                    current_display = self.category_names[idx]
                else:
                    current_display = label_id
            except (ValueError, TypeError):
                current_display = str(label_id)

        self.status_label.setText(f"File: {img_path.name} | Label: {current_display}")
        
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            print(f"Failed to load image: {img_path}")
            return
            
        img_h, img_w = cv_img.shape[:2]
        kps = self.parse_yolo_txt(txt_path, img_w, img_h)
        
        self.cached_head_pixmap = None
        
        if kps:
            # 1. New Head Crop Logic (Shoulders Up)
            kp5 = kps[5] if len(kps) > 5 else None
            kp6 = kps[6] if len(kps) > 6 else None
            
            # Use average shoulder Y as bottom boundary if shoulders exist
            bottom_found = False
            y_max = 0
            shoulder_width = 40
            if kp5 and kp6 and kp5[2] > 0 and kp6[2] > 0:
                y_max = (kp5[1] + kp6[1]) / 2
                shoulder_width = abs(kp5[0] - kp6[0])
                bottom_found = True
            elif kp5 and kp5[2] > 0:
                y_max = kp5[1]
                bottom_found = True
            elif kp6 and kp6[2] > 0:
                y_max = kp6[1]
                bottom_found = True

            head_kps = [pt for pt in kps[:5] if pt[2] > 0]
            if head_kps and bottom_found:
                xs_h, ys_h, _ = zip(*head_kps)
                y_highest = min(ys_h)
                dist_to_top = y_max - y_highest
                if dist_to_top <= 0: dist_to_top = 20
                
                # Skull top padding (50% of face height)
                y_min = max(0, y_highest - 0.5 * dist_to_top)
                
                # Adaptive width: max of shoulders or ears
                ear_width = 0
                if kps[3][2] > 0 and kps[4][2] > 0:
                    ear_width = abs(kps[3][0] - kps[4][0])
                
                crop_w = max(shoulder_width, ear_width) * 1.6 # generous padding
                center_x = sum(xs_h) / len(xs_h)
                
                x1 = max(0, int(center_x - crop_w/2))
                x2 = min(img_w, int(center_x + crop_w/2))
                y1 = int(y_min)
                y2 = int(y_max)
                
                if y2 > y1 and x2 > x1:
                    head_crop = cv_img[y1:y2, x1:x2].copy()
                    self.cached_head_pixmap = self.cvimg_to_qpixmap(head_crop)
                    self.head_crop_label.setText("")
            elif head_kps:
                # Fallback to standard bounding box if shoulders are missing
                xs, ys, _ = zip(*head_kps)
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                w, h = max_x - min_x, max_y - min_y
                if w == 0: w = 20
                if h == 0: h = 20
                pad_x, pad_y = w * 0.3, h * 0.3
                x1 = max(0, int(min_x - pad_x))
                y1 = max(0, int(min_y - pad_y))
                x2 = min(img_w, int(max_x + pad_x))
                y2 = min(img_h, int(max_y + pad_y))
                if y2 > y1 and x2 > x1:
                    head_crop = cv_img[y1:y2, x1:x2].copy()
                    self.cached_head_pixmap = self.cvimg_to_qpixmap(head_crop)
                    self.head_crop_label.setText("")
            
            # 2. Plot Normalized Keypoints and Skeleton with Matplotlib
            self.canvas.axes.clear()
            
            # Use PoseNormalizer to get Min-Max normalized points
            norm_data = PoseNormalizer.normalize_keypoints(kps)
            
            # Map indices to normalized (x, y)
            norm_kps_list = []
            for name in PoseNormalizer.kp_names:
                norm_kps_list.append((norm_data[f"{name}_x"], norm_data[f"{name}_y"]))
            
            # Draw skeleton lines using normalized coordinates
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                p1 = norm_kps_list[start_idx]
                p2 = norm_kps_list[end_idx]
                if p1 != (0.0, 0.0) and p2 != (0.0, 0.0):
                    self.canvas.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=1, alpha=0.7)
            
            # Plot dots for valid points
            valid_xs = [p[0] for p in norm_kps_list if p != (0.0, 0.0)]
            valid_ys = [p[1] for p in norm_kps_list if p != (0.0, 0.0)]
            
            if valid_xs:
                max_val_x = max(valid_xs)
                self.canvas.axes.scatter(valid_xs, valid_ys, c='blue', s=10)
                self.canvas.axes.set_aspect('equal')
                # Strict limits as requested: Y inverted 1.05 to -0.05, X tightened
                self.canvas.axes.set_xlim(-0.05, max_val_x + 0.05)
                self.canvas.axes.set_ylim(1.05, -0.05)
            
            self.canvas.draw_idle()
            
            # 3. Draw keypoints and skeleton onto Main Image Overlay
            drawn_cv_img = cv_img.copy()
            
            # Draw lines first
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx < len(kps) and end_idx < len(kps):
                    p1 = kps[start_idx]
                    p2 = kps[end_idx]
                    if p1[2] > 0 and p2[2] > 0:
                        cv2.line(drawn_cv_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1)

            # Draw dots on top
            for pt in kps:
                x, y, conf = pt
                if conf > 0:
                    cv2.circle(drawn_cv_img, (int(x), int(y)), 1, (0, 255, 0), -1)
            self.cached_main_pixmap = self.cvimg_to_qpixmap(drawn_cv_img)
        else:
            self.cached_main_pixmap = self.cvimg_to_qpixmap(cv_img)
            self.canvas.axes.clear()
            self.canvas.draw_idle()

        # Render Pixmaps cleanly without lag
        self.update_pixmaps()

    def update_pixmaps(self):
        """Fit cached pixmaps to available label geometries preserving aspect ratio."""
        # Main Image Check
        if self.cached_main_pixmap:
            label_w = self.main_image_label.width()
            label_h = self.main_image_label.height()
            if label_w > 0 and label_h > 0:
                scaled_main = self.cached_main_pixmap.scaled(
                    label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.main_image_label.setPixmap(scaled_main)

        # Head Crop Check
        if self.cached_head_pixmap:
            label_w = self.head_crop_label.width()
            label_h = self.head_crop_label.height()
            if label_w > 0 and label_h > 0:
                scaled_head = self.cached_head_pixmap.scaled(
                    label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.head_crop_label.setPixmap(scaled_head)
        else:
            self.head_crop_label.clear()
            self.head_crop_label.setText("No Head Detected")

    def keyPressEvent(self, event):
        """Handle keyboard events for navigation and labeling."""
        key = event.key()
        
        if key == Qt.Key_A:
            if self.current_index > 0:
                self.current_index -= 1
                self.update_ui()
                print(f"Action: Previous Image (A) | Current Index: {self.current_index}")
        elif key == Qt.Key_D:
            if self.current_index < len(self.image_pairs) - 1:
                self.current_index += 1
                self.update_ui()
                print(f"Action: Next Image (D) | Current Index: {self.current_index}")
        elif key == Qt.Key_S:
            print("Action: Open Settings (S)")
            self.open_settings()
        elif key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4]:
            label_val = key - Qt.Key_0
            print(f"Action: Assigned Label {label_val}")
            
            if 0 <= self.current_index < len(self.image_pairs):
                img_path, txt_path = self.image_pairs[self.current_index]
                
                # Fetch keypoints for normalization
                cv_img = cv2.imread(str(img_path))
                img_h, img_w = cv_img.shape[:2] if cv_img is not None else (0, 0)
                kps = self.parse_yolo_txt(txt_path, img_w, img_h)
                
                # Perform normalization & alignment (Min-Max)
                row_data = PoseNormalizer.normalize_keypoints(kps) if kps else {}
                
                # Ensure missing fields are 0.0 if normalizer skipped them
                row_data['label'] = str(label_val)
                row_data['image_path'] = img_path.name
                
                # 1. Determine Source & Handle Re-labeling (Absolute Anchors)
                img_name_key = img_path.name
                current_source_path = img_path
                
                if img_name_key in self.labels_state:
                    try:
                        old_row = self.labels_state[img_name_key]
                        old_label_val = int(old_row['label'])
                        # If label changed, identify the old folder using absolute processed_dir
                        if old_label_val != label_val:
                            old_label_name = self.category_names[old_label_val-1] if old_label_val <= len(self.category_names) else f"label_{old_label_val}"
                            old_categorized_file = self.processed_dir / old_label_name / img_name_key
                            if old_categorized_file.exists():
                                current_source_path = old_categorized_file
                                print(f"Source Switch: Moving from category '{old_label_name}' to new one.")
                    except Exception as e:
                        print(f"Cleanup Error: {e}")

                # 2. Update Annotations State
                self.labels_state[img_name_key] = row_data
                self.save_annotations()
                
                # 3. Perform the Move (Always anchored to self.processed_dir)
                name = self.category_names[label_val-1] if label_val <= len(self.category_names) else f"label_{label_val}"
                self.status_label.setText(f"File: {img_name_key} | Label: {name}")

                try:
                    target_dir = self.processed_dir / name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / img_name_key
                    
                    # Safety: Ensure we don't try to move if file disappeared or is at target
                    if current_source_path.exists() and current_source_path != target_file:
                        shutil.move(str(current_source_path), str(target_file))
                        print(f"Action: Moved Image -> {target_file}")
                    else:
                        print(f"Skip Move: File already at target or missing.")
                except Exception as e:
                    print(f"Error moving image: {e}")

                # 4. Update in-memory path (Maintain queue for backward navigation)
                self.image_pairs[self.current_index] = (Path(target_file), txt_path)
                
                # Automatically move to the next image if available
                if self.current_index < len(self.image_pairs) - 1:
                    self.current_index += 1
                    self.update_ui()
                    print(f"Action: Unified Navigation | Moved to next index: {self.current_index}")
                else:
                    self.update_ui()
                    print("Action: Finished labeling all current images.")
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle window resizing gracefully."""
        super().resizeEvent(event)
        self.update_pixmaps()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationWindow()
    window.show()
    sys.exit(app.exec_())
