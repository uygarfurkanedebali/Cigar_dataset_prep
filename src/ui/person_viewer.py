import sys
import os
import cv2
import gc
import time
import argparse
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QLabel, QSizePolicy, QHBoxLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class InteractionTool(QMainWindow):
    def __init__(self, dataset_path, model_path, mode="view"):
        super().__init__()
        self.setWindowTitle(f"Interaction Workspace [{mode.upper()}]")
        self.resize(1500, 900)
        
        self.dataset_path = Path(dataset_path)
        self.source_images = self.dataset_path / "train" / "images"
        
        # Output Paths
        self.output_interaction = self.dataset_path.parent / "interaction_dataset"
        self.output_no_person = self.dataset_path.parent / "no_person_dataset"
        
        self.mode = mode
        self.samples = []
        self.current_idx = 0
        self.model = None
        self.model_path = model_path
        
        self.init_ui()
        QTimer.singleShot(100, self.start_workflow)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        self.logo_label = QLabel("Comprehensive Interaction Pipeline")
        color = "#3b82f6" if self.mode == "view" else "#f59e0b"
        self.logo_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(self.logo_label)
        header_layout.addStretch()
        self.stats_label = QLabel("Initializing...")
        self.stats_label.setStyleSheet("color: #94a3b8; font-size: 14px;")
        header_layout.addWidget(self.stats_label)
        layout.addLayout(header_layout)
        
        # Progress View
        self.progress_container = QWidget()
        prog_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #334155; border-radius: 4px; background: #0f172a; text-align: center; color: white; height: 16px; }
            QProgressBar::chunk { background-color: #f59e0b; }
        """)
        prog_layout.addWidget(self.progress_bar)
        self.status_msg = QLabel("Ready...")
        self.status_msg.setStyleSheet("color: #fbbf24; font-size: 11px;")
        self.status_msg.setAlignment(Qt.AlignCenter)
        prog_layout.addWidget(self.status_msg)
        self.progress_container.setLayout(prog_layout)
        layout.addWidget(self.progress_container)
        
        if self.mode != "prepare":
            self.progress_container.hide()
        
        # Dual-View Layout
        display_layout = QHBoxLayout()
        # Main
        main_v = QVBoxLayout()
        self.crop_title = QLabel("Focus Area")
        self.crop_title.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        main_v.addWidget(self.crop_title)
        self.display_label = QLabel("Main View")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background: #0f172a; border: 2px solid #1e293b; border-radius: 8px;")
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_v.addWidget(self.display_label)
        display_layout.addLayout(main_v, 70)
        
        # Sidebar
        side_v = QVBoxLayout()
        self.orig_title = QLabel("Reference View")
        self.orig_title.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        side_v.addWidget(self.orig_title)
        self.preview_label = QLabel("Context")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #020617; border: 1px solid #334155; border-radius: 4px;")
        self.preview_label.setFixedSize(350, 260)
        side_v.addWidget(self.preview_label)
        self.details_label = QLabel("Details...")
        self.details_label.setStyleSheet("color: #94a3b8; font-size: 11px; margin-top: 10px;")
        side_v.addWidget(self.details_label)
        side_v.addStretch()
        display_layout.addLayout(side_v, 30)
        
        layout.addLayout(display_layout)
        
        # Footer
        footer_layout = QHBoxLayout()
        self.footer_msg = QLabel(f"Mode: {self.mode.capitalize()}")
        self.footer_msg.setStyleSheet("color: #64748b; font-size: 11px;")
        footer_layout.addWidget(self.footer_msg)
        footer_layout.addStretch()
        controls = QLabel("[A] Prev  |  [D] Next  |  Esc Exit")
        controls.setStyleSheet("color: #475569; font-size: 11px;")
        footer_layout.addWidget(controls)
        layout.addLayout(footer_layout)
        
        central_widget.setLayout(layout)
        self.setStyleSheet("background-color: #020617;")

    def start_workflow(self):
        if self.mode == "prepare":
            self.prepare_dataset()
            self.progress_container.hide()
        self.load_and_view()

    def _wipe_and_prep_dir(self, base_path):
        if base_path.exists(): shutil.rmtree(base_path)
        img_p = base_path / "images" / "train"
        lbl_p = base_path / "labels" / "train"
        img_p.mkdir(parents=True, exist_ok=True)
        lbl_p.mkdir(parents=True, exist_ok=True)
        return img_p, lbl_p

    def prepare_dataset(self):
        self.status_msg.setText("Reseting directories...")
        QApplication.processEvents()
        
        ia_img, ia_lbl = self._wipe_and_prep_dir(self.output_interaction)
        np_img, np_lbl = self._wipe_and_prep_dir(self.output_no_person)
        
        image_files = sorted(list(self.source_images.glob("*.jpg")) + list(self.source_images.glob("*.jpeg")) + list(self.source_images.glob("*.png")))
        total = len(image_files)
        if total == 0: return

        self.model = YOLO(self.model_path)
        self.progress_bar.setMaximum(total)
        start_time = time.time()
        
        for i, img_path in enumerate(image_files):
            processed = i+1
            self.progress_bar.setValue(processed)
            elapsed = time.time() - start_time
            avg = elapsed / processed
            m, s = divmod(int((total - processed) * avg), 60)
            self.status_msg.setText(f"Scanning: {img_path.name} | ETA: {m:02d}:{s:02d}")
            QApplication.processEvents()
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            h_orig, w_orig = img.shape[:2]
            
            # Detect
            results = self.model(str(img_path), verbose=False)
            person_boxes = []
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0: person_boxes.append(list(map(int, box.xyxy[0])))
            
            cigar_boxes = self.get_smoking_boxes(img_path, w_orig, h_orig)
            
            # Categorize
            inter_cigars, inter_persons = [], []
            for pb in person_boxes:
                matched = False
                for cb in cigar_boxes:
                    if self._intersect(pb, cb):
                        if cb not in inter_cigars: inter_cigars.append(cb)
                        matched = True
                if matched: inter_persons.append(pb)
            
            if inter_cigars:
                # Interaction Case -> Save Crop
                ux1, uy1, ux2, uy2 = self._get_super_union(inter_persons, inter_cigars, w_orig, h_orig)
                crop_img = img[uy1:uy2, ux1:ux2]
                if crop_img.size > 0:
                    ch, cw = crop_img.shape[:2]
                    base = f"{img_path.stem}_iactive"
                    cv2.imwrite(str(ia_img / f"{base}.jpg"), crop_img)
                    with open(ia_lbl / f"{base}.txt", 'w') as lf:
                        for cb in inter_cigars:
                            nx1, ny1, nx2, ny2 = max(0, cb[0]-ux1), max(0, cb[1]-uy1), min(cw, cb[2]-ux1), min(ch, cb[3]-uy1)
                            cx, cy = ((nx1 + nx2) / 2) / cw, ((ny1 + ny2) / 2) / ch
                            nw, nh = (nx2 - nx1) / cw, (ny2 - ny1) / ch
                            lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            elif cigar_boxes:
                # No Interaction + Cigar Exists -> Save Full Original Frame
                base = f"{img_path.stem}_cigar_only"
                shutil.copy(str(img_path), str(np_img / f"{base}{img_path.suffix}"))
                # Copy original label file (assuming only smoking labels in source or we filter)
                src_lbl = self.dataset_path / "train" / "labels" / f"{img_path.stem}.txt"
                if src_lbl.exists():
                    shutil.copy(str(src_lbl), str(np_lbl / f"{base}.txt"))
            
            del results
            gc.collect()
        del self.model
        gc.collect()

    def _intersect(self, b1, b2):
        return max(b1[0], b2[0]) < min(b1[2], b2[2]) and max(b1[1], b2[1]) < min(b1[3], b2[3])

    def _get_super_union(self, p_boxes, c_boxes, img_w, img_h, padding=0.2):
        all_b = p_boxes + c_boxes
        x1, y1 = min(b[0] for b in all_b), min(b[1] for b in all_b)
        x2, y2 = max(b[2] for b in all_b), max(b[3] for b in all_b)
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - w * padding)), max(0, int(y1 - h * padding))
        x2, y2 = min(img_w, int(x2 + w * padding)), min(img_h, int(y2 + h * padding))
        return [x1, y1, x2, y2]

    def get_smoking_boxes(self, img_path, img_w, img_h):
        label_file = self.dataset_path / "train" / "labels" / f"{img_path.stem}.txt"
        boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or int(parts[0]) != 4: continue
                    coords = list(map(float, parts[1:]))
                    if len(parts) == 5:
                        cx, cy, w, h = coords
                        x1, y1, x2, y2 = int((cx - w/2) * img_w), int((cy - h/2) * img_h), int((cx + w/2) * img_w), int((cy + h/2) * img_h)
                    else:
                        xs, ys = coords[0::2], coords[1::2]
                        x1, y1, x2, y2 = int(min(xs) * img_w), int(min(ys) * img_h), int(max(xs) * img_w), int(max(ys) * img_h)
                    boxes.append([x1, y1, x2, y2])
        return boxes

    # --- VIEWER MODULE --- (Defaults to interaction_dataset)
    def load_and_view(self):
        self.samples = []
        view_path = self.output_interaction / "images" / "train"
        if not view_path.exists():
            self.display_label.setText("No Interaction Dataset found.\nRun with --prepare.")
            return
            
        label_dir = self.output_interaction / "labels" / "train"
        image_files = sorted(list(view_path.glob("*.jpg")))
        for img_path in image_files:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                orig_stem = img_path.stem.replace("_iactive", "")
                orig_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    potential = self.source_images / f"{orig_stem}{ext}"
                    if potential.exists():
                        orig_path = potential
                        break
                self.samples.append({'img': img_path, 'lbl': label_path, 'orig': orig_path})
        
        if self.samples: self.display_current()
        else: self.display_label.setText("No interaction samples found.")

    def display_current(self):
        if not self.samples: return
        sample = self.samples[self.current_idx]
        self.stats_label.setText(f"Viewing Interaction: {self.current_idx + 1} / {len(self.samples)}")
        self.details_label.setText(f"File: {sample['img'].name}\nSource: {sample['orig'].name if sample['orig'] else 'NA'}")
        
        img = cv2.imread(str(sample['img']))
        if img is not None:
            h, w = img.shape[:2]
            with open(sample['lbl'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, lw, lh = map(float, parts)
                        x1, y1, x2, y2 = int((cx - lw/2) * w), int((cy - lh/2) * h), int((cx + lw/2) * w), int((cy + lh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            self._set_pixmap(img, self.display_label)
        
        if sample['orig']:
            oimg = cv2.imread(str(sample['orig']))
            if oimg is not None: self._set_pixmap(oimg, self.preview_label)
        else: self.preview_label.setText("No Context Found")

    def _set_pixmap(self, cv_img, label):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qImg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qImg)
        lw, lh = label.width() - 8, label.height() - 8
        label.setPixmap(pix.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.close()
        if not self.samples: return
        if event.key() == Qt.Key_D and self.current_idx < len(self.samples) - 1:
            self.current_idx += 1
            self.display_current()
        elif event.key() == Qt.Key_A and self.current_idx > 0:
            self.current_idx -= 1
            self.display_current()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    
    mode = "prepare" if args.prepare else "view"
    app = QApplication(sys.argv)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_PATH = BASE_DIR / "src" / "dataset"
    MODEL_PATH = BASE_DIR / "yolo11n-pose.pt"
    tool = InteractionTool(DATASET_PATH, MODEL_PATH, mode=mode)
    tool.show()
    sys.exit(app.exec_())
