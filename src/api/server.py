import os
import cv2
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "src" / "dataset"
CROPS_DIR = BASE_DIR / "temp_crops"
MODEL_PATH = BASE_DIR / "yolo11n-pose.pt"

# Ensure directories exist
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Load Model
print(f"Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

class CropInfo(BaseModel):
    id: str
    url: str
    source_image: str
    label: str = ""

@app.get("/api/crops", response_model=List[CropInfo])
async def get_crops():
    """
    Scans the dataset, runs inference if needed, and returns person crops.
    For MVP, we scan a subset of 'train/images'.
    """
    image_paths = list((DATASET_DIR / "train" / "images").glob("*.jpg")) + \
                  list((DATASET_DIR / "train" / "images").glob("*.jpeg")) + \
                  list((DATASET_DIR / "train" / "images").glob("*.png"))
    
    # Limit to first 50 images for performance in this demo
    image_paths = image_paths[:50]
    
    all_crops = []
    
    for img_path in image_paths:
        relative_path = img_path.relative_to(DATASET_DIR)
        # Unique prefix for this image's crops
        crop_prefix = img_path.stem
        
        # Check if we already have crops for this image
        existing_crops = list(CROPS_DIR.glob(f"{crop_prefix}_crop_*.jpg"))
        
        if not existing_crops:
            # Run inference
            results = model(str(img_path), verbose=False)
            if len(results) > 0 and results[0].boxes:
                # Find persons (class 0 in COCO/YOLO default for pose model)
                # Actually yolov8-pose detects persons by default.
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls[0])
                    if cls == 0: # Person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Add some padding to the crop
                            h, w = img.shape[:2]
                            pad = 20
                            x1 = max(0, x1 - pad)
                            y1 = max(0, y1 - pad)
                            x2 = min(w, x2 + pad)
                            y2 = min(h, y2 + pad)
                            
                            crop = img[y1:y2, x1:x2]
                            crop_name = f"{crop_prefix}_crop_{i}.jpg"
                            cv2.imwrite(str(CROPS_DIR / crop_name), crop)
                            existing_crops.append(CROPS_DIR / crop_name)
        
        for crop_file in existing_crops:
             all_crops.append(CropInfo(
                id=crop_file.name,
                url=f"/crops/{crop_file.name}",
                source_image=img_path.name
            ))
            
    return all_crops

@app.post("/api/label")
async def label_crop(crop_id: str, label: str):
    # For now, just print and return success
    print(f"Labeled {crop_id} as {label}")
    return {"status": "success"}

# Serve static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "src" / "api" / "static")), name="static")
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(str(BASE_DIR / "src" / "api" / "static" / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
