import os
from pathlib import Path
from ultralytics import YOLO

def run_inference(input_dir: str, output_dir: str, model_path: str):
    """
    Run YOLO pose inference on a directory of images and save outputs.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validation checks
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
        
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO Model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # Collect image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in '{input_dir}'. Exiting.")
        return
        
    print(f"Found {len(images)} images. Starting inference...")
    
    for img_path in images:
        # Run inference
        results = model(str(img_path), verbose=False)
        
        # Save results in YOLO .txt format
        if len(results) > 0:
            out_txt_path = output_path / f"{img_path.stem}.txt"
            # save_txt correctly outputs bbox + keypoints for pose models
            results[0].save_txt(str(out_txt_path))
            print(f"Processed: {img_path.name} -> {out_txt_path.name}")
        else:
            print(f"No detection for: {img_path.name}")

if __name__ == "__main__":
    # Hardcoded default paths for easy "No-Flag" execution
    DEFAULT_INPUT = "data/raw"
    DEFAULT_OUTPUT = "data/txt"
    DEFAULT_MODEL = "yolo11n-pose.pt"
    
    # Use relative paths from the project root
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = str(base_dir / DEFAULT_INPUT)
    output_dir = str(base_dir / DEFAULT_OUTPUT)
    model_path = str(base_dir / DEFAULT_MODEL)
    
    try:
        run_inference(input_dir, output_dir, model_path)
        print("Inference step completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
