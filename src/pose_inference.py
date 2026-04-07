import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def run_inference(input_dir: str, output_dir: str, model_path: str):
    """
    Run YOLO pose inference on a directory of images and save outputs.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    # model_file = Path(model_path)
    
    # Validation checks
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")
        
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
    parser = argparse.ArgumentParser(description="Run YOLO Pose Inference to extract keypoints.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .txt label files.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO pose model weights (.pt).")
    
    args = parser.parse_args()
    
    try:
        run_inference(args.input, args.output, args.model)
        print("Inference step completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
