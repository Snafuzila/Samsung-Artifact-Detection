import os
import time
import argparse
from datetime import datetime
import torch

# Import the tools from your first two files
# (Ensure your files are named binary_model.py and classification_model.py)
from binary_model import ArtifactCNN, get_all_predictions, export_suspects_to_json
from classification_model import SamsungDefectDetector, evaluate_patches_from_json


def run_pipeline(image_path, threshold, iou, output_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Starting pipeline on: {device}")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f" Error: Image '{image_path}' not found.")
        return


    # Paths and Classes
    BINARY_WEIGHTS = os.path.join("weights", "binary_model.pth")
    CLASS_WEIGHTS = os.path.join("weights", "best_samsung_model.pth")
    TEMP_JSON = "temp_suspects.json"
    SAMSUNG_CLASSES = ['Broken Line', 'Edge False Color', 'Over Desaturation', 'Saturated False Color', 'Smears']

    # ==========================================
    # Step 1: Binary Detection
    # ==========================================
    print("\n--- Phase 1: Binary Defect Detection ---")
    binary_model = ArtifactCNN().to(device)

    if not os.path.exists(BINARY_WEIGHTS):
        print(f" Error: Binary model weights '{BINARY_WEIGHTS}' not found.")
        return

    binary_model.load_state_dict(torch.load(BINARY_WEIGHTS, map_location=device))

    # Run predictions
    start_time = time.time()
    _, raw_results = get_all_predictions(image_path, binary_model, window_size=50, stride=10, batch_size=512)

    # Filter and export to a temporary JSON
    suspects = export_suspects_to_json(
        raw_results,
        threshold=threshold,
        nms_iou=iou,
        margin=20,
        output_file=TEMP_JSON
    )

    if not suspects:
        print(" Pipeline finished: No defects found.")
        # If user wanted a file anyway, write an empty list
        with open(output_filename, 'w') as f:
            f.write("[]")
        return

    # ==========================================
    # Step 2: Classification
    # ==========================================
    print("\n--- Phase 2: Defect Classification ---")
    class_model = SamsungDefectDetector(num_classes=len(SAMSUNG_CLASSES)).to(device)

    if not os.path.exists(CLASS_WEIGHTS):
        print(f" Error: Classification weights '{CLASS_WEIGHTS}' not found.")
        return

    class_model.load_state_dict(torch.load(CLASS_WEIGHTS, map_location=device))

    # Read from the temp JSON, classify, and write to the final user-defined output
    evaluate_patches_from_json(
        image_path=image_path,
        json_path=TEMP_JSON,
        output_json_path=output_filename,
        model=class_model,
        device=device,
        classes=SAMSUNG_CLASSES
    )

    # Clean up the temporary intermediate file
    if os.path.exists(TEMP_JSON):
        os.remove(TEMP_JSON)

    total_time = time.time() - start_time
    print(f"\n Pipeline completed in {total_time:.2f} seconds!")
    print(f" Final results saved to: {output_filename}")


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Samsung Image Defect Detection Pipeline")

    # Required positional argument (Can be a specific file OR a directory)
    parser.add_argument("image", type=str, help="Path to the corrupted image file or directory")

    # Optional flags
    parser.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold (default: 0.90)")
    parser.add_argument("--iou", type=float, default=0.05, help="NMS IoU overlap threshold (default: 0.05)")
    parser.add_argument("--output", type=str, default="", help="Output JSON filename (optional)")

    args = parser.parse_args()
    target_path = args.image


    if os.path.isdir(target_path):
        print(f" Detected directory. Processing all images in: {target_path}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(valid_extensions)]
        
        if not images:
            print(" No valid images found in the directory.")
        else:
            for img_path in images:
                print(f"\n{'='*40}\n Processing: {img_path}\n{'='*40}")
                
                # Extract image name without extension (e.g., 'download.jpg' -> 'download')
                image_base = os.path.splitext(os.path.basename(img_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save into the output directory with the image name
                final_output = os.path.join("output", f"{image_base}_{timestamp}.json")
                
                run_pipeline(
                    image_path=img_path,
                    threshold=args.threshold,
                    iou=args.iou,
                    output_filename=final_output
                )


    else:
        final_output = args.output
        if not final_output:
            # Extract image name without extension for the single file
            image_base = os.path.splitext(os.path.basename(target_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output = os.path.join("output", f"{image_base}_{timestamp}.json")
        
        # Execute the pipeline for the single image
        run_pipeline(
            image_path=target_path,
            threshold=args.threshold,
            iou=args.iou,
            output_filename=final_output
        )