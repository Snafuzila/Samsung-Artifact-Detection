import os
import json
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Prevent PIL from throwing errors on large images
Image.MAX_IMAGE_PIXELS = None

# Global Configurations
WINDOW_SIZE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Model Architecture
# ==========================================
class ArtifactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 24x24
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 12x12
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)  # -> 6x6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# Sliding Window Dataset
# ==========================================
class SlidingWindowDataset(Dataset):
    def __init__(self, pil_image, window_size=50, stride=10, transform=None):
        self.image = pil_image
        self.window_size = window_size
        self.transform = transform
        self.coords = []
        w, h = pil_image.size

        # Calculate coordinates for every patch
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                self.coords.append((x, y, x + window_size, y + window_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        box = self.coords[idx]
        window = self.image.crop(box)
        if self.transform:
            window = self.transform(window)
        return window, torch.tensor(box)


# ==========================================
# Inference Engine
# ==========================================
def get_all_predictions(image_path, model, window_size=50, stride=10, batch_size=512):
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SlidingWindowDataset(img, window_size=window_size, stride=stride, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    model.eval()
    results = {"boxes": [], "scores": []}

    print(f"🔍 Analyzing {len(dataset)} patches with Binary Model...")
    with torch.no_grad():
        # Handle AMP autocast depending on device
        if DEVICE.type == 'cuda':
            with torch.amp.autocast('cuda'):
                for windows, boxes in tqdm(loader, desc="Running Inference"):
                    outputs = model(windows.to(DEVICE))
                    probs = torch.softmax(outputs, dim=1)
                    distorted_scores = probs[:, 1]
                    results["boxes"].append(boxes)
                    results["scores"].append(distorted_scores.cpu())
        else:
            for windows, boxes in tqdm(loader, desc="Running Inference"):
                outputs = model(windows.to(DEVICE))
                probs = torch.softmax(outputs, dim=1)
                distorted_scores = probs[:, 1]
                results["boxes"].append(boxes)
                results["scores"].append(distorted_scores.cpu())

    results["boxes"] = torch.cat(results["boxes"]).float()
    results["scores"] = torch.cat(results["scores"])
    return img, results


# ==========================================
# Filtering and Export
# ==========================================
def export_suspects_to_json(raw_results, threshold=0.90, nms_iou=0.05, margin=20, output_file="suspected_patches.json"):
    mask = raw_results["scores"] > threshold

    if not mask.any():
        print("No distortions detected above threshold.")
        with open(output_file, 'w') as f:
            json.dump([], f, indent=4)
        return []

    f_boxes = raw_results["boxes"][mask]
    f_scores = raw_results["scores"][mask]

    # Inflate boxes to group close patches during NMS
    inflated_boxes = f_boxes.clone()
    inflated_boxes[:, 0] -= margin
    inflated_boxes[:, 1] -= margin
    inflated_boxes[:, 2] += margin
    inflated_boxes[:, 3] += margin

    keep = torchvision.ops.nms(inflated_boxes, f_scores, nms_iou)

    final_boxes = f_boxes[keep].cpu().numpy().tolist()
    final_scores = f_scores[keep].cpu().numpy().tolist()

    suspects_list = []
    for box, score in zip(final_boxes, final_scores):
        suspects_list.append({
            "box": box,
            "binary_score": round(float(score), 4)  # Ensure it's a standard python float for JSON
        })

    with open(output_file, 'w') as f:
        json.dump(suspects_list, f, indent=4)

    print(f"✅ Saved {len(suspects_list)} suspicious locations to file {output_file}")
    return suspects_list


# ==========================================
# Execution Module
# ==========================================
if __name__ == "__main__":
    import time
    import argparse

    # 1. Set up Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Run Binary Defect Detection on an Image")

    # Required Argument
    parser.add_argument("image", type=str, help="Path to the image file to analyze")

    # Optional Arguments (Flags)
    parser.add_argument("--model", type=str, default="best_binary_model.pth", help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--iou", type=float, default=0.05, help="NMS IoU threshold")
    parser.add_argument("--output", type=str, default="suspected_patches.json", help="Output JSON filename")

    args = parser.parse_args()

    print(f"Working on: {DEVICE}")

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
    elif not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
    else:
        # 1. Load Model
        model = ArtifactCNN().to(DEVICE)
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
        print(f"Model {args.model} loaded successfully!")

        # 2. Run Inference
        print(f"Starting Inference on {args.image}...")
        start_time = time.time()

        # Using the image path provided via CMD
        original_img, raw_results = get_all_predictions(args.image, model, window_size=50, stride=10, batch_size=512)

        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds.")

        # 3. Filter and Export using CMD arguments
        suspects = export_suspects_to_json(
            raw_results,
            threshold=args.threshold,
            nms_iou=args.iou,
            margin=20,
            output_file=args.output
        )