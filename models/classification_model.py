import os
import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Prevent PIL from throwing errors on large images
Image.MAX_IMAGE_PIXELS = None

# Global Configurations
IMAGE_SIZE = 50
DATASET_MEAN = [0.5223926305770874, 0.5005241632461548, 0.49497371912002563]
DATASET_STD = [0.14497309923171997, 0.14863280951976776, 0.14289118349552155]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Model Architecture
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.gelu(out)
        return out

class SamsungDefectDetector(nn.Module):
    def __init__(self, num_classes=5):
        super(SamsungDefectDetector, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.layer1 = ResidualBlock(64, 128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = ResidualBlock(256, 512, stride=2)
        self.layer4 = ResidualBlock(512, 1024, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ==========================================
# Classification Logic
# ==========================================
def evaluate_patches_from_json(image_path, json_path, output_json_path, model, device, classes):
    """
    Reads suspected patches from JSON, classifies them, appends predictions (and confidence),
    and saves the updated JSON without generating images.
    """
    original_img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
    ])

    with open(json_path, 'r') as f:
        suspected_patches = json.load(f)

    model.eval()
    print(f"🔍 Classifying {len(suspected_patches)} patches from JSON...")

    with torch.no_grad():
        for patch_data in tqdm(suspected_patches, desc="Processing patches"):
            box = patch_data["box"]

            x1, y1, x2, y2 = [int(c) for c in box]

            # Crop window (adjusting slightly to prevent border issues)
            window = original_img.crop((x1, y1, x2 - 1, y2 - 1))

            input_tensor = transform(window).unsqueeze(0).to(device)
            output = model(input_tensor)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

            class_idx = pred.item()
            confidence = conf.item()

            # Append classification results to the existing JSON dictionary
            patch_data["predicted_class_name"] = classes[class_idx]
            patch_data["predicted_class_idx"] = class_idx
            patch_data["classification_confidence"] = round(float(confidence), 4)

    # Save the updated JSON list
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(suspected_patches, f, indent=4)

    print(f"✅ Successfully saved classified JSON to: {output_json_path}")
    return suspected_patches

# ==========================================
# Execution Module
# ==========================================
if __name__ == "__main__":
    import argparse

    # 1. Set up Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Classify suspected defect patches from an image")

    # Required Arguments
    parser.add_argument("image", type=str, help="Path to the original image file")
    parser.add_argument("input_json", type=str, help="Path to the JSON file with suspected patches (from binary model)")

    # Optional Arguments (Flags)
    parser.add_argument("--model", type=str, default="best_samsung_model.pth",
                        help="Path to classification model weights")
    parser.add_argument("--output", type=str, default="updated_patches_with_classes.json", help="Output JSON filename")
    parser.add_argument("--top10", type=str, default="top10_by_class.json", help="Filename for top 10 results export")

    args = parser.parse_args()

    print(f"Working on: {DEVICE}")

    # Define target classes
    SAMSUNG_CLASSES = ['Broken Line', 'Edge False Color', 'Over Desaturation', 'Saturated False Color', 'Smears']

    # Path Validation
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
    elif not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
    elif not os.path.exists(args.input_json):
        print(f"Error: Input JSON '{args.input_json}' not found.")
    else:
        # 2. Load Model
        model = SamsungDefectDetector(num_classes=len(SAMSUNG_CLASSES)).to(DEVICE)
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
        print(f"Classification Model {args.model} loaded successfully!")

        # 3. Evaluate patches and generate new JSON
        evaluate_patches_from_json(
            image_path=args.image,
            json_path=args.input_json,
            output_json_path=args.output,
            model=model,
            device=DEVICE,
            classes=SAMSUNG_CLASSES
        )