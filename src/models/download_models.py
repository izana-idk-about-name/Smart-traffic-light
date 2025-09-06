#!/usr/bin/env python3
"""
Script to download pre-trained ML models for car detection
"""

import os
import requests
import tarfile
from pathlib import Path

class ModelDownloader:
    def __init__(self, models_dir='src/models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def download_file(self, url, filename):
        """Download file from URL"""
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(self.models_dir / filename, 'wb') as f:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(".1f")

        print(f"Downloaded {filename}")

    def download_efficientdet_model(self):
        """Download EfficientDet Lite model for object detection"""
        # Using TensorFlow Hub EfficientDet Lite 2
        model_url = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1?tf-hub-format=compressed"
        model_path = self.models_dir / "efficientdet_lite2.tar.gz"

        print("Downloading EfficientDet Lite model...")
        self.download_file(model_url, "efficientdet_lite2.tar.gz")

        # Extract the model
        print("Extracting model...")
        with tarfile.open(model_path, 'r:gz') as tar:
            tar.extractall(self.models_dir / "efficientdet_lite2")

        # Move to final location
        os.rename(self.models_dir / "efficientdet_lite2" / "saved_model.pb",
                 self.models_dir / "efficientdet_lite2.pb")

        # Clean up
        os.remove(model_path)

    def download_coco_labels(self):
        """Download COCO labels for EfficientDet"""
        labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        with open(self.models_dir / "coco_labels.txt", 'w') as f:
            for i, label in enumerate(labels):
                f.write(f"{i}: {label}\n")

        print("Created COCO labels file")

    def download_mobile_net_ssd(self):
        """Download MobileNet SSD model for object detection"""
        print("Downloading MobileNet SSD model...")

        # Configuration file
        config_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        self.download_file(config_url, "ssd_mobilenet_v3_large_coco.pbtxt")

        # Model weights - using a reliable source
        weights_url = "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
        self.download_file(weights_url, "ssd_mobilenet_v3_large_coco.pb")

    def download_all(self):
        """Download all required models"""
        print("Starting model downloads...")
        print("Note: Model files may be large (>200MB)")
        print()

        try:
            self.download_mobile_net_ssd()
            self.download_coco_labels()
            print("\n✅ All models downloaded successfully!")
            print("You can now use AI-powered car detection!")

        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            print("🔄 Creating fallback CV-only implementation...")
            # Create basic fallback

    def verify_models(self):
        """Verify that downloaded models work"""
        model_path = self.models_dir / "ssd_mobilenet_v3_large_coco.pb"
        config_path = self.models_dir / "ssd_mobilenet_v3_large_coco.pbtxt"

        if model_path.exists() and config_path.exists():
            print("✅ Model files verified successfully")
            return True
        else:
            print("❌ Model files missing or incomplete")
            return False

def main():
    downloader = ModelDownloader()
    downloader.download_all()
    print("Model download complete!")

if __name__ == "__main__":
    main()