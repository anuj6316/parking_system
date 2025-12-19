"""
Test script for Parking Detection Pipeline
Loads local test images and runs the Gemini 3 detection pipeline,
saving visualized outputs with Red/Green masking for occupancy.
"""
import os
import logging
from typing import List
import numpy as np
from pathlib import Path
from PIL import Image

# Adjust path to allow imports from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parking_detector import ParkingDetectionPipeline, DetectionResult
from src.visualizer import ParkingVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPipeline")

# Gemini API Key
GEMINI_API_KEY = "AIzaSyC5ZGOoIeAMgOg0KosdOq10bX4jNIGn_8k"

# Path to local test images
TEST_IMAGES_DIR = Path("/home/anu/update2/parking-detection-system/new/images/test")


def load_local_images() -> List[np.ndarray]:
    """
    Load test images from local directory.
    Returns list of images as numpy arrays in RGB format.
    """
    logger.info(f"Loading test images from {TEST_IMAGES_DIR}...")
    
    images = []
    # Get all jpg/png images (excluding annotated versions)
    image_files = sorted([
        f for f in TEST_IMAGES_DIR.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        and 'annotated' not in f.name.lower()
    ])
    
    for image_path in image_files:
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(np.array(img))
            logger.info(f"Loaded: {image_path.name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
    
    return images


def print_result(idx: int, image_name: str, result: DetectionResult):
    """Print the detection results in a readable format"""
    print(f"\n--- Image {idx+1}: {image_name} ---")
    print(f"Total Spots: {result.total_spots}")
    print(f"  - Empty:    {result.empty_spots}")
    print(f"  - Occupied: {result.occupied_spots}")
    print(f"Occupancy Rate: {result.occupancy_rate:.1%}")
    if result.parking_spots:
        print(f"First spot confidence: {result.parking_spots[0].confidence:.2f}")


def test_pipeline_with_local_images():
    """
    Main test function:
    1. Loads local images.
    2. Runs Gemini detection.
    3. Prints results.
    4. Generates and saves visualized images.
    """
    # 1. Load Images
    images = load_local_images()
    
    if not images:
        logger.error("No images found. Aborting test.")
        return

    logger.info(f"Successfully loaded {len(images)} images.")

    # Get image filenames for output naming
    image_files = sorted([
        f for f in TEST_IMAGES_DIR.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        and 'annotated' not in f.name.lower()
    ])

    # 2. Initialize Pipeline and Visualizer
    pipeline = ParkingDetectionPipeline(
        api_key=GEMINI_API_KEY,
        model_name="gemini-3-flash-preview"
    )
    visualizer = ParkingVisualizer()
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # 3. Run Batch Detection
    logger.info("Starting batch detection...")
    
    # Callback to show progress
    def progress(curr, total, msg):
        logger.info(f"[{curr}/{total}] {msg}")

    results = pipeline.detect_batch(images, progress_callback=progress)

    # 4. Show Results and Save Visualizations
    print("\n" + "="*50)
    print("TEST EXECUTION COMPLETE")
    print("="*50)
    
    for i, res in enumerate(results):
        img_name = image_files[i].name if i < len(image_files) else f"image_{i+1}"
        print_result(i, img_name, res)
        
        # Save visualization
        output_path = output_dir / f"result_{image_files[i].stem}.png"
        visualizer.visualize_tile_with_detections(
            tile_image=images[i],
            detections=res.parking_spots,
            output_path=str(output_path)
        )
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    test_pipeline_with_local_images()
