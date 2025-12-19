"""
Parking Detection using Gemini 3 Flash
Uses Google's Gemini vision model for parking space detection in aerial imagery.
Implements a two-step detection process:
1. Detect ALL parking spaces (structure/geometry)
2. Detect EMPTY parking spaces
3. Merge results to determine occupancy with high accuracy
"""
import os
import base64
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParkingSpot:
    """Represents a detected parking space"""
    id: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: List[Tuple[float, float]]  # Segmentation polygon points
    confidence: float
    is_occupied: bool = False
    vehicle_confidence: Optional[float] = None
    area_pixels: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'polygon': self.polygon,
            'confidence': self.confidence,
            'is_occupied': self.is_occupied,
            'vehicle_confidence': self.vehicle_confidence,
            'area_pixels': self.area_pixels
        }


@dataclass
class Vehicle:
    """Represents a detected vehicle"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str  # car, truck, bus, motorcycle
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_name': self.class_name
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image"""
    parking_spots: List[ParkingSpot] = field(default_factory=list)
    vehicles: List[Vehicle] = field(default_factory=list)
    total_spots: int = 0
    empty_spots: int = 0
    occupied_spots: int = 0
    occupancy_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'parking_spots': [p.to_dict() for p in self.parking_spots],
            'vehicles': [v.to_dict() for v in self.vehicles],
            'total_spots': self.total_spots,
            'empty_spots': self.empty_spots,
            'occupied_spots': self.occupied_spots,
            'occupancy_rate': self.occupancy_rate
        }


class GeminiParkingDetector:
    """
    Parking space detector using Google Gemini 3 Flash
    Uses a two-step process: find all spots -> find empty spots -> merge
    """
    
    PROMPT_ALL_SPOTS = """
Analyze this aerial/satellite image of a parking area in the Netherlands.

TASK: Identify ALL visible parking spaces with maximum accuracy, including both occupied and empty spaces.

DUTCH PARKING CHARACTERISTICS TO CONSIDER:
- Standard parking space dimensions: typically 5m × 2.5m (but may vary)
- Common marking styles: white lines, yellow lines, or painted boxes
- Parking orientations: perpendicular (90°), angled (45-60°), or parallel
- Surface types: asphalt, concrete, or paving stones with visible demarcation
- May include designated spots for: disabled parking (blue markings), electric vehicles, motorcycles/scooters
- Dutch parking lots often have bicycle parking areas - exclude these

DETECTION REQUIREMENTS:
1. Identify the structural layout: lines, painted markings, curbs, or surface patterns indicating parking boundaries
2. Verify each space has clear visual boundaries (at least 2-3 sides visible)
3. Distinguish between parking spaces and:
   - Driving lanes/roadways
   - Loading zones
   - Pedestrian walkways
   - Bicycle parking areas
   - Green spaces or non-parking surfaces

4. For partially occluded spaces (e.g., under trees, shadowed areas):
   - Infer location based on adjacent visible spaces and consistent spacing patterns
   - Lower confidence score accordingly

ACCURACY GUIDELINES:
- confidence ≥ 0.90: Space clearly visible with distinct boundaries
- confidence 0.70-0.89: Space visible but partially obscured or unclear markings
- confidence 0.50-0.69: Space inferred from pattern/layout, limited direct visibility
- confidence < 0.50: Do not include (too uncertain)

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown formatting, no code blocks, no explanatory text):

{
  "parking_spots": [
    {"id": "P001", "bbox": [x1, y1, x2, y2], "confidence": 0.70},
    {"id": "P002", "bbox": [x1, y1, x2, y2], "confidence": 0.70}
  ],
  "metadata": {
    "total_spots": <count>,
    "image_dimensions": [width, height],
    "notes": "Brief description of parking layout or detection challenges"
  }
}

COORDINATES:
- bbox format: [x1, y1, x2, y2] in pixels
- (x1, y1) = top-left corner
- (x2, y2) = bottom-right corner
- Origin (0, 0) is at top-left of image

CRITICAL: Include ALL parking spaces you can identify with reasonable confidence (≥0.50). Err on the side of completeness while maintaining accuracy.
"""

    PROMPT_EMPTY_SPOTS = """Analyze this aerial/satellite image of a parking area to identify EMPTY parking spaces.

OBJECTIVE: Detect ALL empty (unoccupied) parking spaces with maximum accuracy.

EMPTY PARKING SPACE DEFINITION:
A space is EMPTY if:
- Parking lines/boundaries are clearly visible
- NO vehicle is present within the marked area
- The pavement/surface is visible and unobstructed
- Space appears ready for a vehicle to park

DO NOT MARK AS EMPTY:
- Spaces occupied by cars, trucks, vans, motorcycles, or any vehicle
- Spaces with objects blocking them (shopping carts, construction equipment, barriers)
- Spaces with debris, storage items, or temporary obstacles
- Partially occupied spaces (vehicle overhanging from adjacent spot)
- Shadow-only areas that might contain dark-colored vehicles
- Areas that are NOT designated parking spaces (driving lanes, walkways)

DETECTION STRATEGY:
1. First, identify the overall parking grid/layout and all marked spaces
2. Systematically scan each row/section
3. For each space, verify it meets the EMPTY criteria above
4. Pay special attention to:
   - Spaces in shadowed areas (look for vehicle shapes/shadows)
   - Spaces between occupied spots (don't assume they're occupied)
   - Corner and end-row spaces (often overlooked)
   - Compact or motorcycle-specific spaces
   - Spaces with faded markings but clear boundaries

CONFIDENCE SCORING:
- 0.95-1.0: Space clearly empty, excellent visibility, distinct markings
- 0.85-0.94: Space appears empty, good visibility, minor shadows/fading
- 0.70-0.84: Space likely empty, partial obstruction (shadows, image quality)
- 0.50-0.69: Space possibly empty, significant uncertainty (use cautiously)
- Below 0.50: DO NOT include

BBOX ACCURACY:
- Draw tight bounding boxes around the actual parking space boundaries
- Include the full extent of the marked parking area
- Align with visible parking lines when possible
- For spaces without clear lines, estimate based on adjacent spaces and typical dimensions

SYSTEMATIC APPROACH:
- Scan the entire parking area methodically (left-to-right, top-to-bottom)
- Don't skip sections that seem mostly full
- Verify each potential empty space twice before including
- Count total parking spaces and occupied spaces as a cross-check

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no code blocks, no extra text):

{
  "empty_spots": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.70},
    {"bbox": [x1, y1, x2, y2], "confidence": 0.70}
  ]
}

COORDINATES:
- bbox: [x1, y1, x2, y2] in pixels
- (x1, y1) = top-left corner of parking space
- (x2, y2) = bottom-right corner of parking space
- Coordinates relative to image with origin (0,0) at top-left

CRITICAL REMINDERS:
✓ Include ALL empty spaces, even in corners or partially visible areas
✓ Be thorough - missing empty spots is worse than occasional false positives
✓ When uncertain if a space contains a dark vehicle or is just shadowed, examine carefully for vehicle features (wheels, windows, roof lines)
✓ Ensure bounding boxes don't overlap with occupied spaces
✓ Minimum confidence threshold: 0.50
"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize Gemini parking detector
        
        Args:
            api_key: Google API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None
        
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. Set it via environment variable or pass to constructor.")
        else:
            self._init_client()
    
    def _init_client(self):
        """Initialize the Gemini client using latest google-genai SDK"""
        try:
            from google import genai
            
            # Create client with API key
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized with model '{self.model_name}'")
            
        except ImportError:
            logger.error("google-genai not installed. Run: pip install -U google-genai")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    def _compute_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def _parse_spots_response(self, response_text: str, img_width: int, img_height: int, key_name: str = "parking_spots") -> List[Dict]:
        """Generic parser for Gemini response"""
        spots = []
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("No JSON found in Gemini response")
                    return []
            
            data = json.loads(json_str)
            
            for item in data.get(key_name, []):
                try:
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    # Validate and clamp coordinates
                    x1 = max(0, min(int(bbox[0]), img_width))
                    y1 = max(0, min(int(bbox[1]), img_height))
                    x2 = max(0, min(int(bbox[2]), img_width))
                    y2 = max(0, min(int(bbox[3]), img_height))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    item['bbox'] = (x1, y1, x2, y2)
                    spots.append(item)
                except Exception as e:
                    logger.debug(f"Failed to parse item: {e}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Gemini response: {e}")
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            
        return spots

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect parking spaces and determine occupancy using two-step Gemini process.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            
        Returns:
            DetectionResult with parking spots
        """
        if self.client is None:
            if self.api_key:
                self._init_client()
            if self.client is None:
                logger.warning("Gemini client not initialized, returning empty results")
                return DetectionResult()
        
        try:
            img_height, img_width = image.shape[:2]
            pil_image = Image.fromarray(image)
            
            # Import types for config
            from google.genai import types

            # Step 1: Detect ALL parking spots
            logger.info("Step 1: Detecting all parking spots...")
            response_all = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=self.PROMPT_ALL_SPOTS),
                            types.Part.from_bytes(
                                data=self._image_to_base64_bytes(image),
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                ]
            )
            all_spots_data = self._parse_spots_response(response_all.text, img_width, img_height, "parking_spots")
            logger.info(f"Found {len(all_spots_data)} potential parking spots.")

            # Step 2: Detect EMPTY parking spots
            logger.info("Step 2: Detecting empty parking spots...")
            response_empty = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=self.PROMPT_EMPTY_SPOTS),
                            types.Part.from_bytes(
                                data=self._image_to_base64_bytes(image),
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                ]
            )
            empty_spots_data = self._parse_spots_response(response_empty.text, img_width, img_height, "empty_spots")
            logger.info(f"Found {len(empty_spots_data)} empty spots.")

            # Step 3: Merge results
            final_parking_spots = []
            
            for idx, spot_data in enumerate(all_spots_data):
                bbox = spot_data['bbox']
                # Create polygon from bbox corners
                x1, y1, x2, y2 = bbox
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                area = (x2 - x1) * (y2 - y1)
                
                # Check overlap with empty spots
                is_empty = False
                for empty_spot in empty_spots_data:
                    iou = self._compute_iou(bbox, empty_spot['bbox'])
                    if iou > 0.4:  # Threshold for matching
                        is_empty = True
                        break
                
                # If matched with an empty spot, it is NOT occupied.
                # If NOT matched with an empty spot, it IS occupied.
                is_occupied = not is_empty
                
                final_parking_spots.append(ParkingSpot(
                    id=spot_data.get("id", f"PS_{idx:04d}"),
                    bbox=bbox,
                    polygon=polygon,
                    confidence=float(spot_data.get("confidence", 0.8)),
                    is_occupied=is_occupied,
                    area_pixels=float(area)
                ))

            # Calculate statistics
            total_spots = len(final_parking_spots)
            occupied_spots = sum(1 for s in final_parking_spots if s.is_occupied)
            empty_spots = total_spots - occupied_spots
            occupancy_rate = occupied_spots / total_spots if total_spots > 0 else 0.0
            
            result = DetectionResult(
                parking_spots=final_parking_spots,
                vehicles=[], # We are not explicitly detecting vehicles separately anymore
                total_spots=total_spots,
                empty_spots=empty_spots,
                occupied_spots=occupied_spots,
                occupancy_rate=occupancy_rate
            )
            
            logger.info(
                f"Gemini two-step detection complete: {total_spots} spots "
                f"({empty_spots} empty, {occupied_spots} occupied)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini detection failed: {e}")
            import traceback
            traceback.print_exc()
            return DetectionResult()

    def _image_to_base64_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy image to base64 bytes for inline_data"""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


class ParkingDetectionPipeline:
    """
    Complete parking detection pipeline using Gemini 3 Flash
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        # Legacy parameters (ignored, kept for compatibility)
        parking_model_path: Optional[str] = None,
        parking_confidence: float = 0.5,
        vehicle_model_size: str = 'n',
        vehicle_confidence: float = 0.3,
        overlap_threshold: float = 0.3
    ):
        """
        Initialize the Gemini-based detection pipeline
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use (default: gemini-3-flash-preview)
        """
        logger.info("Initializing Gemini Parking Detection Pipeline...")
        
        self.detector = GeminiParkingDetector(
            api_key=api_key,
            model_name=model_name
        )
        
        logger.info(f"Pipeline initialized successfully with {model_name}")
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run detection on image using Gemini
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            
        Returns:
            DetectionResult with parking spots, vehicles, and occupancy stats
        """
        return self.detector.detect(image)
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[callable] = None
    ) -> List[DetectionResult]:
        """
        Run detection on multiple images
        
        Args:
            images: List of input images
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        total = len(images)
        
        for idx, image in enumerate(images):
            if progress_callback:
                progress_callback(idx + 1, total, f"Processing image {idx + 1}/{total}")
            
            result = self.detect(image)
            results.append(result)
        
        return results


# Convenience functions
def create_pipeline(
    api_key: Optional[str] = None,
    config: Optional[Dict] = None
) -> ParkingDetectionPipeline:
    """
    Create a Gemini-based parking detection pipeline
    
    Args:
        api_key: Google API key
        config: Optional configuration dict (ignored for Gemini pipeline)
    """
    return ParkingDetectionPipeline(api_key=api_key)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gemini parking detection")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--output", help="Path to save annotated image")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load image
    image = np.array(Image.open(args.image))
    
    # Create pipeline
    pipeline = create_pipeline(api_key=args.api_key)
    
    # Run detection
    result = pipeline.detect(image)
    
    # Print results
    print("\n" + "="*50)
    print("GEMINI DETECTION RESULTS")
    print("="*50)
    print(f"Total parking spots: {result.total_spots}")
    print(f"Empty spots: {result.empty_spots}")
    print(f"Occupied spots: {result.occupied_spots}")
    print(f"Occupancy rate: {result.occupancy_rate:.1%}")
    
    # Save annotated image if output specified
    if args.output:
        from src.visualizer import ParkingVisualizer, ParkingSpotVisualization
        
        viz_spots = [
            ParkingSpotVisualization(
                corners=spot.polygon,
                confidence=spot.confidence,
                is_occupied=spot.is_occupied,
                spot_id=spot.id
            )
            for spot in result.parking_spots
        ]
        
        visualizer = ParkingVisualizer()
        visualizer.visualize_detections(image, viz_spots, args.output)
        print(f"\nSaved visualization to {args.output}")
