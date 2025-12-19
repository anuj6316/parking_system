"""
Parking Detection Visualizer
Overlays detection results on aerial imagery for debugging
Shows parking spots as green (empty) or red (occupied) with confidence scores
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    # Colors (BGR format for OpenCV, RGBA for PIL)
    empty_color: Tuple[int, int, int, int] = (0, 255, 0, 128)  # Green with transparency
    occupied_color: Tuple[int, int, int, int] = (255, 0, 0, 128)  # Red with transparency
    border_color_empty: Tuple[int, int, int] = (0, 200, 0)  # Dark green border
    border_color_occupied: Tuple[int, int, int] = (200, 0, 0)  # Dark red border
    
    # Font settings
    font_size: int = 12
    font_color: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Drawing settings
    border_width: int = 2
    fill_alpha: float = 0.4  # Transparency for fill
    show_confidence: bool = True
    show_id: bool = True
    
    # Legend settings
    show_legend: bool = True
    legend_position: str = "top-right"


@dataclass 
class ParkingSpotVisualization:
    """Data for visualizing a single parking spot"""
    corners: List[Tuple[float, float]]  # Pixel coordinates
    confidence: float
    is_occupied: bool
    spot_id: str
    parking_type: str = "standard"
    vehicle_confidence: Optional[float] = None  # If occupied, confidence of vehicle detection


class ParkingVisualizer:
    """
    Visualizer for parking detection results
    Overlays parking spots on imagery with occupancy status
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Try to load a nice font, fall back to default if not available
        self.font = None
        try:
            # Try common system fonts
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    self.font = ImageFont.truetype(font_path, self.config.font_size)
                    break
        except Exception:
            pass
        
        if self.font is None:
            self.font = ImageFont.load_default()
    
    def visualize_detections(
        self,
        image: np.ndarray,
        parking_spots: List[ParkingSpotVisualization],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Overlay parking spot detections on image
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            parking_spots: List of parking spots with visualization data
            output_path: Optional path to save the result
            
        Returns:
            Annotated image as numpy array
        """
        # Convert to PIL for easier drawing with transparency
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Create overlay layer with transparency
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Create main drawing layer
        draw = ImageDraw.Draw(pil_image)
        
        # Count for legend
        empty_count = 0
        occupied_count = 0
        
        # Draw each parking spot
        for spot in parking_spots:
            if spot.is_occupied:
                fill_color = self.config.occupied_color
                border_color = self.config.border_color_occupied
                occupied_count += 1
            else:
                fill_color = self.config.empty_color
                border_color = self.config.border_color_empty
                empty_count += 1
            
            # Convert corners to polygon points
            polygon_points = [(int(x), int(y)) for x, y in spot.corners]
            
            # Draw filled polygon with transparency on overlay
            overlay_draw.polygon(polygon_points, fill=fill_color)
            
            # Draw border on main image
            draw.polygon(polygon_points, outline=border_color, width=self.config.border_width)
            
            # Calculate center for text
            center_x = sum(p[0] for p in polygon_points) // len(polygon_points)
            center_y = sum(p[1] for p in polygon_points) // len(polygon_points)
            
            # Draw confidence score and ID
            if self.config.show_confidence or self.config.show_id:
                label_parts = []
                if self.config.show_id:
                    label_parts.append(spot.spot_id)
                if self.config.show_confidence:
                    label_parts.append(f"{spot.confidence:.0%}")
                
                label = " ".join(label_parts)
                
                # Draw text with background for readability
                bbox = draw.textbbox((center_x, center_y), label, font=self.font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Background rectangle
                bg_padding = 2
                bg_x1 = center_x - text_width // 2 - bg_padding
                bg_y1 = center_y - text_height // 2 - bg_padding
                bg_x2 = center_x + text_width // 2 + bg_padding
                bg_y2 = center_y + text_height // 2 + bg_padding
                
                # Draw semi-transparent background
                overlay_draw.rectangle(
                    [bg_x1, bg_y1, bg_x2, bg_y2],
                    fill=(0, 0, 0, 180)
                )
                
                # Draw text
                overlay_draw.text(
                    (center_x - text_width // 2, center_y - text_height // 2),
                    label,
                    fill=(255, 255, 255, 255),
                    font=self.font
                )
        
        # Composite overlay onto main image
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, overlay)
        
        # Draw legend if enabled
        if self.config.show_legend:
            pil_image = self._draw_legend(pil_image, empty_count, occupied_count)
        
        # Convert back to RGB
        result = pil_image.convert('RGB')
        result_array = np.array(result)
        
        # Save if output path provided
        if output_path:
            result.save(output_path)
            logger.info(f"Saved visualization to {output_path}")
        
        return result_array
    
    def _draw_legend(
        self,
        image: Image.Image,
        empty_count: int,
        occupied_count: int
    ) -> Image.Image:
        """Draw legend on image"""
        draw = ImageDraw.Draw(image)
        
        # Legend dimensions
        legend_width = 180
        legend_height = 80
        padding = 10
        
        # Position based on config
        img_width, img_height = image.size
        if self.config.legend_position == "top-right":
            x = img_width - legend_width - padding
            y = padding
        elif self.config.legend_position == "top-left":
            x = padding
            y = padding
        elif self.config.legend_position == "bottom-right":
            x = img_width - legend_width - padding
            y = img_height - legend_height - padding
        else:  # bottom-left
            x = padding
            y = img_height - legend_height - padding
        
        # Draw legend background
        draw.rectangle(
            [x, y, x + legend_width, y + legend_height],
            fill=(0, 0, 0, 200)
        )
        
        # Draw legend title
        draw.text((x + 10, y + 5), "Parking Status", fill=(255, 255, 255, 255), font=self.font)
        
        # Draw empty indicator
        draw.rectangle([x + 10, y + 25, x + 25, y + 40], fill=self.config.empty_color)
        draw.text((x + 35, y + 25), f"Empty: {empty_count}", fill=(255, 255, 255, 255), font=self.font)
        
        # Draw occupied indicator
        draw.rectangle([x + 10, y + 50, x + 25, y + 65], fill=self.config.occupied_color)
        draw.text((x + 35, y + 50), f"Occupied: {occupied_count}", fill=(255, 255, 255, 255), font=self.font)
        
        return image
    
    def visualize_tile_with_detections(
        self,
        tile_image: np.ndarray,
        detections: List[Any],  # Detection objects from parking_detector
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on a single tile
        Uses the is_occupied flag from detections directly (Gemini 3 Flash output)
        
        Args:
            tile_image: The aerial imagery tile
            detections: Parking space detections from the detector (with is_occupied set)
            output_path: Path to save the visualization
            
        Returns:
            Annotated image
        """
        parking_spots = []
        
        # Process each parking detection
        for idx, det in enumerate(detections):
            # Get corners from detection
            if hasattr(det, 'polygon') and det.polygon:
                corners = det.polygon
            elif hasattr(det, 'corners') and det.corners:
                corners = det.corners
            elif hasattr(det, 'bbox'):
                # Convert bbox to corners
                x1, y1, x2, y2 = det.bbox
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                continue
            
            # Create visualization object
            # Use occupancy directly from detection result (Gemini logic)
            is_occupied = getattr(det, 'is_occupied', False)
            spot_id = getattr(det, 'id', f"P{idx+1:03d}")
            confidence = getattr(det, 'confidence', 0.0)
            
            parking_spots.append(ParkingSpotVisualization(
                corners=corners,
                confidence=confidence,
                is_occupied=is_occupied,
                spot_id=spot_id,
            ))
        
        return self.visualize_detections(tile_image, parking_spots, output_path)

