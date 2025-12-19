"""
Geo-coordinate Transformation and GeoJSON Output
Converts pixel detections to WGS84 coordinates and exports as GeoJSON
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pyproj import Transformer
from rasterio.transform import from_bounds
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeoCoordinate:
    """Geographic coordinate in WGS84"""
    longitude: float
    latitude: float
    
    def to_list(self) -> List[float]:
        """Return as [lon, lat] list (GeoJSON format)"""
        return [self.longitude, self.latitude]


@dataclass 
class ParkingSpaceGeo:
    """Parking space with geographic coordinates"""
    # Unique identifier
    id: str
    
    # 4 corner coordinates in WGS84 [(lon, lat), ...]
    corners: List[Tuple[float, float]]
    
    # Center point
    center: Tuple[float, float]
    
    # Confidence score
    confidence: float
    
    # Parking type classification
    parking_type: str
    
    # Area in square meters
    area_sqm: float
    
    # Source tile information
    source_tile: Optional[str] = None
    
    # Detection timestamp
    detected_at: Optional[str] = None
    
    # Original image dimensions
    dimensions_meters: Optional[Tuple[float, float]] = None
    
    # Occupancy status (True = vehicle present, False = empty)
    is_occupied: bool = False


class CoordinateConverter:
    """
    Converts between pixel coordinates and geographic coordinates
    Handles Dutch RD (EPSG:28992) to WGS84 (EPSG:4326) transformation
    """
    
    def __init__(self):
        # Initialize transformers
        self.rd_to_wgs84 = Transformer.from_crs(
            "EPSG:28992", "EPSG:4326", always_xy=True
        )
        self.wgs84_to_rd = Transformer.from_crs(
            "EPSG:4326", "EPSG:28992", always_xy=True
        )
    
    def pixel_to_rd(
        self,
        pixel_x: float,
        pixel_y: float,
        image_bounds_rd: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to Dutch RD coordinates
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels  
            image_bounds_rd: (west, south, east, north) in RD
            image_size: (width, height) in pixels
            
        Returns:
            (x_rd, y_rd) in Dutch RD coordinates
        """
        west, south, east, north = image_bounds_rd
        width, height = image_size
        
        # Create affine transform
        transform = from_bounds(west, south, east, north, width, height)
        
        # Apply transform (note: rasterio uses row, col order)
        x_rd, y_rd = transform * (pixel_x, pixel_y)
        
        return x_rd, y_rd
    
    def pixel_to_wgs84(
        self,
        pixel_x: float,
        pixel_y: float,
        image_bounds_rd: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to WGS84 lat/lon
        
        Returns:
            (longitude, latitude)
        """
        x_rd, y_rd = self.pixel_to_rd(pixel_x, pixel_y, image_bounds_rd, image_size)
        lon, lat = self.rd_to_wgs84.transform(x_rd, y_rd)
        return lon, lat
    
    def corners_to_wgs84(
        self,
        corners_pixels: List[Tuple[float, float]],
        image_bounds_rd: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Convert list of corner points from pixels to WGS84
        
        Args:
            corners_pixels: List of (x, y) pixel coordinates
            image_bounds_rd: Image bounds in RD coordinates
            image_size: Image dimensions in pixels
            
        Returns:
            List of (longitude, latitude) tuples
        """
        corners_wgs84 = []
        for px, py in corners_pixels:
            lon, lat = self.pixel_to_wgs84(px, py, image_bounds_rd, image_size)
            corners_wgs84.append((lon, lat))
        return corners_wgs84
    
    def rotated_rect_to_corners(
        self,
        rotated_rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> List[Tuple[float, float]]:
        """
        Convert OpenCV rotated rectangle to 4 corner points
        
        Args:
            rotated_rect: ((cx, cy), (w, h), angle)
            
        Returns:
            List of 4 corner points in pixel coordinates
        """
        import cv2
        
        rect = rotated_rect
        box = cv2.boxPoints(rect)
        return [(float(p[0]), float(p[1])) for p in box]


class GeoJSONExporter:
    """
    Exports parking space detections to GeoJSON format
    """
    
    def __init__(self, precision: int = 8):
        """
        Args:
            precision: Decimal places for coordinates
        """
        self.precision = precision
        self.converter = CoordinateConverter()
    
    def detection_to_geojson_feature(
        self,
        parking_space: ParkingSpaceGeo
    ) -> Dict[str, Any]:
        """
        Convert single parking space to GeoJSON Feature
        """
        # Close the polygon (first and last point must match)
        coordinates = parking_space.corners + [parking_space.corners[0]]
        
        # Round coordinates
        coordinates = [
            [round(lon, self.precision), round(lat, self.precision)]
            for lon, lat in coordinates
        ]
        
        feature = {
            "type": "Feature",
            "id": parking_space.id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": {
                "id": parking_space.id,
                "confidence": round(parking_space.confidence, 3),
                "parking_type": parking_space.parking_type,
                "area_sqm": round(parking_space.area_sqm, 2) if parking_space.area_sqm else None,
                "center_lon": round(parking_space.center[0], self.precision),
                "center_lat": round(parking_space.center[1], self.precision),
                "is_occupied": parking_space.is_occupied,
                "status": "occupied" if parking_space.is_occupied else "empty",
                "detected_at": parking_space.detected_at,
                "source_tile": parking_space.source_tile,
                # Map links for verification
                "map_links": self._generate_map_links(parking_space.center)
            }
        }
        
        if parking_space.dimensions_meters:
            feature["properties"]["length_m"] = round(parking_space.dimensions_meters[0], 2)
            feature["properties"]["width_m"] = round(parking_space.dimensions_meters[1], 2)
        
        return feature
    
    def _generate_map_links(self, center: Tuple[float, float]) -> Dict[str, str]:
        """
        Generate links to various map viewers for the parking spot
        
        Args:
            center: (longitude, latitude) in WGS84
            
        Returns:
            Dictionary with map service names and URLs
        """
        lon, lat = center
        zoom = 19  # High zoom level for parking spots
        
        return {
            # PDOK Viewer - Dutch government map with aerial imagery
            "pdok_viewer": f"https://www.pdok.nl/viewer/?lat={lat}&lon={lon}&zoom={zoom}",
            
            # PDOK Luchtfoto direct link
            "pdok_luchtfoto": f"https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Actueel_orthoHR&STYLE=default&TILEMATRIXSET=EPSG:28992&FORMAT=image/jpeg",
            
            # OpenStreetMap
            "openstreetmap": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}",
            
            # Google Maps
            "google_maps": f"https://www.google.com/maps/@{lat},{lon},{zoom}z",
            
            # Google Maps Satellite view
            "google_satellite": f"https://www.google.com/maps/@{lat},{lon},{zoom}z/data=!3m1!1e3",
            
            # Bing Maps Aerial
            "bing_aerial": f"https://www.bing.com/maps?cp={lat}~{lon}&lvl={zoom}&style=a"
        }
    
    def export_to_geojson(
        self,
        parking_spaces: List[ParkingSpaceGeo],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export list of parking spaces to GeoJSON FeatureCollection
        
        Args:
            parking_spaces: List of parking spaces with geo coordinates
            metadata: Optional metadata to include
            
        Returns:
            GeoJSON FeatureCollection dictionary
        """
        features = [
            self.detection_to_geojson_feature(ps)
            for ps in parking_spaces
        ]
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Add metadata
        if metadata:
            geojson["metadata"] = metadata
        
        # Add summary statistics
        geojson["properties"] = {
            "total_spaces": len(parking_spaces),
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "coordinate_system": "WGS84 (EPSG:4326)"
        }
        
        # Count by type
        type_counts = {}
        for ps in parking_spaces:
            ptype = ps.parking_type
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        geojson["properties"]["counts_by_type"] = type_counts
        
        # Occupancy statistics
        empty_count = sum(1 for ps in parking_spaces if not ps.is_occupied)
        occupied_count = len(parking_spaces) - empty_count
        occupancy_rate = occupied_count / len(parking_spaces) if parking_spaces else 0.0
        
        geojson["properties"]["occupancy"] = {
            "empty": empty_count,
            "occupied": occupied_count,
            "rate": round(occupancy_rate, 3)
        }
        
        # Total area
        total_area = sum(ps.area_sqm for ps in parking_spaces if ps.area_sqm)
        geojson["properties"]["total_area_sqm"] = round(total_area, 2)
        
        return geojson
    
    def save_geojson(
        self,
        parking_spaces: List[ParkingSpaceGeo],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save parking spaces to GeoJSON file"""
        geojson = self.export_to_geojson(parking_spaces, metadata)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved {len(parking_spaces)} parking spaces to {output_path}")


class DetectionToGeoConverter:
    """
    Converts raw detections from ML models to geo-referenced parking spaces
    """
    
    def __init__(self, resolution: float = 0.08):
        """
        Args:
            resolution: Image resolution in meters/pixel
        """
        self.resolution = resolution
        self.coord_converter = CoordinateConverter()
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique parking space ID"""
        self._id_counter += 1
        return f"PS_{self._id_counter:06d}"
    
    def convert_detection(
        self,
        detection: Any,  # Detection object from parking_detector
        image_bounds_rd: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        source_tile: Optional[str] = None
    ) -> Optional[ParkingSpaceGeo]:
        """
        Convert a single detection to geo-referenced parking space
        
        Args:
            detection: Detection object with bbox, rotated_rect, corners
            image_bounds_rd: (west, south, east, north) in Dutch RD
            image_size: (width, height) in pixels
            source_tile: Identifier for source tile
            
        Returns:
            ParkingSpaceGeo object or None if conversion fails
        """
        try:
            # Get corners from detection
            if detection.corners:
                corners_pixels = detection.corners
            elif detection.rotated_rect:
                corners_pixels = self.coord_converter.rotated_rect_to_corners(
                    detection.rotated_rect
                )
            else:
                # Fall back to axis-aligned bbox
                x1, y1, x2, y2 = detection.bbox
                corners_pixels = [
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                ]
            
            # Convert to WGS84
            corners_wgs84 = self.coord_converter.corners_to_wgs84(
                corners_pixels, image_bounds_rd, image_size
            )
            
            # Calculate center
            center_lon = sum(c[0] for c in corners_wgs84) / 4
            center_lat = sum(c[1] for c in corners_wgs84) / 4
            
            # Calculate dimensions in meters
            dimensions = None
            if detection.rotated_rect:
                w_px, h_px = detection.rotated_rect[1]
                w_m = w_px * self.resolution
                h_m = h_px * self.resolution
                length = max(w_m, h_m)
                width = min(w_m, h_m)
                dimensions = (length, width)
            
            # Calculate area
            area_sqm = detection.area_sqm
            if area_sqm is None and dimensions:
                area_sqm = dimensions[0] * dimensions[1]
            
            return ParkingSpaceGeo(
                id=self._generate_id(),
                corners=corners_wgs84,
                center=(center_lon, center_lat),
                confidence=detection.confidence,
                parking_type=getattr(detection, 'parking_type', 'unknown'),
                area_sqm=area_sqm or 0.0,
                source_tile=source_tile,
                detected_at=datetime.utcnow().isoformat() + "Z",
                dimensions_meters=dimensions
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert detection: {e}")
            return None
    
    def convert_all_detections(
        self,
        detections: List[Any],
        image_bounds_rd: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        source_tile: Optional[str] = None
    ) -> List[ParkingSpaceGeo]:
        """
        Convert all detections from a tile to geo-referenced parking spaces
        """
        parking_spaces = []
        
        for det in detections:
            ps = self.convert_detection(det, image_bounds_rd, image_size, source_tile)
            if ps:
                parking_spaces.append(ps)
        
        return parking_spaces


class SpatialDeduplicator:
    """
    Removes duplicate detections from overlapping tiles
    """
    
    def __init__(self, distance_threshold_meters: float = 1.0):
        """
        Args:
            distance_threshold_meters: Distance threshold for considering duplicates
        """
        self.threshold = distance_threshold_meters
    
    def haversine_distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """
        Calculate distance between two WGS84 coordinates in meters
        
        Args:
            coord1: (longitude, latitude)
            coord2: (longitude, latitude)
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lon1, lat1 = radians(coord1[0]), radians(coord1[1])
        lon2, lat2 = radians(coord2[0]), radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def deduplicate(
        self,
        parking_spaces: List[ParkingSpaceGeo]
    ) -> List[ParkingSpaceGeo]:
        """
        Remove duplicate parking spaces based on center distance
        Keeps the detection with highest confidence
        
        Args:
            parking_spaces: List of all parking spaces
            
        Returns:
            Deduplicated list
        """
        if not parking_spaces:
            return []
        
        # Sort by confidence (highest first)
        sorted_spaces = sorted(
            parking_spaces, 
            key=lambda ps: ps.confidence, 
            reverse=True
        )
        
        kept = []
        used_indices = set()
        
        for i, ps in enumerate(sorted_spaces):
            if i in used_indices:
                continue
            
            kept.append(ps)
            used_indices.add(i)
            
            # Mark nearby spaces as duplicates
            for j, other in enumerate(sorted_spaces[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                dist = self.haversine_distance(ps.center, other.center)
                if dist < self.threshold:
                    used_indices.add(j)
        
        logger.info(f"Deduplicated {len(parking_spaces)} -> {len(kept)} parking spaces")
        return kept


def calculate_parking_statistics(
    parking_spaces: List[ParkingSpaceGeo]
) -> Dict[str, Any]:
    """
    Calculate statistics for detected parking spaces
    
    Returns:
        Dictionary with various statistics
    """
    if not parking_spaces:
        return {"total_count": 0}
    
    stats = {
        "total_count": len(parking_spaces),
        "counts_by_type": {},
        "confidence": {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0
        },
        "area": {
            "total_sqm": 0.0,
            "mean_sqm": 0.0,
            "min_sqm": 0.0,
            "max_sqm": 0.0
        }
    }
    
    # Count by type
    for ps in parking_spaces:
        ptype = ps.parking_type
        stats["counts_by_type"][ptype] = stats["counts_by_type"].get(ptype, 0) + 1
    
    # Confidence stats
    confidences = [ps.confidence for ps in parking_spaces]
    stats["confidence"]["mean"] = round(np.mean(confidences), 3)
    stats["confidence"]["min"] = round(min(confidences), 3)
    stats["confidence"]["max"] = round(max(confidences), 3)
    
    # Area stats
    areas = [ps.area_sqm for ps in parking_spaces if ps.area_sqm]
    if areas:
        stats["area"]["total_sqm"] = round(sum(areas), 2)
        stats["area"]["mean_sqm"] = round(np.mean(areas), 2)
        stats["area"]["min_sqm"] = round(min(areas), 2)
        stats["area"]["max_sqm"] = round(max(areas), 2)
    
    # Calculate bounding box
    all_lons = [ps.center[0] for ps in parking_spaces]
    all_lats = [ps.center[1] for ps in parking_spaces]
    
    stats["bounding_box"] = {
        "west": round(min(all_lons), 6),
        "south": round(min(all_lats), 6),
        "east": round(max(all_lons), 6),
        "north": round(max(all_lats), 6)
    }
    
    return stats
