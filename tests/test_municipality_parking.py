"""
Test script for Municipality Parking Detection with Geographic Coordinates
Fetches top 10 aerial images from a specified Dutch municipality,
runs Gemini parking detection, and outputs results with real WGS84 coordinates
compatible with Google Maps.
"""
import asyncio
import os
import logging
import argparse
import json
from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Adjust path to allow imports from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.imagery_fetcher import PDOKImageryFetcher, BoundingBox, MunicipalityGeocoder, Tile
from src.parking_detector import ParkingDetectionPipeline, DetectionResult
from src.visualizer import ParkingVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MunicipalityTest")

# Gemini API Key
GEMINI_API_KEY = "AIzaSyC5ZGOoIeAMgOg0KosdOq10bX4jNIGn_8k"

# Default settings
DEFAULT_MUNICIPALITY = "Amersfoort"
DEFAULT_NUM_IMAGES = 10
DEFAULT_TILE_SIZE_METERS = 50.0


def pixel_to_wgs84(
    pixel_x: int,
    pixel_y: int,
    image_width: int,
    image_height: int,
    bounds_wgs84: BoundingBox
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to WGS84 (lat/lon) coordinates.
    
    Args:
        pixel_x: X coordinate in pixels (from left)
        pixel_y: Y coordinate in pixels (from top)
        image_width: Image width in pixels
        image_height: Image height in pixels
        bounds_wgs84: Geographic bounds of the image in WGS84
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Normalize pixel coordinates to 0-1 range
    x_ratio = pixel_x / image_width
    y_ratio = pixel_y / image_height  # Note: images have origin at top-left
    
    # Convert to geographic coordinates
    # Longitude increases left to right (west to east)
    lon = bounds_wgs84.west + x_ratio * (bounds_wgs84.east - bounds_wgs84.west)
    
    # Latitude decreases top to bottom (north to south) - hence (1 - y_ratio)
    lat = bounds_wgs84.north - y_ratio * (bounds_wgs84.north - bounds_wgs84.south)
    
    return lat, lon


def bbox_pixel_to_wgs84(
    bbox_pixels: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    bounds_wgs84: BoundingBox
) -> Dict[str, float]:
    """
    Convert a pixel bounding box to WGS84 coordinates.
    
    Returns:
        Dict with 'north', 'south', 'east', 'west', 'center_lat', 'center_lon'
    """
    x1, y1, x2, y2 = bbox_pixels
    
    # Get corner coordinates
    lat1, lon1 = pixel_to_wgs84(x1, y1, image_width, image_height, bounds_wgs84)
    lat2, lon2 = pixel_to_wgs84(x2, y2, image_width, image_height, bounds_wgs84)
    
    # Calculate center
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2
    
    return {
        "north": max(lat1, lat2),
        "south": min(lat1, lat2),
        "east": max(lon1, lon2),
        "west": min(lon1, lon2),
        "center_lat": center_lat,
        "center_lon": center_lon
    }


async def fetch_municipality_tiles(
    municipality: str,
    num_images: int = 10,
    tile_size_meters: float = 50.0
) -> List[Tile]:
    """
    Fetch aerial tiles from the center of a municipality.
    Returns Tile objects with geographic bounds.
    """
    logger.info(f"Fetching {num_images} tiles from {municipality}...")
    
    fetcher = PDOKImageryFetcher(use_high_res=True)
    geocoder = MunicipalityGeocoder()
    
    try:
        bbox = await geocoder.get_municipality_bbox(municipality)
        logger.info(f"Municipality bbox: {bbox}")
        
        import math
        grid_size = math.ceil(math.sqrt(num_images))
        area_size = grid_size * tile_size_meters * 1.1
        
        center_x, center_y = bbox.center
        test_bbox = BoundingBox(
            west=center_x - area_size / 2,
            south=center_y - area_size / 2,
            east=center_x + area_size / 2,
            north=center_y + area_size / 2,
            crs=bbox.crs
        )
        
        logger.info(f"Fetching tiles from {area_size:.0f}m x {area_size:.0f}m area")
        
        tiles = await fetcher.fetch_tiles_for_area(
            bbox=test_bbox,
            tile_size_pixels=512,
            tile_size_meters=tile_size_meters,
            max_concurrent=8
        )
        
        valid_tiles = [t for t in tiles if t.image is not None and t.bounds_wgs84 is not None]
        
        if len(valid_tiles) < num_images:
            logger.warning(f"Only found {len(valid_tiles)} valid tiles")
            return valid_tiles
        
        logger.info(f"Fetched {len(valid_tiles)} tiles, returning top {num_images}")
        return valid_tiles[:num_images]
        
    finally:
        await fetcher.close()
        await geocoder.close()


def print_result(idx: int, result: DetectionResult):
    """Print the detection results"""
    print(f"\n--- Tile {idx+1} ---")
    print(f"Total Spots: {result.total_spots}")
    print(f"  - Empty:    {result.empty_spots}")
    print(f"  - Occupied: {result.occupied_spots}")
    print(f"Occupancy Rate: {result.occupancy_rate:.1%}")


async def test_municipality_parking(
    municipality: str = DEFAULT_MUNICIPALITY,
    num_images: int = DEFAULT_NUM_IMAGES,
    tile_size_meters: float = DEFAULT_TILE_SIZE_METERS,
    output_dir: str = "municipality_output"
):
    """
    Main test function with geographic coordinates output.
    """
    print(f"\n{'='*60}")
    print(f"MUNICIPALITY PARKING DETECTION (with GPS Coordinates)")
    print(f"Municipality: {municipality}")
    print(f"Number of tiles: {num_images}")
    print(f"Tile size: {tile_size_meters}m x {tile_size_meters}m")
    print(f"{'='*60}\n")
    
    # 1. Fetch Tiles (with geographic bounds)
    tiles = await fetch_municipality_tiles(
        municipality=municipality,
        num_images=num_images,
        tile_size_meters=tile_size_meters
    )
    
    if not tiles:
        logger.error("No tiles fetched. Aborting test.")
        return

    logger.info(f"Successfully fetched {len(tiles)} tiles with geographic bounds.")
    
    # Extract images for detection
    images = [t.image for t in tiles]

    # 2. Initialize Pipeline and Visualizer
    pipeline = ParkingDetectionPipeline(
        api_key=GEMINI_API_KEY,
        model_name="gemini-3-flash-preview"
    )
    visualizer = ParkingVisualizer()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 3. Run Batch Detection
    logger.info("Starting batch detection...")
    
    def progress(curr, total, msg):
        logger.info(f"[{curr}/{total}] {msg}")

    results = pipeline.detect_batch(images, progress_callback=progress)

    # 4. Build JSON output with geographic coordinates
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    json_output = {
        "municipality": municipality,
        "tile_size_meters": tile_size_meters,
        "coordinate_system": "WGS84 (EPSG:4326)",
        "google_maps_compatible": True,
        "summary": {
            "total_tiles": len(tiles),
            "total_spots": 0,
            "total_empty": 0,
            "total_occupied": 0
        },
        "tiles": [],
        "all_empty_spots": [],  # Flat list of all empty spots for easy mapping
        "all_occupied_spots": []  # Flat list of all occupied spots
    }
    
    total_spots = 0
    total_empty = 0
    total_occupied = 0
    
    for i, (tile, res) in enumerate(zip(tiles, results)):
        print_result(i, res)
        total_spots += res.total_spots
        total_empty += res.empty_spots
        total_occupied += res.occupied_spots
        
        # Save visualization
        output_file = output_path / f"tile_{i+1:03d}.png"
        visualizer.visualize_tile_with_detections(
            tile_image=tile.image,
            detections=res.parking_spots,
            output_path=str(output_file)
        )
        
        # Get image dimensions
        img_height, img_width = tile.image.shape[:2]
        
        # Build tile data with geographic coordinates
        tile_data = {
            "tile_index": i + 1,
            "tile_bounds_wgs84": {
                "north": tile.bounds_wgs84.north,
                "south": tile.bounds_wgs84.south,
                "east": tile.bounds_wgs84.east,
                "west": tile.bounds_wgs84.west,
                "center_lat": (tile.bounds_wgs84.north + tile.bounds_wgs84.south) / 2,
                "center_lon": (tile.bounds_wgs84.east + tile.bounds_wgs84.west) / 2
            },
            "statistics": {
                "total_spots": res.total_spots,
                "empty_spots": res.empty_spots,
                "occupied_spots": res.occupied_spots,
                "occupancy_rate": res.occupancy_rate
            },
            "parking_spots": []
        }
        
        for spot in res.parking_spots:
            # Convert pixel bbox to WGS84
            geo_coords = bbox_pixel_to_wgs84(
                spot.bbox,
                img_width,
                img_height,
                tile.bounds_wgs84
            )
            
            spot_data = {
                "id": spot.id,
                "is_occupied": spot.is_occupied,
                "status": "occupied" if spot.is_occupied else "empty",
                "confidence": spot.confidence,
                "pixel_bbox": list(spot.bbox),
                "geo_coordinates": {
                    "center_lat": geo_coords["center_lat"],
                    "center_lon": geo_coords["center_lon"],
                    "bounds": {
                        "north": geo_coords["north"],
                        "south": geo_coords["south"],
                        "east": geo_coords["east"],
                        "west": geo_coords["west"]
                    }
                },
                "google_maps_url": f"https://www.google.com/maps?q={geo_coords['center_lat']},{geo_coords['center_lon']}"
            }
            tile_data["parking_spots"].append(spot_data)
            
            # Add to flat lists for easy access
            flat_spot = {
                "tile_index": i + 1,
                "id": spot.id,
                "lat": geo_coords["center_lat"],
                "lon": geo_coords["center_lon"],
                "confidence": spot.confidence,
                "google_maps_url": spot_data["google_maps_url"]
            }
            
            if spot.is_occupied:
                json_output["all_occupied_spots"].append(flat_spot)
            else:
                json_output["all_empty_spots"].append(flat_spot)
        
        json_output["tiles"].append(tile_data)
    
    # Update summary
    json_output["summary"]["total_spots"] = total_spots
    json_output["summary"]["total_empty"] = total_empty
    json_output["summary"]["total_occupied"] = total_occupied
    if total_spots > 0:
        json_output["summary"]["overall_occupancy_rate"] = total_occupied / total_spots
    else:
        json_output["summary"]["overall_occupancy_rate"] = 0.0
    
    # Save JSON output
    json_path = output_path / "detection_results_geo.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Municipality: {municipality}")
    print(f"Tiles processed: {len(tiles)}")
    print(f"Total parking spots: {total_spots}")
    print(f"  - Empty:    {total_empty}")
    print(f"  - Occupied: {total_occupied}")
    if total_spots > 0:
        print(f"Occupancy rate: {total_occupied/total_spots:.1%}")
    print(f"\nOutput saved to: {output_path.absolute()}")
    print(f"JSON with GPS coordinates: {json_path}")
    print("="*60)
    
    # Print empty spots with Google Maps links
    print(f"\n{'='*60}")
    print("EMPTY PARKING SPOTS (Google Maps Links)")
    print("="*60)
    
    if json_output["all_empty_spots"]:
        for spot in json_output["all_empty_spots"]:
            print(f"\nTile {spot['tile_index']}, Spot {spot['id']}:")
            print(f"  Location: {spot['lat']:.6f}, {spot['lon']:.6f}")
            print(f"  Google Maps: {spot['google_maps_url']}")
    else:
        print("No empty spots detected.")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect parking with real GPS coordinates for Google Maps"
    )
    parser.add_argument(
        "--municipality", "-m",
        type=str,
        default=DEFAULT_MUNICIPALITY,
        help=f"Municipality name (default: {DEFAULT_MUNICIPALITY})"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of tiles (default: {DEFAULT_NUM_IMAGES})"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=float,
        default=DEFAULT_TILE_SIZE_METERS,
        help=f"Tile size in meters (default: {DEFAULT_TILE_SIZE_METERS})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="municipality_output",
        help="Output directory (default: municipality_output)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_municipality_parking(
        municipality=args.municipality,
        num_images=args.num_images,
        tile_size_meters=args.tile_size,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()
