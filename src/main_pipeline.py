"""
Main Parking Detection Pipeline
Orchestrates imagery fetching, Gemini 3 Flash detection, and geo-referenced output

Pipeline:
1. Input: Municipality name or bounding box
2. Fetch aerial imagery from PDOK (8cm resolution) OR Google Maps Static API
3. Send to Gemini 3 Flash for two-step parking detection (All spots -> Empty spots)
4. Merge results to determine occupancy
5. Convert to GeoJSON with WGS84 coordinates

Requires: GEMINI_API_KEY or GOOGLE_API_KEY environment variable
For Google Maps: GOOGLE_MAPS_API_KEY
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import os

from src.imagery_fetcher import (
    PDOKImageryFetcher,
    BoundingBox,
    MunicipalityGeocoder,
    CoordinateTransformer
)
from src.google_maps_fetcher import GoogleMapsImageryFetcher

from src.parking_detector import (
    ParkingDetectionPipeline,
    DetectionResult,
    ParkingSpot
)
from src.geo_converter import (
    DetectionToGeoConverter,
    GeoJSONExporter,
    SpatialDeduplicator,
    ParkingSpaceGeo,
    calculate_parking_statistics
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    # Tile settings
    tile_size_pixels: int = 512
    tile_size_meters: float = 50.0
    tile_overlap: float = 0.15
    max_concurrent_tiles: int = 8
    
    # Gemini API settings
    api_key: Optional[str] = None  # Google API key (or set GOOGLE_API_KEY env var)
    model_name: str = "gemini-3-flash-preview"  # Gemini model to use
    
    # Google Maps settings
    google_maps_api_key: Optional[str] = None # set via GOOGLE_MAPS_API_KEY env var

    # Legacy settings (kept for compatibility, ignored by Gemini pipeline)
    parking_model_path: Optional[str] = None
    parking_confidence: float = 0.7
    vehicle_model_size: str = 'n'
    vehicle_confidence: float = 0.7
    overlap_threshold: float = 0.3
    
    # Post-processing
    dedup_distance_meters: float = 1.0
    
    # Output
    resolution: float = 0.08  # 8cm per pixel


@dataclass
class ProcessingResult:
    """Result of parking detection pipeline"""
    municipality: str
    total_area_km2: float
    total_parking_spaces: int
    empty_spaces: int
    occupied_spaces: int
    occupancy_rate: float
    parking_spaces: List[ParkingSpaceGeo]
    statistics: Dict[str, Any]
    processing_time_seconds: float
    tiles_processed: int
    geojson_path: Optional[str] = None


class ParkingDetectionOrchestrator:
    """
    Main orchestrator for the parking detection pipeline
    Handles municipality input to GeoJSON output
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the orchestrator
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.imagery_fetcher = PDOKImageryFetcher(use_high_res=True)
        self.google_fetcher = None

        self.geocoder = MunicipalityGeocoder()
        self.coord_transformer = CoordinateTransformer()
        self.geo_converter = DetectionToGeoConverter(resolution=self.config.resolution)
        self.geojson_exporter = GeoJSONExporter()
        self.deduplicator = SpatialDeduplicator(distance_threshold_meters=self.config.dedup_distance_meters)
        
        # Detection pipeline (lazy loaded)
        self._detection_pipeline = None
        
        logger.info("ParkingDetectionOrchestrator initialized")
    
    def _get_detection_pipeline(self) -> ParkingDetectionPipeline:
        """Get or create Gemini detection pipeline (lazy loading)"""
        if self._detection_pipeline is None:
            logger.info(f"Initializing Gemini detection pipeline with model {self.config.model_name}...")
            self._detection_pipeline = ParkingDetectionPipeline(
                api_key=self.config.api_key,
                model_name=self.config.model_name
            )
        return self._detection_pipeline
    
    def _get_google_fetcher(self) -> GoogleMapsImageryFetcher:
        """Get or create Google Maps fetcher"""
        if self.google_fetcher is None:
            self.google_fetcher = GoogleMapsImageryFetcher(api_key=self.config.google_maps_api_key)
        return self.google_fetcher

    async def process_municipality(
        self,
        municipality_name: str,
        output_dir: Optional[str] = None,
        subset_bbox: Optional[BoundingBox] = None,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Process entire municipality and detect all parking spaces using PDOK imagery
        
        Args:
            municipality_name: Name of Dutch municipality
            output_dir: Directory for output files
            subset_bbox: Optional subset bounding box for testing
            progress_callback: Optional callback(progress, message)
            
        Returns:
            ProcessingResult with all detected parking spaces
        """
        start_time = datetime.now()
        output_dir = Path(output_dir) if output_dir else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing municipality: {municipality_name}")
        
        # Step 1: Get municipality bounding box
        if progress_callback:
            progress_callback(5.0, "Getting municipality boundaries...")
        
        if subset_bbox:
            bbox = subset_bbox
            logger.info(f"Using subset bbox: {bbox}")
        else:
            bbox = await self.geocoder.get_municipality_bbox(municipality_name)
            logger.info(f"Municipality bbox: {bbox}")
        
        # Calculate area
        area_km2 = (bbox.width * bbox.height) / 1_000_000
        logger.info(f"Processing area: {area_km2:.2f} km²")
        
        # Step 2: Fetch tiles
        if progress_callback:
            progress_callback(10.0, "Fetching aerial imagery...")
        
        tiles = await self.imagery_fetcher.fetch_tiles_for_area(
            bbox=bbox,
            tile_size_pixels=self.config.tile_size_pixels,
            tile_size_meters=self.config.tile_size_meters,
            max_concurrent=self.config.max_concurrent_tiles
        )
        
        logger.info(f"Fetched {len(tiles)} tiles")
        
        if not tiles:
            logger.error("No tiles fetched")
            return ProcessingResult(
                municipality=municipality_name,
                total_area_km2=area_km2,
                total_parking_spaces=0,
                empty_spaces=0,
                occupied_spaces=0,
                occupancy_rate=0.0,
                parking_spaces=[],
                statistics={},
                processing_time_seconds=0,
                tiles_processed=0
            )
        
        # Process detection on these tiles (Shared logic)
        return await self._process_tiles(
            tiles=tiles,
            municipality_name=municipality_name,
            output_dir=output_dir,
            area_km2=area_km2,
            start_time=start_time,
            progress_callback=progress_callback
        )

    async def process_google_maps_area(
        self,
        lat: float,
        lon: float,
        radius: float = 200.0,
        name: str = "google_maps_area",
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Process an area using Google Maps imagery
        """
        start_time = datetime.now()
        output_dir = Path(output_dir) if output_dir else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate approximate area
        area_km2 = (3.14159 * radius * radius) / 1_000_000
        logger.info(f"Processing Google Maps area: {name} ({lat}, {lon}) r={radius}m")

        if progress_callback:
            progress_callback(10.0, "Fetching Google Maps imagery...")

        fetcher = self._get_google_fetcher()
        tiles = await fetcher.fetch_tiles_for_area(
            center_lat=lat,
            center_lon=lon,
            radius_meters=radius,
            zoom=21
        )

        logger.info(f"Fetched {len(tiles)} tiles from Google Maps")

        if not tiles:
            logger.error("No tiles fetched from Google Maps")
            return ProcessingResult(
                municipality=name,
                total_area_km2=area_km2,
                total_parking_spaces=0,
                empty_spaces=0,
                occupied_spaces=0,
                occupancy_rate=0.0,
                parking_spaces=[],
                statistics={},
                processing_time_seconds=0,
                tiles_processed=0
            )

        return await self._process_tiles(
            tiles=tiles,
            municipality_name=name,
            output_dir=output_dir,
            area_km2=area_km2,
            start_time=start_time,
            progress_callback=progress_callback,
            source_type="google_maps"
        )

    async def _process_tiles(
        self,
        tiles: List[Any], # List[Tile]
        municipality_name: str,
        output_dir: Path,
        area_km2: float,
        start_time: datetime,
        progress_callback: Optional[callable] = None,
        source_type: str = "pdok"
    ) -> ProcessingResult:
        """
        Common processing logic for tiles (detection, deduplication, export)
        """
        # Step 3: Run detection on all tiles
        if progress_callback:
            progress_callback(20.0, "Running parking detection...")
        
        pipeline = self._get_detection_pipeline()
        all_parking_spaces = []
        total_empty = 0
        total_occupied = 0
        
        for idx, tile in enumerate(tiles):
            if tile.image is None:
                continue
            
            # Update progress
            tile_progress = 20 + (idx / len(tiles)) * 60
            if progress_callback and idx % 5 == 0:
                progress_callback(tile_progress, f"Processing tile {idx+1}/{len(tiles)}")
            
            # Run detection
            result = pipeline.detect(tile.image)
            
            # Convert to geo-referenced parking spaces
            # Check if tile has RD bounds or WGS84 bounds

            if result.parking_spots:
                image_size = tile.image.shape[:2][::-1] # (width, height)
                
                for spot in result.parking_spots:
                    geo_space = self._convert_spot_to_geo(
                        spot=spot,
                        bounds_rd=tile.bounds_rd.to_tuple() if tile.bounds_rd else None,
                        bounds_wgs84=tile.bounds_wgs84.to_tuple() if tile.bounds_wgs84 else None,
                        image_size=image_size,
                        source_tile=f"tile_{idx:04d}"
                    )
                    if geo_space:
                        all_parking_spaces.append(geo_space)
                
                total_empty += result.empty_spots
                total_occupied += result.occupied_spots
        
        logger.info(f"Detected {len(all_parking_spaces)} parking spaces before deduplication")
        
        # Step 4: Deduplicate across tile boundaries
        if progress_callback:
            progress_callback(85.0, "Removing duplicates...")
        
        parking_spaces = self.deduplicator.deduplicate(all_parking_spaces)
        logger.info(f"After deduplication: {len(parking_spaces)} parking spaces")
        
        # Recalculate occupancy counts after dedup
        empty_count = sum(1 for s in parking_spaces if not getattr(s, 'is_occupied', False))
        occupied_count = len(parking_spaces) - empty_count
        occupancy_rate = occupied_count / len(parking_spaces) if parking_spaces else 0.0
        
        # Step 5: Calculate statistics
        if progress_callback:
            progress_callback(90.0, "Calculating statistics...")
        
        statistics = calculate_parking_statistics(parking_spaces)
        statistics['occupancy'] = {
            'empty': empty_count,
            'occupied': occupied_count,
            'rate': occupancy_rate
        }
        
        # Step 6: Export to GeoJSON
        if progress_callback:
            progress_callback(95.0, "Exporting GeoJSON...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_filename = f"{municipality_name.lower()}_{timestamp}.geojson"
        geojson_path = output_dir / geojson_filename
        
        metadata = {
            "municipality": municipality_name,
            "processed_at": datetime.now().isoformat(),
            "area_km2": area_km2,
            "tiles_processed": len(tiles),
            "pipeline": "Gemini 3 Flash (Two-step detection)",
            "source": source_type,
            "config": {
                "model": self.config.model_name,
                "overlap_threshold": self.config.overlap_threshold
            }
        }
        
        self.geojson_exporter.save_geojson(
            parking_spaces=parking_spaces,
            output_path=str(geojson_path),
            metadata=metadata
        )
        
        logger.info(f"GeoJSON saved to {geojson_path}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if progress_callback:
            progress_callback(100.0, "Complete!")
        
        return ProcessingResult(
            municipality=municipality_name,
            total_area_km2=area_km2,
            total_parking_spaces=len(parking_spaces),
            empty_spaces=empty_count,
            occupied_spaces=occupied_count,
            occupancy_rate=occupancy_rate,
            parking_spaces=parking_spaces,
            statistics=statistics,
            processing_time_seconds=processing_time,
            tiles_processed=len(tiles),
            geojson_path=str(geojson_path)
        )
    
    def _convert_spot_to_geo(
        self,
        spot: ParkingSpot,
        bounds_rd: Optional[Tuple[float, float, float, float]],
        image_size: Tuple[int, int],
        source_tile: str,
        bounds_wgs84: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[ParkingSpaceGeo]:
        """Convert a ParkingSpot to geo-referenced ParkingSpaceGeo"""
        try:
            # Convert polygon points to WGS84
            corners_wgs84 = self.geo_converter.converter.corners_to_wgs84(
                corners_pixels=spot.polygon,
                image_bounds_rd=bounds_rd,
                image_size=image_size,
                image_bounds_wgs84=bounds_wgs84
            )
            
            if not corners_wgs84:
                return None
            
            # Calculate center
            center_lon = sum(c[0] for c in corners_wgs84) / len(corners_wgs84)
            center_lat = sum(c[1] for c in corners_wgs84) / len(corners_wgs84)
            
            # Calculate area in square meters
            area_sqm = spot.area_pixels * (self.config.resolution ** 2)
            
            return ParkingSpaceGeo(
                id=spot.id,
                corners=corners_wgs84,
                center=(center_lon, center_lat),
                confidence=spot.confidence,
                parking_type="standard",  # Could be enhanced with classification
                area_sqm=area_sqm,
                source_tile=source_tile,
                detected_at=datetime.now().isoformat(),
                is_occupied=spot.is_occupied
            )
        except Exception as e:
            logger.warning(f"Failed to convert spot to geo: {e}")
            return None
    
    async def process_custom_area(
        self,
        bbox: BoundingBox,
        area_name: str = "custom_area",
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a custom bounding box area (RD coordinates)
        """
        return await self.process_municipality(
            municipality_name=area_name,
            output_dir=output_dir,
            subset_bbox=bbox
        )
    
    async def close(self):
        """Clean up resources"""
        await self.imagery_fetcher.close()
        if self.google_fetcher:
            await self.google_fetcher.close()
        await self.geocoder.close()


class BatchProcessor:
    """Process multiple municipalities in batch"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    async def process_batch(
        self,
        municipalities: List[str],
        output_dir: str,
        max_concurrent: int = 1
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple municipalities
        
        Args:
            municipalities: List of municipality names
            output_dir: Base output directory
            max_concurrent: Max municipalities to process concurrently
            
        Returns:
            Dictionary mapping municipality name to results
        """
        results = {}
        
        async def process_one(name: str) -> Tuple[str, ProcessingResult]:
            orchestrator = ParkingDetectionOrchestrator(self.config)
            try:
                result = await orchestrator.process_municipality(
                    name,
                    output_dir=output_dir
                )
                return name, result
            finally:
                await orchestrator.close()
        
        # Process sequentially for now (can be parallelized later)
        for name in municipalities:
            logger.info(f"Processing {name}...")
            try:
                _, result = await process_one(name)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to process {name}: {e}")
        
        return results


# Convenience functions for CLI usage
async def detect_parking_spaces(
    municipality: str,
    output_dir: str = "output",
    test_mode: bool = False,
    parking_model_path: Optional[str] = None
) -> ProcessingResult:
    """
    Main entry point for parking space detection (Legacy wrapper for PDOK)
    """
    config = ProcessingConfig(
        parking_model_path=parking_model_path,
        tile_size_meters=40.0 if not test_mode else 100.0
    )
    
    orchestrator = ParkingDetectionOrchestrator(config)
    
    try:
        if test_mode:
            # Get municipality bbox and create 1km² test area
            geocoder = MunicipalityGeocoder()
            full_bbox = await geocoder.get_municipality_bbox(municipality)
            await geocoder.close()
            
            # Create 1km² test area in center
            cx, cy = full_bbox.center
            test_bbox = BoundingBox(
                west=cx - 500,
                south=cy - 500,
                east=cx + 500,
                north=cy + 500,
                crs="EPSG:28992"
            )
            
            logger.info(f"Test mode: Processing 1km² area at center of {municipality}")
            
            result = await orchestrator.process_municipality(
                municipality,
                output_dir=output_dir,
                subset_bbox=test_bbox
            )
        else:
            result = await orchestrator.process_municipality(
                municipality,
                output_dir=output_dir
            )
        
        return result
        
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect parking spaces in Dutch municipalities or via Google Maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a municipality (PDOK)
  python -m src.main_pipeline Amersfoort --test --output output/
  
  # Process Google Maps area (lat/lon)
  python -m src.main_pipeline --source google --lat 52.3702 --lon 4.8952 --radius 100
        """
    )
    
    parser.add_argument(
        "municipality",
        nargs="?",
        default="Amersfoort",
        help="Municipality name (default: Amersfoort, ignored if --source google)"
    )

    parser.add_argument(
        "--source",
        choices=["pdok", "google"],
        default="pdok",
        help="Imagery source (default: pdok)"
    )

    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude (for Google Maps source)"
    )

    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude (for Google Maps source)"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=200.0,
        help="Radius in meters (for Google Maps source)"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: process only 1km² area (PDOK only)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--parking-model",
        help="Path to trained parking detection model"
    )
    
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help="Custom bounding box in RD coordinates (PDOK only)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("🅿️ PARKING SPACE DETECTION SYSTEM")
    print("="*60)
    print(f"\nPipeline: Gemini 3 Flash (Two-step detection)")
    
    config = ProcessingConfig(parking_model_path=args.parking_model)
    orchestrator = ParkingDetectionOrchestrator(config)

    try:
        if args.source == "google":
            if args.lat is None or args.lon is None:
                print("Error: --lat and --lon are required for source 'google'")
                exit(1)

            print(f"\nProcessing Google Maps area: {args.lat}, {args.lon}")
            result = asyncio.run(orchestrator.process_google_maps_area(
                lat=args.lat,
                lon=args.lon,
                radius=args.radius,
                output_dir=args.output,
                name=f"google_{args.lat}_{args.lon}"
            ))

        elif args.bbox:
            print(f"\nProcessing custom bounding box (RD)...")
            bbox = BoundingBox(
                west=args.bbox[0],
                south=args.bbox[1],
                east=args.bbox[2],
                north=args.bbox[3]
            )

            result = asyncio.run(orchestrator.process_custom_area(
                bbox=bbox,
                area_name="custom_area",
                output_dir=args.output
            ))
        else:
            print(f"\nProcessing municipality (PDOK): {args.municipality}")
            print(f"Test mode: {args.test}")

            result = asyncio.run(orchestrator.process_municipality(
                args.municipality,
                output_dir=args.output,
                subset_bbox=None # Handle inside process_municipality for test mode if we were calling the cli func, but here we call orchestrator directly.
            ))
            # Note: The CLI args.test logic was inside the `detect_parking_spaces` helper.
            # If using orchestrator directly, we need to replicate that logic if we want it.
            # For now, sticking to the PDOK standard path for municipality arg.
            if args.test:
                 # Re-implement test mode logic here since we bypassed helper
                geocoder = MunicipalityGeocoder()
                full_bbox = asyncio.run(geocoder.get_municipality_bbox(args.municipality))
                asyncio.run(geocoder.close())
                cx, cy = full_bbox.center
                test_bbox = BoundingBox(
                    west=cx - 500, south=cy - 500, east=cx + 500, north=cy + 500, crs="EPSG:28992"
                )
                result = asyncio.run(orchestrator.process_municipality(
                     args.municipality, output_dir=args.output, subset_bbox=test_bbox
                ))
            else:
                 result = asyncio.run(orchestrator.process_municipality(
                     args.municipality, output_dir=args.output
                ))

        # Print results
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        print(f"Name: {result.municipality}")
        print(f"Area: {result.total_area_km2:.6f} km²")
        print(f"Tiles processed: {result.tiles_processed}")
        print(f"Total parking spaces: {result.total_parking_spaces}")
        print(f"Empty spaces: {result.empty_spaces}")
        print(f"Occupied spaces: {result.occupied_spaces}")
        print(f"Occupancy rate: {result.occupancy_rate:.1%}")
        print(f"Processing time: {result.processing_time_seconds:.1f} seconds")
        print(f"GeoJSON: {result.geojson_path}")
        
    finally:
        asyncio.run(orchestrator.close())
