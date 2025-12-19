"""
Test Google Maps Integration
"""
import asyncio
import os
import sys
import logging
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.google_maps_fetcher import GoogleMapsImageryFetcher
from src.main_pipeline import ParkingDetectionOrchestrator, ProcessingConfig
from src.imagery_fetcher import Tile, BoundingBox

# Mock Logger
logging.basicConfig(level=logging.INFO)

async def test_google_maps_fetcher_mock():
    """Test fetcher logic without actual API calls"""
    print("Testing GoogleMapsImageryFetcher (Mocked)...")

    with patch('src.google_maps_fetcher.aiohttp.ClientSession') as mock_session:
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200

        async def mock_read():
            return b"fake_image_data"

        mock_response.read.side_effect = mock_read

        # Mock context manager
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aexit__.return_value = None

        mock_session.return_value.get.return_value = mock_ctx

        # Mock Image.open to return a valid numpy array
        with patch('src.google_maps_fetcher.Image.open') as mock_img:
            fake_img = Image.new('RGB', (640, 640), color='red')
            mock_img.return_value = fake_img

            fetcher = GoogleMapsImageryFetcher(api_key="fake_key")

            # Test 1: Fetch single tile
            tile = await fetcher.fetch_tile(52.3702, 4.8952, zoom=20)

            assert tile is not None
            assert tile.image.shape == (640, 640, 3)
            assert tile.bounds_wgs84 is not None
            print("  - Single tile fetch: OK")
            print(f"  - Calculated Bounds: {tile.bounds_wgs84}")

            # Test 2: Fetch area (grid calculation)
            tiles = await fetcher.fetch_tiles_for_area(52.3702, 4.8952, radius_meters=100)

            assert len(tiles) > 0
            print(f"  - Area fetch: OK (Got {len(tiles)} tiles)")

            await fetcher.close()

async def test_pipeline_integration_mock():
    """Test full pipeline integration with mocks"""
    print("\nTesting Pipeline Integration (Mocked)...")

    # Mock GoogleMapsImageryFetcher within the orchestrator
    with patch('src.main_pipeline.GoogleMapsImageryFetcher') as MockFetcher:
        mock_fetcher_instance = MockFetcher.return_value

        # Create a fake tile with WGS84 bounds
        fake_img = np.zeros((640, 640, 3), dtype=np.uint8)
        fake_tile = Tile(
            x=0, y=0, zoom=20,
            image=fake_img,
            bounds_rd=None,
            bounds_wgs84=BoundingBox(4.89, 52.36, 4.90, 52.38, "EPSG:4326")
        )

        async def mock_fetch_tiles(*args, **kwargs):
            return [fake_tile]
        mock_fetcher_instance.fetch_tiles_for_area.side_effect = mock_fetch_tiles

        # Mock Detection Pipeline
        with patch('src.main_pipeline.ParkingDetectionPipeline') as MockPipeline:
            mock_pipeline_instance = MockPipeline.return_value

            # Mock Detection Result
            from src.parking_detector import DetectionResult, ParkingSpot

            # Create a fake spot in pixel coordinates
            spot = ParkingSpot(
                id="test_spot",
                bbox=(100, 100, 200, 200),
                confidence=0.9,
                polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
                is_occupied=True
            )

            mock_result = DetectionResult(
                total_spots=1,
                empty_spots=0,
                occupied_spots=1,
                occupancy_rate=1.0,
                parking_spots=[spot]
            )

            mock_pipeline_instance.detect.return_value = mock_result

            # Run Orchestrator
            config = ProcessingConfig(google_maps_api_key="fake_key", api_key="fake_gemini")
            orchestrator = ParkingDetectionOrchestrator(config)

            result = await orchestrator.process_google_maps_area(
                lat=52.3702, lon=4.8952, radius=100, output_dir="test_output_google"
            )

            print(f"  - Pipeline result: {result.municipality}")
            print(f"  - Spots detected: {result.total_parking_spaces}")
            print(f"  - GeoJSON path: {result.geojson_path}")

            assert result.total_parking_spaces == 1
            assert result.parking_spaces[0].center is not None
            print("  - Pipeline Integration: OK")

if __name__ == "__main__":
    asyncio.run(test_google_maps_fetcher_mock())
    asyncio.run(test_pipeline_integration_mock())
