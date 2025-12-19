"""
Google Maps Imagery Fetcher
Fetches satellite imagery from Google Static Maps API
"""
import asyncio
import aiohttp
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass
import math
import logging
import os

from src.imagery_fetcher import Tile, BoundingBox

logger = logging.getLogger(__name__)

class GoogleMapsImageryFetcher:
    """Fetches satellite imagery from Google Static Maps API"""

    BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
    TILE_SIZE = 640  # Max free size for standard plan, Premium can go higher but 640 is safe default

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Maps fetcher

        Args:
            api_key: Google Maps Static API Key (or set GOOGLE_MAPS_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("No GOOGLE_MAPS_API_KEY found. Fetching will fail.")

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _lat_lon_to_point(self, lat: float, lon: float, zoom: int) -> Tuple[float, float]:
        """Convert lat/lon to world coordinates at specific zoom level"""
        # World coordinates (0-256 for zoom 0)
        tile_size = 256
        sin_y = math.sin(lat * math.pi / 180)
        sin_y = min(max(sin_y, -0.9999), 0.9999)

        x = tile_size * (0.5 + lon / 360)
        y = tile_size * (0.5 - math.log((1 + sin_y) / (1 - sin_y)) / (4 * math.pi))

        scale = 2 ** zoom
        return x * scale, y * scale

    def _point_to_lat_lon(self, x: float, y: float, zoom: int) -> Tuple[float, float]:
        """Convert world coordinates to lat/lon"""
        tile_size = 256
        scale = 2 ** zoom

        unscaled_x = x / scale
        unscaled_y = y / scale

        lon = (unscaled_x / tile_size - 0.5) * 360
        lat_rad = (math.exp((0.5 - unscaled_y / tile_size) * 4 * math.pi) - 1) / \
                  (math.exp((0.5 - unscaled_y / tile_size) * 4 * math.pi) + 1)
        lat = math.asin(lat_rad) * 180 / math.pi

        return lat, lon

    def _calculate_image_bounds(self, center_lat: float, center_lon: float, zoom: int, width: int, height: int) -> BoundingBox:
        """
        Calculate the exact WGS84 bounding box for a Google Static Maps image
        given its center, zoom, and dimensions.
        """
        center_x, center_y = self._lat_lon_to_point(center_lat, center_lon, zoom)

        # Calculate corners in world pixel coordinates
        west_x = center_x - width / 2
        east_x = center_x + width / 2
        north_y = center_y - height / 2
        south_y = center_y + height / 2

        # Convert back to lat/lon
        north_lat, west_lon = self._point_to_lat_lon(west_x, north_y, zoom)
        south_lat, east_lon = self._point_to_lat_lon(east_x, south_y, zoom)

        return BoundingBox(
            west=west_lon,
            south=south_lat,
            east=east_lon,
            north=north_lat,
            crs="EPSG:4326"
        )

    async def fetch_tile(
        self,
        center_lat: float,
        center_lon: float,
        zoom: int = 20,
        width: int = 640,
        height: int = 640,
        scale: int = 1,
        maptype: str = "satellite"
    ) -> Optional[Tile]:
        """
        Fetch a single image tile from Google Maps
        """
        if not self.api_key:
            raise ValueError("API Key is required")

        params = {
            "center": f"{center_lat},{center_lon}",
            "zoom": str(zoom),
            "size": f"{width}x{height}",
            "maptype": maptype,
            "key": self.api_key,
            "scale": str(scale)
        }

        # Scale implies higher res, effectively doubling pixels if scale=2
        # However, we treat the output image as the ground truth size

        url = f"{self.BASE_URL}?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
        session = await self._get_session()

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Google Maps API failed: {response.status} - {text}")
                    return None

                data = await response.read()
                image = Image.open(BytesIO(data)).convert('RGB')

                # If scale=2, the image returned is width*2 x height*2
                # We need to adjust our bounds calculation if we use the returned image dimensions
                img_array = np.array(image)
                actual_h, actual_w = img_array.shape[:2]

                # Calculate bounds based on the requested logical dimensions (width/height)
                # Google Maps zoom levels are based on logical pixels (256px tiles)
                # The bounds calculation relies on world coordinates which align with logical pixels
                bounds = self._calculate_image_bounds(center_lat, center_lon, zoom, width, height)

                return Tile(
                    x=int(center_lon * 1000), # Dummy grid coordinates
                    y=int(center_lat * 1000),
                    zoom=zoom,
                    image=img_array,
                    bounds_rd=None,
                    bounds_wgs84=bounds
                )

        except Exception as e:
            logger.error(f"Error fetching Google Maps image: {e}")
            return None

    async def fetch_tiles_for_area(
        self,
        center_lat: float,
        center_lon: float,
        radius_meters: float = 200,
        zoom: int = 21,
        max_tiles: int = 16
    ) -> List[Tile]:
        """
        Cover an area with Google Maps tiles.
        Simple strategy: Just fetch one large tile if radius is small,
        or a grid if larger (though static maps has strict limits).

        For simplicity in this version, we will fetch a grid of images
        spaced appropriately to cover the radius.
        """
        # Calculate coverage needed
        # At zoom 21, resolution is approx 0.075 meters/pixel
        resolution = 156543.03392 * math.cos(center_lat * math.pi / 180) / (2 ** zoom)

        # Max image size is 640x640
        tile_size_px = 640
        tile_size_meters = tile_size_px * resolution

        # Determine grid size (radius is from center to edge)
        # width needed = radius * 2
        total_width = radius_meters * 2

        tiles_axis = math.ceil(total_width / tile_size_meters)
        # Center the grid
        start_offset = -((tiles_axis - 1) * tile_size_meters) / 2

        tasks = []

        # Meters to degrees approx (for simple grid steps)
        # This is a Rough approximation for generating center points
        # The exact bounds are calculated per tile in fetch_tile
        lat_deg_per_meter = 1 / 111320
        lon_deg_per_meter = 1 / (111320 * math.cos(center_lat * math.pi / 180))

        for i in range(tiles_axis):
            for j in range(tiles_axis):
                offset_x = start_offset + j * tile_size_meters
                offset_y = start_offset + i * tile_size_meters # Y goes up (North)

                # Invert Y for loop to go Top->Bottom or Bottom->Top?
                # Usually map tiles go Top->Bottom (North->South)
                # Let's align with North-South axis: positive Y is North
                # Loop i=0 is 'start_offset' (South-most if we treat it as Cartesian)
                # Let's just step in meters from center

                # Let's do a centered grid relative to the input lat/lon
                # Grid indices from -N to +N

                # Simplified:
                # Calculate meters offset from center
                dx = offset_x
                dy = offset_y

                # New center
                tile_center_lat = center_lat + dy * lat_deg_per_meter
                tile_center_lon = center_lon + dx * lon_deg_per_meter

                tasks.append(self.fetch_tile(
                    tile_center_lat,
                    tile_center_lon,
                    zoom=zoom,
                    width=tile_size_px,
                    height=tile_size_px
                ))

        if len(tasks) > max_tiles:
            logger.warning(f"Requested area requires {len(tasks)} tiles, limiting to {max_tiles}")
            tasks = tasks[:max_tiles]

        results = await asyncio.gather(*tasks)
        return [t for t in results if t is not None]
