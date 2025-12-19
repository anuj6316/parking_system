"""
PDOK Imagery Fetcher
Handles fetching aerial imagery from Dutch government PDOK services
"""
import asyncio
import aiohttp
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass
from pyproj import Transformer
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box in specified coordinate system"""
    west: float
    south: float
    east: float
    north: float
    crs: str = "EPSG:28992"  # Default to Dutch RD
    
    @property
    def width(self) -> float:
        return self.east - self.west
    
    @property
    def height(self) -> float:
        return self.north - self.south
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)
    
    def expand(self, factor: float) -> 'BoundingBox':
        """Expand bounding box by a factor (for overlap)"""
        dx = self.width * (factor - 1) / 2
        dy = self.height * (factor - 1) / 2
        return BoundingBox(
            west=self.west - dx,
            south=self.south - dy,
            east=self.east + dx,
            north=self.north + dy,
            crs=self.crs
        )


@dataclass
class Tile:
    """Represents a single image tile"""
    x: int
    y: int
    zoom: int
    image: Optional[np.ndarray] = None
    bounds_rd: Optional[BoundingBox] = None
    bounds_wgs84: Optional[BoundingBox] = None


class CoordinateTransformer:
    """Handles coordinate transformations between different CRS"""
    
    def __init__(self):
        # Dutch RD (EPSG:28992) to WGS84 (EPSG:4326)
        self.rd_to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        self.wgs84_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        
        # Web Mercator transformations
        self.wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    def rd_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Dutch RD coordinates to WGS84 lat/lon"""
        lon, lat = self.rd_to_wgs84.transform(x, y)
        return lat, lon
    
    def latlon_to_rd(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS84 lat/lon to Dutch RD coordinates"""
        return self.wgs84_to_rd.transform(lon, lat)
    
    def bbox_rd_to_wgs84(self, bbox: BoundingBox) -> BoundingBox:
        """Convert bounding box from RD to WGS84"""
        sw_lon, sw_lat = self.rd_to_wgs84.transform(bbox.west, bbox.south)
        ne_lon, ne_lat = self.rd_to_wgs84.transform(bbox.east, bbox.north)
        return BoundingBox(
            west=sw_lon, south=sw_lat,
            east=ne_lon, north=ne_lat,
            crs="EPSG:4326"
        )


class PDOKImageryFetcher:
    """Fetches aerial imagery from PDOK WMS/WMTS services"""
    
    WMS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    WMTS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0"
    
    def __init__(self, use_high_res: bool = True):
        """
        Initialize the imagery fetcher
        
        Args:
            use_high_res: If True, use 8cm resolution (winter imagery)
                         If False, use 25cm resolution (summer imagery)
        """
        self.layer = "Actueel_orthoHR" if use_high_res else "Actueel_ortho25"
        self.resolution = 0.08 if use_high_res else 0.25
        self.transformer = CoordinateTransformer()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_wms_url(
        self,
        bbox: BoundingBox,
        width: int = 512,
        height: int = 512
    ) -> str:
        """Build WMS GetMap URL"""
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": self.layer,
            "STYLES": "",
            "CRS": bbox.crs,
            "BBOX": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "FORMAT": "image/jpeg"
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.WMS_URL}?{query}"
    
    async def fetch_image_wms(
        self,
        bbox: BoundingBox,
        width: int = 512,
        height: int = 512
    ) -> np.ndarray:
        """
        Fetch a single image from WMS service
        
        Args:
            bbox: Bounding box in RD coordinates
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            numpy array of shape (height, width, 3)
        """
        url = self._build_wms_url(bbox, width, height)
        session = await self._get_session()
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"WMS request failed: {response.status}")
                
                data = await response.read()
                image = Image.open(BytesIO(data))
                return np.array(image)
                
        except Exception as e:
            logger.error(f"Error fetching image: {e}")
            raise
    
    def calculate_tiles_for_bbox(
        self,
        bbox: BoundingBox,
        tile_size_meters: float = 40.0,
        overlap: float = 0.15
    ) -> Generator[Tuple[BoundingBox, int, int], None, None]:
        """
        Calculate tile grid for a bounding box
        
        Args:
            bbox: Overall bounding box to tile
            tile_size_meters: Size of each tile in meters
            overlap: Overlap between tiles (0.0 to 0.5)
            
        Yields:
            Tuples of (tile_bbox, tile_x, tile_y)
        """
        effective_size = tile_size_meters * (1 - overlap)
        
        num_x = math.ceil(bbox.width / effective_size)
        num_y = math.ceil(bbox.height / effective_size)
        
        for y in range(num_y):
            for x in range(num_x):
                tile_west = bbox.west + x * effective_size
                tile_south = bbox.south + y * effective_size
                tile_east = tile_west + tile_size_meters
                tile_north = tile_south + tile_size_meters
                
                # Clip to original bbox
                tile_east = min(tile_east, bbox.east + tile_size_meters * overlap)
                tile_north = min(tile_north, bbox.north + tile_size_meters * overlap)
                
                tile_bbox = BoundingBox(
                    west=tile_west,
                    south=tile_south,
                    east=tile_east,
                    north=tile_north,
                    crs=bbox.crs
                )
                
                yield tile_bbox, x, y
    
    async def fetch_tiles_for_area(
        self,
        bbox: BoundingBox,
        tile_size_pixels: int = 1024,
        tile_size_meters: float = 20.0,
        max_concurrent: int = 8
    ) -> List[Tile]:
        """
        Fetch all tiles for a given area
        
        Args:
            bbox: Bounding box of area to process
            tile_size_pixels: Size of each tile in pixels
            tile_size_meters: Size of each tile in meters
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of Tile objects with imagery
        """
        tiles = []
        tile_configs = list(self.calculate_tiles_for_bbox(bbox, tile_size_meters))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_tile(tile_bbox: BoundingBox, x: int, y: int) -> Tile:
            async with semaphore:
                try:
                    image = await self.fetch_image_wms(
                        tile_bbox,
                        tile_size_pixels,
                        tile_size_pixels
                    )
                    return Tile(
                        x=x,
                        y=y,
                        zoom=0,
                        image=image,
                        bounds_rd=tile_bbox,
                        bounds_wgs84=self.transformer.bbox_rd_to_wgs84(tile_bbox)
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch tile ({x}, {y}): {e}")
                    return Tile(x=x, y=y, zoom=0, image=None, bounds_rd=tile_bbox)
        
        tasks = [
            fetch_tile(tile_bbox, x, y)
            for tile_bbox, x, y in tile_configs
        ]
        
        results = await asyncio.gather(*tasks)
        tiles = [t for t in results if t.image is not None]
        
        logger.info(f"Fetched {len(tiles)}/{len(tile_configs)} tiles successfully")
        return tiles
    
    ## ANUJ
    # ---
    def save_image_to_file(
        self,
        image: np.ndarray,
        filepath: str,
        format: str = "JPEG"
    ) -> str:
        """
        Save a numpy array image to a file
        
        Args:
            image: numpy array of shape (height, width, 3)
            filepath: Path to save the image (with or without extension)
            format: Image format ('JPEG', 'PNG', 'TIFF')
            
        Returns:
            Absolute path to the saved file
        """
        from pathlib import Path as PathLib
        
        filepath = PathLib(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add extension if not present
        ext_map = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff"}
        if not filepath.suffix:
            filepath = filepath.with_suffix(ext_map.get(format, ".jpg"))
        
        img = Image.fromarray(image)
        img.save(str(filepath), format=format)
        
        logger.info(f"Saved image to {filepath}")
        return str(filepath.absolute())
    
    async def save_tiles_to_directory(
        self,
        tiles: List[Tile],
        output_dir: str,
        prefix: str = "tile",
        format: str = "JPEG",
        save_metadata: bool = True
    ) -> List[str]:
        """
        Save a list of tiles to a directory
        
        Args:
            tiles: List of Tile objects with images
            output_dir: Directory to save images
            prefix: Filename prefix for tiles
            format: Image format ('JPEG', 'PNG', 'TIFF')
            save_metadata: If True, save a JSON file with tile metadata
            
        Returns:
            List of saved file paths
        """
        import json
        from pathlib import Path as PathLib
        
        output_path = PathLib(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        metadata = []
        
        for tile in tiles:
            if tile.image is None:
                continue
            
            filename = f"{prefix}_{tile.x}_{tile.y}"
            filepath = output_path / filename
            saved_path = self.save_image_to_file(tile.image, str(filepath), format)
            saved_paths.append(saved_path)
            
            # Collect metadata
            if save_metadata and tile.bounds_rd:
                metadata.append({
                    "filename": PathLib(saved_path).name,
                    "x": tile.x,
                    "y": tile.y,
                    "bounds_rd": {
                        "west": tile.bounds_rd.west,
                        "south": tile.bounds_rd.south,
                        "east": tile.bounds_rd.east,
                        "north": tile.bounds_rd.north
                    },
                    "bounds_wgs84": {
                        "west": tile.bounds_wgs84.west,
                        "south": tile.bounds_wgs84.south,
                        "east": tile.bounds_wgs84.east,
                        "north": tile.bounds_wgs84.north
                    } if tile.bounds_wgs84 else None
                })
        
        # Save metadata JSON
        if save_metadata and metadata:
            metadata_path = output_path / f"{prefix}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info(f"Saved {len(saved_paths)} tiles to {output_dir}")
        return saved_paths
    
    async def download_area_images(
        self,
        bbox: BoundingBox,
        output_dir: str,
        tile_size_pixels: int = 512,
        tile_size_meters: float = 40.0,
        prefix: str = "tile",
        format: str = "JPEG",
        max_concurrent: int = 8
    ) -> List[str]:
        """
        Fetch tiles for an area and save them directly to a local directory
        
        Args:
            bbox: Bounding box of area to download
            output_dir: Directory to save images
            tile_size_pixels: Size of each tile in pixels
            tile_size_meters: Size of each tile in meters
            prefix: Filename prefix for tiles
            format: Image format ('JPEG', 'PNG', 'TIFF')
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of saved file paths
        """
        # Fetch tiles
        tiles = await self.fetch_tiles_for_area(
            bbox,
            tile_size_pixels=tile_size_pixels,
            tile_size_meters=tile_size_meters,
            max_concurrent=max_concurrent
        )
        
        # Save to directory
        saved_paths = await self.save_tiles_to_directory(
            tiles,
            output_dir=output_dir,
            prefix=prefix,
            format=format,
            save_metadata=True
        )
        
        return saved_paths
# ---

class MunicipalityGeocoder:
    """Geocodes Dutch municipality names to bounding boxes"""
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    
    def __init__(self):
        self.transformer = CoordinateTransformer()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "ParkingDetectionSystem/1.0"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_municipality_bbox(
        self,
        municipality_name: str,
        country: str = "Netherlands"
    ) -> BoundingBox:
        """
        Get bounding box for a municipality
        
        Args:
            municipality_name: Name of the municipality (e.g., "Amsterdam", "Amersfoort")
            country: Country name for disambiguation
            
        Returns:
            BoundingBox in Dutch RD coordinates
        """
        session = await self._get_session()
        
        params = {
            "q": f"{municipality_name}, {country}",
            "format": "json",
            "polygon_geojson": "0",
            "limit": "1"
        }
        
        async with session.get(self.NOMINATIM_URL, params=params) as response:
            if response.status != 200:
                raise Exception(f"Geocoding failed: {response.status}")
            
            data = await response.json()
            
            if not data:
                raise ValueError(f"Municipality not found: {municipality_name}")
            
            result = data[0]
            bbox_wgs84 = result["boundingbox"]
            
            # Convert from WGS84 to RD
            south, north = float(bbox_wgs84[0]), float(bbox_wgs84[1])
            west, east = float(bbox_wgs84[2]), float(bbox_wgs84[3])
            
            sw_rd = self.transformer.latlon_to_rd(south, west)
            ne_rd = self.transformer.latlon_to_rd(north, east)
            
            return BoundingBox(
                west=sw_rd[0],
                south=sw_rd[1],
                east=ne_rd[0],
                north=ne_rd[1],
                crs="EPSG:28992"
            )
    
    async def get_municipality_area_km2(self, municipality_name: str) -> float:
        """Get approximate area of municipality in square kilometers"""
        bbox = await self.get_municipality_bbox(municipality_name)
        # Approximate area (RD coordinates are in meters)
        return (bbox.width * bbox.height) / 1_000_000


# Utility functions
def meters_to_pixels(meters: float, resolution: float = 0.08) -> int:
    """Convert meters to pixels at given resolution"""
    return int(meters / resolution)


def pixels_to_meters(pixels: int, resolution: float = 0.08) -> float:
    """Convert pixels to meters at given resolution"""
    return pixels * resolution


async def demo():
    """Demo function showing usage - downloads images to output/images"""
    geocoder = MunicipalityGeocoder()
    fetcher = PDOKImageryFetcher(use_high_res=True)
    
    try:
        # Get bounding box for Amersfoort
        bbox = await geocoder.get_municipality_bbox("Amersfoort")
        print(f"Amersfoort bbox (RD): {bbox}")
        
        area = await geocoder.get_municipality_area_km2("Amersfoort")
        print(f"Approximate area: {area:.2f} km²")
        
        # Create a small test area (500m x 500m for faster demo)
        test_bbox = BoundingBox(
            west=bbox.center[0] - 250,
            south=bbox.center[1] - 250,
            east=bbox.center[0] + 250,
            north=bbox.center[1] + 250,
            crs="EPSG:28992"
        )
        
        print(f"Downloading images for test area: {test_bbox}")
        
        # Download and save images to local directory
        saved_files = await fetcher.download_area_images(
            bbox=test_bbox,
            output_dir="output/images",
            prefix="amersfoort",
            format="JPEG",
            tile_size_pixels=512,
            tile_size_meters=40.0,
            max_concurrent=4
        )
        
        print(f"Downloaded and saved {len(saved_files)} images to output/images/")
        for f in saved_files[:5]:  # Show first 5 files
            print(f"  - {f}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")
        
    finally:
        await geocoder.close()
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(demo())
