# System Architecture

This document details the technical architecture of the Dutch Parking Space Detection System. The pipeline is designed to be modular, robust, and geographically accurate.

## 🏗️ Pipeline Overview

The system operates as a linear pipeline with the following stages:

1.  **Geocoding & Input:** Convert municipality name to a bounding box (RD coordinates).
2.  **Imagery Fetching:** Download high-resolution aerial imagery tiles from PDOK.
3.  **AI Detection:** Analyze each tile using Google Gemini 3 Flash.
4.  **Geo-referencing:** Convert pixel detections to real-world coordinates.
5.  **Post-Processing:** Deduplicate overlapping detections and calculate statistics.
6.  **Export:** Save results as GeoJSON.

```mermaid
graph TD
    A[Input: Municipality] -->|Geocoding| B[Bounding Box (RD)]
    B -->|Tiling Strategy| C[Image Tiles]
    C -->|Fetch WMS| D[PDOK Aerial Imagery]
    D -->|Gemini 3 Flash| E[Detection Results (Pixels)]
    E -->|Coordinate Transform| F[Geo-referenced Spots (WGS84)]
    F -->|Spatial Dedup| G[Unique Parking Spots]
    G -->|Export| H[GeoJSON Output]
```

## 🌍 Coordinate Systems

The Netherlands uses a specific coordinate system called **Rijksdriehoek (RD)** or **Amersfoort / RD New** (EPSG:28992).

-   **Input/Processing:** All internal spatial operations (fetching imagery, calculating areas, tiling) are done in **RD coordinates (Meters)**. This ensures accurate distance and area calculations.
-   **Output:** The final output is converted to **WGS84 (EPSG:4326)** (Latitude/Longitude) for compatibility with standard mapping tools like Google Maps and GeoJSON viewers.

**Coordinate Transformer (`src.imagery_fetcher.CoordinateTransformer`):**
Uses `pyproj` to perform high-precision transformations between these systems.

## 🧠 AI Detection Strategy (Gemini 3 Flash)

The system uses a **Two-Step Prompting Strategy** to maximize accuracy using the Gemini 3 Flash vision model.

### Step 1: Structural Detection
**Goal:** Find *every* parking space, regardless of occupancy.
**Prompt:** Focuses on lines, painted markings, curbs, and surface patterns. It asks the model to identify the "grid" of the parking lot.
**Result:** A list of all potential parking spots.

### Step 2: Occupancy Detection
**Goal:** Find *only* empty parking spaces.
**Prompt:** Focuses on the visibility of pavement. "A space is empty if pavement is visible and no vehicle is present."
**Result:** A list of empty spots.

### Step 3: Merging (The Logic)
The system performs an **Intersection over Union (IoU)** check between the results of Step 1 and Step 2.
-   If a "structural spot" (Step 1) significantly overlaps with an "empty spot" (Step 2), it is marked as **Empty**.
-   If a "structural spot" has NO overlap with any "empty spot", it is inferred to be **Occupied**.

This "inference by exclusion" is often more reliable than asking a model to explicitly detect diverse vehicle types (cars, trucks, vans).

## 🧩 Spatial Deduplication

To handle large areas, the municipality is split into overlapping tiles (50m x 50m with 15% overlap). This creates a problem: a parking spot on the edge of a tile might be detected twice (once in Tile A, once in Tile B).

**Solution: `SpatialDeduplicator`**
1.  All detected spots are converted to global WGS84 coordinates.
2.  Spots are sorted by confidence.
3.  The system iterates through the list. For each spot, it removes any other spots within a small threshold distance (e.g., 1.0 meter) from its center.
4.  This ensures that the highest-confidence detection is kept, and duplicates are discarded.

## 💾 Data Flow & Storage

-   **Imagery:** Fetched in-memory (or cached to disk in debug modes).
-   **GeoJSON:** The primary output format.
    -   `geometry`: Polygon of the parking spot.
    -   `properties`: Metadata including `confidence`, `is_occupied`, `area_sqm`, `detected_at`.
