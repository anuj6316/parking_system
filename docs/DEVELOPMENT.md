# Development Guide

This guide is intended for developers who want to contribute to the project or understand the code structure for maintenance.

## 🛠️ Development Environment

### 1. Setup

We recommend using a virtual environment (venv) or Conda.

```bash
# Create venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
For development, create a `.env` file (if you use `python-dotenv`) or export the variables in your shell.
Required:
-   `GEMINI_API_KEY`: Your Google Gemini API key.

## 📂 Project Structure

```
.
├── src/
│   ├── main_pipeline.py      # Entry point and orchestration logic
│   ├── parking_detector.py   # Gemini API integration & detection logic
│   ├── imagery_fetcher.py    # PDOK WMS fetching & Coordinate transformation
│   ├── geo_converter.py      # Pixel -> Geo coordinates & GeoJSON export
│   └── visualizer.py         # Debug visualization (drawing boxes on images)
├── tests/
│   ├── test_parking_pipeline.py  # End-to-end pipeline tests (local images)
│   └── ...
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
└── README.md                 # Main documentation
```

## 🧪 Running Tests

The project includes tests that can run against local images to verify the pipeline without fetching new data from PDOK every time.

**Note:** The tests require the `GEMINI_API_KEY` to be set, as they make real calls to the Gemini API.

```bash
# Run the pipeline test script
python tests/test_parking_pipeline.py
```

*Note: You may need to adjust the `TEST_IMAGES_DIR` path in `tests/test_parking_pipeline.py` to point to your local test dataset.*

## 🤝 Contribution Guidelines

1.  **Code Style:** We use standard Python PEP 8 conventions.
2.  **Type Hinting:** All new functions should have Python type hints.
3.  **Docstrings:** Classes and functions must have docstrings explaining arguments and return values.
4.  **Verification:** Before submitting a PR, ensure you have run the tests and verified the GeoJSON output is valid.

## 🔍 Key Modules Explained

-   **`src.imagery_fetcher`**: This is where the PDOK integration lives. If the PDOK API changes (WMS/WMTS), this is the file to update. It handles the `aiohttp` sessions and `pyproj` transformations.
-   **`src.parking_detector`**: This contains the prompt engineering. If detection accuracy is low, iterate on the `PROMPT_ALL_SPOTS` and `PROMPT_EMPTY_SPOTS` constants here.
-   **`src.geo_converter`**: This handles the math for converting pixel polygons to GPS coordinates. It uses `rasterio`'s `from_bounds` logic manually to ensure precision.
