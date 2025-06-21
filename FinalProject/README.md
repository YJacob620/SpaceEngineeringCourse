# Bright Star Identifier

A Python application that detects and identifies stars in night sky images by matching them to stars in the [Yale Bright Star Catalog](http://tdc-www.harvard.edu/catalogs/bsc5.html). This tool can analyze single images, process batches of images, or compare stars between multiple images.

## Features

- **Star Detection**: Automatically detects bright stars in astronomical images
- **Star Identification**: Matches detected stars to the Yale Bright Star Catalog (BSC5)
- **Multiple Modes**:
  - Single image analysis with detailed star information
  - Batch processing of image directories
  - Cross-image matching to find common stars between two photos
- **Visual Output**: Generates annotated images showing identified stars with catalog information
- **Composite Creation**: Creates aligned composite images from matching star patterns

## Requirements

- Python 3.7+
- Packages (install with _pip install -r requirements.txt)_: matplotlib, pillow, numpy, opencv-python, scipy

## Installation

1. Clone or download this repository
2. Install dependencies with:

_pip install -r requirements.txt_

## Usage

Open terminal/CMD in the program folder and run:

_python BrightStarIdentifier.py_

The program will prompt you to select a mode and guide you through file selection.

#### Command Line Options

(Re)Generate the star database:

_python BrightStarIdentifier.py -g_

Enable additional visualizations:

_python BrightStarIdentifier.py -v_

## Operating Modes

**Mode 1: Single Image Matching**

- Analyzes one image to identify stars
- Shows star positions, catalog IDs, and magnitudes
- Displays solution information (RA/Dec, field of view, etc.)

**Mode 2: Batch Image Matching**

- Processes all images in a selected directory
- Saves annotated results as images to a subdirectory
- Displays summary

**Mode 3: Cross-Image Matching**

- Compares two images to find common stars
- Creates side-by-side comparison visualization
- Can generate aligned composite images

**Technical Details**

- Uses a [modified version of the tetra3](https://github.com/smroid/cedar-solve/tree/master/tetra3) plate-solving algorithm
- Supports common image formats: PNG, JPG, TIFF, BMP, FITS
- An image with at least 4-5 visible bright stars is recommended for successful matching
- Works best with images having field of views (FOV) between 1-80 degrees
- Star detection optimized for stars with magnitude <=7

## Troubleshooting

- **"Failed to load database"**: Run with -g flag to regenerate the database
- **"No stars detected"**: Try images with brighter stars or adjust camera settings
- **"No matching found"**: Ensure sufficient bright stars are visible and properly focused