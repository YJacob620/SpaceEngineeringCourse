import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "./2019-07-29T204726_Alt60_Azi-135_Try1.tiff"
MAX_STAR_RADIUS = 20
BRIGHTNESS_THRESHOLD = 20
DEBUG = 1
DISPLAY_FILTERED = False
SAVE_CSV = False


def detect_stars(image_path, threshold=BRIGHTNESS_THRESHOLD, min_radius=1, max_radius=MAX_STAR_RADIUS):
    """
    Detect stars in a night sky image and return their coordinates, radius, and brightness.

    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    threshold : int, optional
        Brightness threshold to identify stars
    min_radius, max_radius : int, optional
        Minimum and maximum radius to consider for star detection

    Returns:
    --------
    list of tuples (x, y, r, b) where:
        x, y: coordinates of the star center
        r: approximate radius of the star in pixels
        b: brightness value (0-255)
    """

    # Read the image
    img = cv2.imread(str(image_path))

    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # This improves contrast locally, making dim stars more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 2. Background estimation and subtraction
    # Use a large kernel to estimate the background
    background = cv2.medianBlur(contrast_enhanced, 51)
    subtracted = cv2.subtract(contrast_enhanced, background)

    pixels = img.flatten()
    if threshold is None:
        threshold = np.average(pixels)
    if DEBUG > 0:
        print(f"Set brightness threshold to {threshold:.3f}")

    # Threshold the image to identify bright spots (stars)
    _, thresh = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if DISPLAY_FILTERED:
        plt.figure(figsize=(8, 8))
        plt.imshow(thresh)
        plt.show()

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars = []
    for contour in contours:
        # Get the bounding circle of the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Filter by radius
        if min_radius <= radius <= max_radius:
            # Center coordinates (as integers)
            center = (int(x), int(y))

            # Calculate brightness (average intensity in the original grayscale image)
            # Create a mask for the star
            mask = np.zeros_like(gray)
            cv2.circle(mask, center, int(radius), 255, -1)

            # Calculate average brightness within the circle
            masked_star = cv2.bitwise_and(gray, gray, mask=mask)
            brightness = np.mean(masked_star[masked_star > 0])

            # Add to the list of stars
            stars.append((center[0], center[1], radius, brightness))

    return stars


def visualize_detected_stars(image_path, stars, output_path=None):
    """
    Visualize the detected stars on the original image.

    Parameters:
    -----------
    image_path : str or Path
        Path to the original image
    stars : list
        List of (x, y, r, b) tuples representing detected stars
    output_path : str or Path, optional
        Path to save the visualization. If None, the image is displayed but not saved.
    """
    # Read the original image
    img = cv2.imread(str(image_path))

    # Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a figure for visualization
    plt.figure(figsize=(12, 7))
    plt.imshow(img_rgb)

    # Draw circles for each detected star using matplotlib
    for x, y, r, b in stars:
        # Draw circle around the star (green)
        circle = plt.Circle((x, y), r + 2, color='lime', fill=False, linewidth=1)
        plt.gca().add_patch(circle)

        # Draw center point (red)
        # plt.plot(x, y, 'ro', markersize=2)

    plt.title(f"Detected {len(stars)} stars")
    plt.axis('off')

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


def save_star_data(stars, output_file):
    """
    Save star data to a text file.

    Parameters:
    -----------
    stars : list
        List of (x, y, r, b) tuples representing detected stars
    output_file : str or Path
        Path to save the data
    """
    with open(output_file, 'w') as f:
        f.write("x,y,radius,brightness\n")
        for x, y, r, b in stars:
            f.write(f"{x},{y},{r},{b}\n")


def main():
    # Detect stars
    stars = detect_stars(IMAGE_PATH)
    print(f"Detected {len(stars)} stars")

    # Visualize the results
    visualize_detected_stars(IMAGE_PATH, stars, output_path="detected_stars.jpg")

    # Save the data
    if SAVE_CSV:
        save_star_data(stars, "./temp/star_data.csv")

    # Print some stats
    if stars:
        brightest_star = max(stars, key=lambda x: x[3])
        print(f"Brightest star: (x={brightest_star[0]}, y={brightest_star[1]}, radius={brightest_star[2]:.2f}, brightness={brightest_star[3]:.2f})")

        dimmest_star = min(stars, key=lambda x: x[3])
        print(f"Dimmest star: (x={dimmest_star[0]}, y={dimmest_star[1]}, radius={dimmest_star[2]:.2f}, brightness={dimmest_star[3]:.2f})")

        largest_star = max(stars, key=lambda x: x[2])
        print(f"Largest star: (x={largest_star[0]}, y={largest_star[1]}, radius={largest_star[2]:.2f}, brightness={largest_star[3]:.2f})")

        smallest_star = min(stars, key=lambda x: x[2])
        print(f"Smallest star: (x={smallest_star[0]}, y={smallest_star[1]}, radius={smallest_star[2]:.2f}, brightness={smallest_star[3]:.2f})")


if __name__ == "__main__":
    main()
