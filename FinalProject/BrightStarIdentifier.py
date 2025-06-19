import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import numpy as np
import os
import time
import logging
import tetra3
from typing import Dict, List, Tuple, Optional
import cv2


def get_solution_for_image(image, return_visual: bool):
    solution = T3.solve_from_image(
        image=image,
        return_matches=True,
        return_visual=return_visual,
        return_images=return_visual,
        # fov_max_error=0.2,  # ±20% FOV tolerance
        # match_radius=0.01,  # Star matching tolerance (degrees)
        # match_threshold=0.001,  # Statistical confidence threshold
        # solve_timeout=7000,  # Maximum solving time (miliseconds)
        # fov_estimate=70,
    )
    return solution


def label_star_on_plot(ax: matplotlib.axes.Axes, x, y, color, cat_id, magnitude, fontsize=8, arrow_linewidth=1,
                       radius=15, offset_x=20, offset_y=-20):
    """
    Add a circle and annotation for a single star.

    Parameters:
    - ax: matplotlib axes object
    - x, y: star coordinates
    - color: color for circle and annotation
    - cat_id: catalog ID for the star
    - magnitude: star magnitude
    - fontsize: font size for annotation (default: 8)
    - arrow_linewidth: linewidth for annotation arrow (default: 1.2)
    - radius: circle radius (default: 15)
    - offset_x, offset_y: annotation text offset (default: 20, -20)
    """
    # Draw circle around star
    circle = patches.Circle((x, y), radius=radius, fill=False,
                            edgecolor=color, linewidth=2)
    ax.add_patch(circle)

    # Add annotation
    ax.annotate(
        f'BSC {cat_id}\n(Mag: {magnitude:.1f})',
        xy=(x, y),
        xytext=(x + offset_x, y + offset_y),
        fontsize=fontsize, color=color, weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                  alpha=0.8, edgecolor=color),
        arrowprops=dict(arrowstyle='->', color=color, lw=arrow_linewidth)
    )


def visualize_solution(image, solution):
    """
    Visualize the original image with detailed star annotations based on tetra3's solution.

    Args:
        image: PIL Image object - the original image
        solution: dict - the solution dictionary from tetra3
    """
    if solution['RA'] is None:
        print("No solution found - unable to create visualization")
        return

    # Single-plot figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.array(image), cmap='gray')
    ax.set_title('Matching Visualization', fontsize=14, fontweight='bold')

    matched_centroids = solution['matched_centroids']
    matched_catIDs = solution['matched_catID']
    matched_stars = solution['matched_stars']

    # noinspection PyUnresolvedReferences
    colors = plt.cm.Set3(np.linspace(0, 1, len(matched_centroids)))

    for i, (centroid, cat_id, star_data) in enumerate(zip(matched_centroids, matched_catIDs, matched_stars)):
        y, x = centroid
        color = colors[i]
        label_star_on_plot(ax, x, y, color, cat_id, star_data[2], fontsize=7, arrow_linewidth=1)

    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)  # Flip y-axis for image coordinates
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Pixel', fontsize=10)
    ax.set_ylabel('Y Pixel', fontsize=10)

    plt.tight_layout()
    return fig


def print_matched_stars_info(solution):
    print("\nMatched Stars Information:")
    matched_stars = solution.get('matched_stars')
    matched_centroids = solution.get('matched_centroids')
    matched_catID = solution.get('matched_catID')
    for i, (star, centroid, cat_id) in enumerate(zip(matched_stars, matched_centroids, matched_catID), 1):
        ra, dec, mag = star
        y, x = centroid
        print(f"  {i}:", end=' ')
        print(f"[Pixel Position (y,x): ({y:.1f}, {x:.1f})]", end=', ')
        print(f"[Catalog ID: {cat_id}]", end=', ')
        print(f"[RA/Dec: {ra}°/{dec}°]", end=', ')
        print(f"[Magnitude: {mag:.2f}]")


def match_stars_in_image(image_path: str, show_star_detection: bool = False) -> None:
    """
    Given an image of the night sky, try to match stars in it to those that are in
    a Tetra3 database.
    :param image_path: Path to the image
    :param show_star_detection: Whether to show a separate window with star detection results visualization.
    """
    image = Image.open(image_path)
    rgb_image = image.convert("RGB")
    solution = get_solution_for_image(rgb_image, True)
    matched = solution[0]['RA'] is not None
    detected_stars = 'final_centroids' in solution[1]

    if show_star_detection and detected_stars:
        img = solution[0]['visual'] if matched else solution[1]['final_centroids']
        plt.imshow(img)
        plt.title("Bright Stars Detection Result")
        plt.axis('off')
    if not matched:
        if not detected_stars:
            print('NO STARS WERE DETECTED')
        else:
            print('NO MATCHING WAS FOUND')
    else:
        print('\nSolution found:')
        for key, value in solution[0].items():
            if isinstance(value, (int, float, str)) and key in SOLUTION_KEY_DESCRIPTIONS:
                print(f"\t{key} ({SOLUTION_KEY_DESCRIPTIONS[key]}): {value}")
        print_matched_stars_info(solution[0])
        visualize_solution(rgb_image, solution[0])
    plt.show()


def match_stars_in_images_in(input_dir: str, output_subdir_name: str = "Images_Matched", ):
    """
    Iterate over all image files in input_dir. For each:
      - open image
      - detect star centroids
      - attempt matching
      - if matched, save the figure from visualize_solution() into output_dir
    At the end, print a summary: how many images were matched, unmatched, skipped, errors, etc.
    """

    # Prepare output directory
    output_dir = os.path.join(input_dir, output_subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Matched images will be saved into: {input_dir}/{output_subdir_name}")

    # File extensions to consider as images
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.fits', 'jfif')

    # Summary counters and timer
    total_files = 0
    matched_count = 0
    unmatched_count = 0
    skipped_count = 0  # no stars detected
    error_count = 0
    start_all = time.time()

    # Iterate over images
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            continue
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue

        total_files += 1
        print(f"\nProcessing '{fname}' ({total_files})...")
        t_start = time.time()

        try:
            # Open image
            try:
                image = Image.open(path)
            except Exception as e:
                print(f"  [ERROR] Could not open image: {e}")
                error_count += 1
                continue

            # Solve / match
            rgb_image = image.convert("RGB")
            solution = get_solution_for_image(rgb_image, False)
            if solution.get('RA', None) is None:
                print("  No matching was found for this image.")
                unmatched_count += 1
                continue

            matched_count += 1
            elapsed = time.time() - t_start
            print(f"  Solution found (in {elapsed:.2f} sec):")
            for key, value in solution.items():
                if isinstance(value, (int, float, str)) and key in SOLUTION_KEY_DESCRIPTIONS:
                    print(f"    {key} ({SOLUTION_KEY_DESCRIPTIONS[key]}): {value}")
            print_matched_stars_info(solution)

            # Plot + save
            visualize_solution(rgb_image, solution)
            fig = plt.gcf()
            base, _ext = os.path.splitext(fname)
            out_fname = f"{base}_matched.png"
            out_path = os.path.join(output_dir, out_fname)
            try:
                fig.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"  Saved matched plot to: {input_dir}/{out_fname}")
            except Exception as e:
                print(f"  [ERROR] Failed to save figure: {e}")
                error_count += 1
            finally:
                plt.close(fig)

        except Exception as e:
            print(f"  [ERROR] Exception during processing: {e}")
            error_count += 1

    total_elapsed = time.time() - start_all

    # Summary
    print("\nBatch processing complete.")
    print("Summary:")
    print(f"  Total image files found with matching extensions: {total_files}")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched (processed but no match): {unmatched_count}")
    print(f"  Skipped (no stars detected): {skipped_count}")
    print(f"  Errors during processing: {error_count}")
    if total_files > 0:
        match_rate = matched_count / total_files * 100
        print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Total elapsed time: {total_elapsed:.2f} seconds")
    if total_files > 0:
        avg_time = total_elapsed / total_files
        print(f"  Average time per image: {avg_time:.2f} seconds")


def match_stars_between_images(image_path1: str, image_path2: str,
                               create_composite: bool = True, show_comparison: bool = True):
    """Complete pipeline to match stars between two images."""

    def find_common_stars(solution1: Dict, solution2: Dict) -> Tuple[List, List, List]:
        """
        Identify stars common to two image solutions by matching star catalog IDs.

        Parameters
        ----------
        solution1 : Dict
            A dictionary representing the first image’s solution. Expected keys:
            - 'RA': Right Ascension data (used here to check if a valid solution exists). If None, the function returns empty lists.
            - 'matched_catID': List of catalog IDs matched in solution1. If missing or empty, treated as no matches.
        solution2 : Dict
            A dictionary for the second image’s solution with the same expected structure:
            - 'RA': Right Ascension; if None, return empty lists.
            - 'matched_catID': List of catalog IDs matched in solution2.

        Returns
        -------
        common_ids : List
            A list of catalog IDs that appear in both solution1['matched_catID'] and solution2['matched_catID'].
        indices_1 : List[int]
            For each entry in `common_ids`, the index in solution1['matched_catID'] where that ID was found.
        indices_2 : List[int]
            For each entry in `common_ids`, the index in solution2['matched_catID'] where that ID was found.
        """

        if solution1['RA'] is None or solution2['RA'] is None:
            return [], [], []

        cat_ids_1 = solution1.get('matched_catID', [])
        cat_ids_2 = solution2.get('matched_catID', [])

        if not cat_ids_1 or not cat_ids_2:
            return [], [], []

        # Find common catalog IDs
        common_ids = []
        indices_1 = []
        indices_2 = []

        for i, cat_id_1 in enumerate(cat_ids_1):
            for j, cat_id_2 in enumerate(cat_ids_2):
                if cat_id_1 == cat_id_2:
                    common_ids.append(cat_id_1)
                    indices_1.append(i)
                    indices_2.append(j)
                    break

        return common_ids, indices_1, indices_2

    def compute_transformation_matrix(points1: np.ndarray, points2: np.ndarray) -> Optional[np.ndarray]:
        """Compute transformation matrix from points1 to points2."""
        if len(points1) < 3:
            return None

        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)

        if len(points1) >= 4:
            # Use perspective transformation for 4+ points
            matrix = cv2.getPerspectiveTransform(points1[:4], points2[:4])
        else:
            # Use affine transformation for 3 points
            matrix = cv2.getAffineTransform(points1, points2)
            # Convert to 3x3 matrix
            matrix = np.vstack([matrix, [0, 0, 1]])

        return matrix

    # noinspection PyTypeChecker
    def create_composite_figure(image1: Image.Image, image2: Image.Image,
                                transformation_matrix: np.ndarray,
                                solution1: dict,
                                common_stars_tuple,
                                blend_alpha: float = 0.5):
        """
        Transform image2 and put it on top of image1 using the common stars found in them,
        but expand the canvas so that neither image is cropped, filling new regions with white.
        Also labels the common stars (from image1) on the composite.

        Args:
            image1: PIL Image for the base.
            image2: PIL Image to be transformed and overlaid.
            transformation_matrix: 2x3 affine or 3x3 affine/perspective matrix mapping image2 → image1 coordinates.
            solution1: dict containing at least:
                - 'matched_centroids': list of (y, x) tuples in image1
                - 'matched_stars': list of star info, e.g., (..., magnitude, ...)
            common_stars_tuple: (common_ids_found, indices1, indices2)
            blend_alpha: blending weight for image2 where it overlaps image1.
        Returns:
            Matplotlib Figure containing the composite with labels.
        """
        # Convert PIL images to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2)

        # Get dimensions
        h1, w1 = img1_array.shape[:2]
        h2, w2 = img2_array.shape[:2]

        # Determine if perspective or affine
        is_perspective = False
        if transformation_matrix.shape == (3, 3):
            # If last row not [0,0,1], treat as full perspective
            if not np.allclose(transformation_matrix[2, :], [0, 0, 1]):
                is_perspective = True
            else:
                # pure affine in 3x3 form
                is_perspective = False
        else:
            # 2x3 affine
            is_perspective = False

        # Compute the transformed corners of image2 in image1 coordinate space
        # Corners of image2 in homogeneous coords:
        corners2 = np.array([
            [0, 0, 1],
            [w2, 0, 1],
            [w2, h2, 1],
            [0, h2, 1],
        ]).T  # shape (3, 4)

        if is_perspective:
            # 3x3 perspective
            pts2_trans = transformation_matrix @ corners2  # shape (3,4)
            pts2_trans = pts2_trans / pts2_trans[2:3, :]  # normalize by w
            xs2 = pts2_trans[0, :]
            ys2 = pts2_trans[1, :]
        else:
            # Affine: ensure we have 2x3 matrix
            if transformation_matrix.shape == (3, 3):
                M_affine = transformation_matrix[:2, :]
            else:
                M_affine = transformation_matrix  # assumed shape (2,3)
            # For each corner: [x,y] → M_affine[:, :2] @ [x,y] + M_affine[:,2]
            pts2 = corners2[:2, :]  # shape (2,4)
            xs2 = M_affine[0, 0] * pts2[0, :] + M_affine[0, 1] * pts2[1, :] + M_affine[0, 2]
            ys2 = M_affine[1, 0] * pts2[0, :] + M_affine[1, 1] * pts2[1, :] + M_affine[1, 2]

        # Corners of image1 in image1 coords:
        corners1_x = np.array([0, w1, w1, 0])
        corners1_y = np.array([0, 0, h1, h1])

        # Combined extents
        all_x = np.concatenate([corners1_x, xs2])
        all_y = np.concatenate([corners1_y, ys2])

        min_x = np.min(all_x)
        min_y = np.min(all_y)
        max_x = np.max(all_x)
        max_y = np.max(all_y)

        # Compute integer canvas size and offsets
        # We want integer pixel indices.
        # If min_x < 0, offset_x = -min_x; else offset_x = 0.
        offset_x = int(np.ceil(-min_x)) if min_x < 0 else 0
        offset_y = int(np.ceil(-min_y)) if min_y < 0 else 0

        canvas_w = int(np.ceil(max_x - min_x))
        canvas_h = int(np.ceil(max_y - min_y))

        # Determine if color or grayscale
        is_color = (img1_array.ndim == 3 and img1_array.shape[2] == 3) or (img2_array.ndim == 3 and img2_array.shape[2] == 3)
        # Create white canvas
        if is_color:
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        else:
            canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

        # Place image1 onto canvas at (offset_x, offset_y)
        # Note: ensure slicing indices are int
        y1_start = offset_y
        x1_start = offset_x
        y1_end = y1_start + h1
        x1_end = x1_start + w1
        canvas[y1_start:y1_end, x1_start:x1_end] = img1_array

        # Prepare mask for image1 presence:  where image1 is placed, mask1 = True
        mask1 = np.zeros((canvas_h, canvas_w), dtype=bool)
        mask1[y1_start:y1_end, x1_start:x1_end] = True

        # Prepare adjusted transformation for image2: add offset
        if is_perspective:
            M_new = transformation_matrix.copy().astype(np.float64)
            # Add translation: x' = M * [x,y,1] + [offset_x, offset_y] in homogeneous: we adjust the [0,2] and [1,2] entries
            M_new[0, 2] += offset_x
            M_new[1, 2] += offset_y
            # M_new[2,2] remains 1
            # Warp onto full canvas
            warped2 = cv2.warpPerspective(
                img2_array,
                M_new,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255  # white background
            )
            # Create mask by warping a mask image
            mask_src = np.ones((h2, w2), dtype=np.uint8) * 255
            warped_mask2 = cv2.warpPerspective(
                mask_src,
                M_new,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            mask2 = warped_mask2 > 0
        else:
            # Affine: get 2x3
            if transformation_matrix.shape == (3, 3):
                M_affine = transformation_matrix[:2, :].copy().astype(np.float64)
            else:
                M_affine = transformation_matrix.copy().astype(np.float64)
            # adjust translation terms
            M_affine[0, 2] += offset_x
            M_affine[1, 2] += offset_y
            # Warp onto full canvas
            warped2 = cv2.warpAffine(
                img2_array,
                M_affine,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255  # white
            )
            mask_src = np.ones((h2, w2), dtype=np.uint8) * 255
            warped_mask2 = cv2.warpAffine(
                mask_src,
                M_affine,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            mask2 = warped_mask2 > 0

        # Now blend: canvas currently has image1 (or white) in canvas.
        # For pixels where mask2 is True:
        #   if mask1 is True (overlap): blend image1 and warped2
        #   else: take warped2 directly
        canvas_float = canvas.astype(np.float32)
        warped2_float = warped2.astype(np.float32)

        # For color, we assume shape (...,3); for grayscale, shape (...)
        if is_color:
            H, W, _ = canvas.shape
            for c in range(3):
                # Overlap region
                overlap = mask2 & mask1
                canvas_float[overlap, c] = (
                        (1 - blend_alpha) * canvas_float[overlap, c] +
                        blend_alpha * warped2_float[overlap, c]
                )
                # Non-overlap but where image2 exists
                only2 = mask2 & (~mask1)
                canvas_float[only2, c] = warped2_float[only2, c]
        else:
            # grayscale
            overlap = mask2 & mask1
            canvas_float[overlap] = (
                    (1 - blend_alpha) * canvas_float[overlap] +
                    blend_alpha * warped2_float[overlap]
            )
            only2 = mask2 & (~mask1)
            canvas_float[only2] = warped2_float[only2]

        composite_uint8 = np.clip(canvas_float, 0, 255).astype(np.uint8)

        # Prepare figure for visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(composite_uint8, cmap=None if is_color else 'gray')
        ax.set_title(f'Composite Image - {len(common_stars_tuple[0])} Common Stars Labeled',
                     fontsize=14, fontweight='bold')
        ax.axis('off')

        # Label common stars: shift their coordinates by the offset
        common_ids_found, indices1, indices2 = common_stars_tuple
        centroids1 = solution1['matched_centroids']  # list of (y, x)
        stars1 = solution1['matched_stars']  # e.g., list where [2] is magnitude

        # noinspection PyUnresolvedReferences
        colors = plt.cm.Set3(np.linspace(0, 1, len(common_ids_found)))

        for i, (cat_id, idx1, idx2) in enumerate(zip(common_ids_found, indices1, indices2)):
            color = colors[i]
            y1, x1 = centroids1[idx1]
            # Shift by offsets to composite coords
            x_plot = x1 + offset_x
            y_plot = y1 + offset_y
            magnitude = stars1[idx1][2]
            # Use your existing labeling function; signature: label_star_on_plot(ax, x, y, color, cat_id, magnitude)
            label_star_on_plot(ax, x_plot, y_plot, color, cat_id, magnitude)

        plt.tight_layout(pad=0)
        return fig

    def visualize_cross_image_matches(
            image1: Image.Image,
            image2: Image.Image,
            solution1: Dict,
            solution2: Dict,
            common_ids: List,
            indices1: List,
            indices2: List,
            gap: int = 50,
    ) -> plt.Figure:
        """
            Create a combined visualization showing star matches between two images, with a blank gap in between.
            - Removes axes.
            - Combines image1 (left) and image2 (right) into one canvas, separated by `gap` pixels.
            - Draws circles and labels on matched centroids in each image.
            - Draws lines connecting each matched pair across the gap.

            Args:
                image1: PIL.Image instance.
                image2: PIL.Image instance.
                solution1: dict for image2 with keys 'matched_centroids' (list of (y, x)) and 'matched_stars' (with e.g. magnitude at index 2).
                solution2: same as solution1 but for image2
                common_ids: list of IDs for each match.
                indices1: indice into solution1['matched_centroids'] for each common_id.
                indices2: indice into solution2['matched_centroids'] for each common_id.
                gap: number of pixels of blank space between the two images (default 50).
            Returns:
                Matplotlib Figure object.
            """
        # Convert PIL to numpy arrays
        arr1 = np.array(image1)
        arr2 = np.array(image2)

        # Validate dims
        if arr1.ndim not in (2, 3) or arr2.ndim not in (2, 3):
            raise ValueError("Images must be 2D (grayscale) or 3D (color).")

        # If one grayscale and one color, convert grayscale to 3-channel
        def ensure_3ch(arr):
            if arr.ndim == 2:
                return np.stack([arr] * 3, axis=-1)
            return arr

        if arr1.ndim != arr2.ndim:
            arr1 = ensure_3ch(arr1)
            arr2 = ensure_3ch(arr2)

        # Determine shapes
        is_grayscale = arr1.ndim == 2
        if is_grayscale:
            h1, w1 = arr1.shape
            h2, w2 = arr2.shape
            fill_val = 0
        else:
            h1, w1, c1 = arr1.shape
            h2, w2, c2 = arr2.shape
            if c1 != c2:
                raise ValueError("Channel mismatch after conversion.")
            fill_val = (0,) * c1
        H = max(h1, h2)
        W = w1 + gap + w2

        # Create combined canvas
        # noinspection PyUnboundLocalVariable
        combined_shape = (H, W) if is_grayscale else (H, W, c1)
        combined = np.full(combined_shape, dtype=arr1.dtype, fill_value=255)

        yoff1 = (H - h1) // 2
        yoff2 = (H - h2) // 2
        combined[yoff1:yoff1 + h1, 0:w1] = arr1
        combined[yoff2:yoff2 + h2, w1 + gap: w1 + gap + w2] = arr2

        # Create figure and axis and display
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        if combined.ndim == 2:
            ax.imshow(combined, cmap='gray')
        else:
            ax.imshow(combined)
        ax.axis('off')
        ax.set_title(f'Star Matching Between Images', fontsize=14, fontweight='bold')

        # Retrieve centroids and stars
        centroids1 = solution1['matched_centroids']  # list of (y, x)
        centroids2 = solution2['matched_centroids']
        stars1 = solution1['matched_stars']
        stars2 = solution2['matched_stars']

        # noinspection PyUnresolvedReferences
        colors = plt.cm.Set3(np.linspace(0, 1, len(common_ids)))

        # Offset for image2 + vertical centering when heights differ
        offset_x = w1 + gap
        y_offset1 = (H - h1) // 2
        y_offset2 = (H - h2) // 2

        # Create labels and lines on images
        for i, (cat_id, idx1, idx2) in enumerate(zip(common_ids, indices1, indices2)):
            color = colors[i]

            # Image1 star
            y1, x1 = centroids1[idx1]
            y1 = y1 + y_offset1
            star1 = stars1[idx1]
            label_star_on_plot(ax, x1, y1, color, cat_id, star1[2])

            # Image2 star
            y2, x2 = centroids2[idx2]
            y2 = y2 + y_offset2
            x2 = x2 + offset_x
            star2 = stars2[idx2]
            label_star_on_plot(ax, x2, y2, color, cat_id, star2[2])

            # Line connecting
            ax.plot([x1, x2], [y1, y2],
                    linestyle='--', linewidth=1, color=color, alpha=0.7)

        plt.tight_layout(pad=0)
        return fig

    print(f"Loading images...")
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    print("Solving Image 1...")
    start_time = time.time()
    solution1 = get_solution_for_image(image1, return_visual=create_composite or show_comparison)
    time1 = time.time() - start_time
    print(f"Solving Image 2...")
    start_time = time.time()
    solution2 = get_solution_for_image(image2, return_visual=create_composite or show_comparison)
    time2 = time.time() - start_time

    # Extract solutions
    sol1 = solution1[0] if isinstance(solution1, tuple) else solution1
    sol2 = solution2[0] if isinstance(solution2, tuple) else solution2
    solved1 = sol1['RA'] is not None
    solved2 = sol2['RA'] is not None

    def print_solved(img_name: str, is_solved: bool, time: float, num_recognized_stars: int):
        if is_solved:
            print(f'✓ {img_name} solved ({time:.2f}: {num_recognized_stars} recognized stars')
            return
        print(f"✗ {img_name} could not be solved")

    print_solved('Image 1', solved1, time1, len(sol1['matched_stars']))
    print_solved('Image 2', solved2, time2, len(sol2['matched_stars']))
    if not (solved1 and solved2):
        print("At least one of the images could not be solved")
        return

    results = {
        "solved1": solved1,
        "solved2": solved2,
        "solution1": sol1,
        "solution2": sol2,
    }

    # Find common stars
    common_ids, indices1, indices2 = find_common_stars(sol1, sol2)
    if len(common_ids) == 0:
        print("✗ No common stars were found between the images")
        return results

    # Print common stars info
    print(f"✓ {len(common_ids)} common stars found:")
    for i, (cat_id, idx1, idx2) in enumerate(zip(common_ids, indices1, indices2)):
        star1 = sol1['matched_stars'][idx1]
        cent1 = sol1['matched_centroids'][idx1]
        cent2 = sol2['matched_centroids'][idx2]
        print(f"  {i + 1}. BSC {cat_id} - Mag: {star1[2]:.1f}")
        print(f"     Image 1: pixel ({cent1[1]:.1f}, {cent1[0]:.1f})")
        print(f"     Image 2: pixel ({cent2[1]:.1f}, {cent2[0]:.1f})")

    # Create cross-image visualization
    if show_comparison:
        cross_match_fig = visualize_cross_image_matches(
            image1, image2, sol1, sol2, common_ids, indices1, indices2
        )
    else:
        cross_match_fig = None
    results["common_stars"] = len(common_ids)
    results["common_catalog_ids"] = common_ids
    results["cross_match_figure"] = cross_match_fig
    results["processing_time"] = {"image1": time1, "image2": time2}

    # Create composite image if requested and enough stars
    if create_composite and len(common_ids) >= 3:
        print(f"\nCreating composite image...")

        # Get corresponding pixel coordinates (x, y format for cv2)
        points1 = np.array([[sol1['matched_centroids'][i][1],
                             sol1['matched_centroids'][i][0]] for i in indices1])
        points2 = np.array([[sol2['matched_centroids'][i][1],
                             sol2['matched_centroids'][i][0]] for i in indices2])

        # Create composite visualization
        transform_matrix = compute_transformation_matrix(points2, points1)
        if transform_matrix is not None:
            results["composite_figure"] = create_composite_figure(
                image1,
                image2,
                transform_matrix,
                solution1=sol1,
                common_stars_tuple=(common_ids, indices1, indices2),
                blend_alpha=.5,
            )
            results["transformation_matrix"] = transform_matrix

            print(f"✓ Composite image created using {len(common_ids)} common stars")
        else:
            print("✗ Could not compute transformation matrix")
    elif len(common_ids) < 3:
        print(f"✗ Cannot create composite image: need ≥3 common stars, found {len(common_ids)}")

    plt.show()
    return results


if __name__ == '__main__':
    GENERATE_DB = False
    CUSTOM_DB_PATH = Path('./Generated_DB/yale_database')
    SOLUTION_KEY_DESCRIPTIONS = {
        'RA': "Right ascension of center of image in degrees",
        'Dec': "Declination of center of image in degrees",
        'Roll': "Rotation of image relative to north celestial pole",
        'FOV': "Calculated horizontal field of view of the provided image",
        'distortion': "Calculated distortion of the provided image",
        'RMSE': "RMS residual of matched stars in arcseconds",
        'Matches': "Number of stars in the image matched to the database",
        'Prob': "Probability that the solution is a false-positive",
        'epoch_equinox': "The celestial RA/Dec equinox reference epoch",
        'epoch_proper_motion': "The epoch the database proper motions were propagated to",
        'T_solve': "Time spent searching for a match in milliseconds",
        'T_extract': "Time spent extracting star centroids in milliseconds",
        'RA_target': "Right ascension in degrees of the target pixel positions",
        'Dec_target': "Declination in degrees of the target pixel positions",
    }
    T3 = tetra3.Tetra3(load_database=None)

    if GENERATE_DB:
        print("Generating bright star database...")
        T3.generate_database(
            min_fov=1,
            max_fov=80,
            star_catalog='bsc5',
            star_max_magnitude=7,
            save_as=CUSTOM_DB_PATH
        )
    logging.getLogger('tetra3.Tetra3').setLevel(logging.WARNING)
    print("Loading bright star database...")
    T3.load_database(CUSTOM_DB_PATH)

    print("Performing", end=' ')
    USAGE = 3
    if USAGE == 1:
        # Usage 1: Single image matching
        print("single image matching:")
        match_stars_in_image("./Images/19.jfif", False)
    elif USAGE == 2:
        # Usage 2: Batch processing
        print("batch image matching:")
        match_stars_in_images_in('./Images')
    elif USAGE == 3:
        # Usage 3: Cross-image matching
        print("cross-image matching:")
        image1_path = "./Images/ST_db1.png"
        image2_path = "./Images/ST_db2.png"
        match_stars_between_images(image1_path, image2_path, show_comparison=True)
