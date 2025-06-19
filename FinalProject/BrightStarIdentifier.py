from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import numpy as np
import os
import time
import logging
import tetra3


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

        circle = patches.Circle((x, y), radius=15, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.annotate(
            f'BSC {cat_id} (Mag: {star_data[2]:.1f})',
            xy=(x, y), xytext=(x + 20, y - 20),
            fontsize=7, color=color, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor=color),
            arrowprops=dict(arrowstyle='->', color=color, lw=1)
        )

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


if __name__ == '__main__':
    GENERATE_DB = False  # Whether to create a new database for star-matching
    CUSTOM_DB_PATH = Path('./Generated_DB/yale_database')  # Where to generate the database or to load a database from
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
        'epoch_proper_motion': "The epoch the database proper motions were propagated 1to",
        'T_solve': "Time spent searching for a match in milliseconds",
        'T_extract': "Time spent extracting star centroids in milliseconds",
        'RA_target': "Right ascension in degrees of the target pixel positions",
        'Dec_target': "Declination in degrees of the target pixel positions",
    }  # Short descriptions from the documentation for the solution dictionary
    T3 = tetra3.Tetra3(load_database=None)  # The Tetra3 class variable for this execution

    if GENERATE_DB:
        print("Generating bright star database...")
        T3.generate_database(
            min_fov=1,
            max_fov=80,
            star_catalog='bsc5',  # Yale Bright Star Catalog
            star_max_magnitude=7,  # Magnitude limit (Yale catalog range)
            # pattern_stars_per_fov=10,  # Pattern density for matching
            # verification_stars_per_fov=20,  # Stars for solution verification
            # star_min_separation=0.05,  # Minimum star separation (degrees)
            # pattern_max_error=0.005,  # Pattern matching tolerance
            save_as=CUSTOM_DB_PATH
        )
    logging.getLogger('tetra3.Tetra3').setLevel(logging.WARNING)
    print("Loading bright star database...")
    T3.load_database(CUSTOM_DB_PATH)

    print("Detecting and matching stars...")
    match_stars_in_image("./Images/19.jfif", True)
    # match_stars_in_images_in('Images')
