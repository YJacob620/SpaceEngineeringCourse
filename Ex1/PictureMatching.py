import numpy as np
import cv2
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import os
from StarDetection import detect_stars

# IMAGE_PATH_1 = "./star_images_2/IMG_3046.jpg"
# IMAGE_PATH_2 = "./star_images_2/IMG_3063.jpg"
# IMAGE_PATH_1 = "./star_images/fr1.jpg"
IMAGE_PATH_2 = "./star_images/fr2.jpg"
IMAGE_PATH_1 = "./star_images/ST_db1.png"
# IMAGE_PATH_2 = "./star_images/ST_db2.png"

PRINTS = 1  # Whether to print additional information
TMP_FOLDER_PATH = "./temp/"
if not os.path.exists(TMP_FOLDER_PATH):
    os.mkdir(TMP_FOLDER_PATH)


def compute_triangle_descriptors(pts: np.ndarray) -> List[Tuple[float, float, Tuple[int, int, int]]]:
    """
    A triangle descriptor consists of the ratios of the smallest and medium sides
    to the largest side of the triangle, along with the indices of the points.
    These ratios are invariant to scale, rotation, and translation.

    Args:
        pts: Array of shape (N, 2) containing N 2D points

    Returns:
        List of descriptors, each being (ratio1, ratio2, (i,j,k)) where:
            - ratio1: smallest_side/largest_side
            - ratio2: medium_side/largest_side
            - (i,j,k): Indices of the three points forming the triangle
    """
    descriptors = []
    for i, j, k in combinations(range(len(pts)), 3):
        p1, p2, p3 = pts[i], pts[j], pts[k]
        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d31 = np.linalg.norm(p3 - p1)
        ds = np.array([d12, d23, d31])
        d_small, d_med, d_large = np.sort(ds)  # sort to get smallest, middle, largest
        if d_large == 0:
            continue
        descriptors.append((d_small / d_large, d_med / d_large, (i, j, k)))
    return descriptors


def estimate_affine(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate affine transformation (M, t) that maps source points to target points.

    The transformation is: target = source @ M.T + t

    Args:
        source_points: Array of source points (N, 2)
        target_points: Array of target points (N, 2)

    Returns:
        Tuple of (M, t) where:
            - M: 2x2 transformation matrix
            - t: Translation vector of shape (2,)
    """
    num_points = source_points.shape[0]
    equation_matrix = np.zeros((2 * num_points, 6))
    target_vector = np.zeros((2 * num_points,))

    for i in range(num_points):
        x, y = source_points[i]
        target_x, target_y = target_points[i]
        equation_matrix[2 * i] = [x, y, 0, 0, 1, 0]
        equation_matrix[2 * i + 1] = [0, 0, x, y, 0, 1]
        target_vector[2 * i] = target_x
        target_vector[2 * i + 1] = target_y

    params, *_ = np.linalg.lstsq(equation_matrix, target_vector, rcond=None)
    transformation_matrix = np.array([[params[0], params[1]], [params[2], params[3]]])
    translation_vector = np.array([params[4], params[5]])

    return transformation_matrix, translation_vector


def select_best_warp_matrix(
        results: list,
        path_a: str,
        path_b: str,
        top_k: int = 5,
        max_radius: int = 30
) -> np.ndarray:
    """
    From a list of affine-candidate results, warp image B for the top_k matches,
    detect stars in each warped image, compute the sum of distances from stars in A
    to nearest stars in warped B, and return the warp_matrix with minimal total loss.

    Args:
        results: List of dicts, each with 'M', 't', and 'inlier_ratio'.
        path_a: Path to reference image A.
        path_b: Path to image B to warp.
        top_k: Number of top candidates (by inlier_ratio) to evaluate.
        max_radius: passed to detect_stars for warp detection.

    Returns:
        2×3 affine warp_matrix for cv2.warpAffine.
    """
    # Load and detect stars in image A
    img_a = cv2.imread(path_a)
    h_a, w_a = img_a.shape[:2]
    stars_a = detect_stars(path_a)
    points_a = np.array([(x, y) for x, y, *_ in stars_a])

    # Load image B
    img_b = cv2.imread(path_b)

    # Sort candidates and take top_k
    sorted_res = sorted(results, key=lambda r: r['inlier_ratio'], reverse=True)
    candidates = sorted_res[:min(top_k, len(sorted_res))]

    best_loss = np.inf
    best_warp = None

    for r in candidates:
        M, t = r['M'], r['t']
        # Build inverse warp (to map B→A coords)
        invM = np.linalg.inv(M)
        invt = -invM @ t
        warp_mat = np.hstack([invM, invt.reshape(2, 1)])  # 2×3 matrix

        # Apply warp
        warped = cv2.warpAffine(img_b, warp_mat, (w_a, h_a))
        cv2.imwrite(f'{TMP_FOLDER_PATH}tmp_warp.jpg', warped)

        # Detect stars in warped image
        stars_w = detect_stars(f'{TMP_FOLDER_PATH}tmp_warp.jpg', max_radius=max_radius)
        if not stars_w:
            continue
        points_w = np.array([(x, y) for x, y, *_ in stars_w])

        # Compute loss: sum of nearest-neighbor distances from A to warped B
        dists, _ = KDTree(points_w).query(points_a)
        loss = np.sum(dists)

        if loss < best_loss:
            best_loss = loss
            best_warp = warp_mat

    if best_warp is None:
        raise RuntimeError("No valid warp found among top candidates")
    return best_warp


def match_star_fields(
        path_a: str,
        path_b: str,
        max_stars: int = 100,
        ratio_tol: float = 0.01,
        inlier_tol: float = 5.0
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[float]]:
    """
    Match stars between two astronomical images.

    This function detects stars in both images, computes triangle descriptors to find
    corresponding star patterns, and determines the transformation between the images.

    Args:
        path_a: Path to first image
        path_b: Path to second image
        max_stars: Maximum number of stars to use for initial matching.
        ratio_tol: Tolerance for matching triangle descriptors
        inlier_tol: Tolerance (in thousandths of image diagonal) for matching points

    Returns:
        Tuple of (M, t, matched_coordinates, quality_ratings) where:
            - M: 2x2 transformation matrix (None if no match found)
            - t: Translation vector (None if no match found)
            - matched_coordinates: List of matched point pairs ((x_a, y_a), (x_b, y_b))
            - quality_ratings: List of match quality ratings (0-100)
    """
    # Detect and sample stars
    stars_a_full = detect_stars(path_a)
    stars_b_full = detect_stars(path_b)

    # Sort by size and brightness and limit to max_stars
    stars_a = sorted(stars_a_full, key=lambda s: 0.5 * s[3] + 0.5 * s[2], reverse=True)[:max_stars]
    stars_b = sorted(stars_b_full, key=lambda s: 0.5 * s[3] + 0.5 * s[2], reverse=True)[:max_stars]

    # Load images and get dimensions
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]

    # Extract coordinates
    points_a = np.array([(x, y) for x, y, *_ in stars_a])
    points_b = np.array([(x, y) for x, y, *_ in stars_b])

    # Ensure points_a is the smaller set (fewer stars)
    if len(points_a) > len(points_b):
        points_a, points_b = points_b, points_a
        stars_a, stars_b = stars_b, stars_a
        img_a, img_b = img_b, img_a
        height_a, height_b = height_b, height_a
        width_a, width_b = width_b, width_a

    # Calculate adaptive inlier tolerance based on image size
    diagonal = np.hypot(max(height_a, height_b), max(width_a, width_b))
    adaptive_tolerance = inlier_tol * diagonal / 1000

    # Compute triangle descriptors for sets A and B
    if PRINTS > 0:
        print("Computing star-triangles descriptors")
    descriptors_a = compute_triangle_descriptors(points_a)
    tree_a = KDTree(np.array([[r1, r2] for r1, r2, _ in descriptors_a]))
    samples_b = compute_triangle_descriptors(points_b)

    # Process samples
    if PRINTS > 0:
        print("Matching descriptors")
    results = []
    for sample in samples_b:
        # Unpack and query
        r1, r2, idxB = sample
        dist, idxA = tree_a.query([r1, r2])
        if dist > ratio_tol:
            continue

        # Extract triangles
        iA1, iA2, iA3 = descriptors_a[idxA][2]
        A_tri = points_a[[iA1, iA2, iA3]]
        B_tri = points_b[list(idxB)]

        # Estimate and count inliers
        M, t = estimate_affine(A_tri, B_tri)
        A_trans = points_a @ M.T + t
        dists, _ = KDTree(points_b).query(A_trans)
        inliers = np.sum(dists < adaptive_tolerance)
        results.append({'inliers': inliers, 'M': M, 't': t, 'inlier_ratio': inliers / len(points_a)})
    if not results:
        print("No matches")
        return [], []

    # best_match = max(results, key=lambda r: r['inlier_ratio'])
    #
    # # Warp B into A coordinates
    # inverse_matrix = np.linalg.inv(best_match['M'])
    # inverse_translation = -inverse_matrix @ best_match['t']
    # warp_matrix = np.hstack([inverse_matrix, inverse_translation.reshape(2, 1)])

    # Find best warp matrix for image B
    if PRINTS > 0:
        print("Finding best match")
    warp_matrix = select_best_warp_matrix(results, path_a, path_b)
    inv_warp_matrix = cv2.invertAffineTransform(warp_matrix)  # Invertion
    warped_image = cv2.warpAffine(img_b, warp_matrix, (width_a, height_a))
    cv2.imwrite(f'{TMP_FOLDER_PATH}warped_to_A.jpg', warped_image)

    # Find nearest matches between A and warped B
    if PRINTS > 0:
        print("Grading matching")
    all_stars_warped = detect_stars(f'{TMP_FOLDER_PATH}warped_to_A.jpg')
    if len(all_stars_warped) == 0:
        raise Exception("Star detection on warped image failed")
    all_points_a = np.array([(x, y) for x, y, *_ in stars_a_full])
    all_points_warped = np.array([(x, y) for x, y, *_ in all_stars_warped])
    all_points_b_full = np.array([(x, y) for x, y, *_ in stars_b_full])
    tree_b_full = KDTree(all_points_b_full)  # Create KDTree for original image B stars for precise matching
    distances, indices = KDTree(all_points_warped).query(all_points_a)
    match_mask = distances < adaptive_tolerance

    matched_coordinates = []
    quality_ratings = []

    # for i, (idx, dist) in enumerate(zip(indices, distances)):
    for i, (is_match, idx, dist) in enumerate(zip(match_mask, indices, distances)):
        if not is_match:
            continue
        # Calculate quality score (0-100)
        quality = np.clip(100 - 100 * dist / (2 * adaptive_tolerance), 0, 100)

        # Transform warped coordinates back to original B coordinates
        x_warped, y_warped = all_stars_warped[idx][0], all_stars_warped[idx][1]
        x, y = x_warped, y_warped
        mapped = inv_warp_matrix.dot([x, y, 1])  # shape (2,)
        mapped_point = np.array((mapped[0], mapped[1]))

        # Find the closest actual star in original image B
        b_dist, b_idx = tree_b_full.query(mapped_point.reshape(1, -1))
        x_b, y_b = all_points_b_full[b_idx[0]]

        matched_coordinates.append((
            (stars_a_full[i][0], stars_a_full[i][1]),
            (int(round(x_b)), int(round(y_b)))
        ))

        # Adjust quality based on how well the point matches an actual star in B
        b_quality_factor = np.clip(1.0 - b_dist[0] / adaptive_tolerance, 0.0, 1.0)
        adjusted_quality = quality * b_quality_factor
        quality_ratings.append(adjusted_quality)

    if PRINTS > 0 and len(matched_coordinates) > 0:
        print(f"Found {len(matched_coordinates)} matches among all stars after warp")
    return matched_coordinates, quality_ratings


def annotate_matches(
        path_a: str,
        path_b: str,
        matched_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        output_a: str = f'{TMP_FOLDER_PATH}annotated_A.jpg',
        output_b: str = f'{TMP_FOLDER_PATH}annotated_B.jpg'
) -> None:
    """
    Annotate matched star pairs on both images and save the results.

    Args:
        path_a: Path to first image
        path_b: Path to second image
        matched_coords: List of matched point pairs ((x_a, y_a), (x_b, y_b))
        output_a: Path to save annotated version of first image
        output_b: Path to save annotated version of second image
    """
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    for idx, ((x_a, y_a), (x_b, y_b)) in enumerate(matched_coords, 1):
        cv2.circle(img_a, (x_a, y_a), 6, (0, 255, 0), 2)
        cv2.putText(img_a, str(idx), (x_a + 8, y_a - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img_b, (x_b, y_b), 6, (0, 255, 0), 2)
        cv2.putText(img_b, str(idx), (x_b + 8, y_b - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_a, img_a)
    cv2.imwrite(output_b, img_b)


def display_matches(
        path_a: str,
        path_b: str,
        matched_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        quality_ratings: Optional[List[float]] = None
) -> None:
    """
    Display matched star pairs side by side with color-coded confidence.

    Args:
        path_a: Path to first image
        path_b: Path to second image
        matched_coords: List of matched point pairs ((x_a, y_a), (x_b, y_b))
        quality_ratings: List of match quality ratings (0-100), defaults to all 100
    """
    img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_a)
    axs[0].set_title('Image A')
    axs[0].axis('off')
    axs[1].imshow(img_b)
    axs[1].set_title('Image B')
    axs[1].axis('off')

    if quality_ratings is None:
        quality_ratings = [100] * len(matched_coords)

    # Color map from red (low quality) to green (high quality)
    color_map = ListedColormap(['#ff0000', '#ffff00', '#00ff00'])

    for idx, ((x_a, y_a), (x_b, y_b)) in enumerate(matched_coords, 1):
        quality = quality_ratings[idx - 1] / 100
        color = color_map(quality)
        axs[0].scatter(x_a, y_a, s=50, edgecolors=color, facecolors='none')
        axs[0].text(x_a + 5, y_a - 5, str(idx), color=color)
        axs[1].scatter(x_b, y_b, s=50, edgecolors=color, facecolors='none')
        axs[1].text(x_b + 5, y_b - 5, str(idx), color=color)

    # Add legend
    legend_elements = [
        Patch(edgecolor='#ff0000', facecolor='none', label='Low Confidence'),
        Patch(edgecolor='#ffff00', facecolor='none', label='Medium Confidence'),
        Patch(edgecolor='#00ff00', facecolor='none', label='High Confidence')
    ]
    legend = fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    legend.get_frame().set_facecolor('#d9d9d9')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.show()


if __name__ == "__main__":
    coord_pairs, quality_rating = \
        (match_star_fields(IMAGE_PATH_1, IMAGE_PATH_2, max_stars=20))

    if coord_pairs and len(coord_pairs) > 0:
        print("Matched star pairs:")
        for idx, ((x_a, y_a), (x_b, y_b)) in enumerate(coord_pairs, 1):
            print(f"{idx:3d}: A({x_a:4d},{y_a:4d})  →  B({x_b:4d},{y_b:4d})")
        annotate_matches(IMAGE_PATH_1, IMAGE_PATH_2, coord_pairs)
        display_matches(IMAGE_PATH_1, IMAGE_PATH_2, coord_pairs, quality_rating)
    else:
        print("Failed to find a good match between the images")
