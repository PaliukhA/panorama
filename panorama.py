import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv
import random

DEFAULT_TRANSFORM = ProjectiveTransform

from skimage import data


def find_orb(img, n_keypoints=200):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    # your code here

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(rgb2gray(img))
    return (descriptor_extractor.keypoints,
            descriptor_extractor.descriptors)


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))

    # your code here
    cx, cy = points.mean(axis=0)
    pts = points.astype('float')
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    n = np.sqrt(2) / np.sqrt((pts * pts).sum(axis=1)).mean()
    matrix = np.array([[n, 0, -n * cx], [0, n, -n * cy], [0, 0, 1]])
    return matrix, matrix @ pointsh


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))

    # your code here
    x1 = src[0]
    y1 = src[1]
    x2 = dest[0]
    y2 = dest[1]
    zeros = np.zeros(x1.size)
    ones = np.ones(x1.size)

    Ax = np.stack((-x1, -y1, -ones, zeros, zeros, zeros, x2 * x1, x2 * y1, x2)).T
    Ay = np.stack((zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2)).T
    A = np.concatenate([Ax, Ay], axis=0)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    result = vh[-1].reshape((3, 3))
    return inv(dest_matrix) @ result @ src_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=25,
                     residual_threshold=10,
                     return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    # your code here
    matches = match_descriptors(src_descriptors, dest_descriptors, cross_check=True)
    srt_matches = src_keypoints[matches[..., 0]]
    dest_matches = dest_keypoints[matches[..., 1]]
    size = matches.shape[0]
    assert (size > 20)
    random.seed(97)
    min = 1e18
    best_H = np.zeros((3, 3))

    def penalty(H):
        res = np.linalg.norm(dest_matches - ProjectiveTransform(H)(srt_matches), axis=1)
        res[res > residual_threshold] = residual_threshold
        return res.sum()

    for i in range(max_trials):
        subset = random.choices(np.arange(size), k=4)
        while np.unique(subset).size < 4:
            subset = random.choices(np.arange(size), k=4)
        H = find_homography(srt_matches[subset], dest_matches[subset])
        pen = penalty(H)
        if pen < min:
            min = pen
            best_H = H

    mask = np.ones(size)
    res = np.linalg.norm(dest_matches - ProjectiveTransform(best_H)(srt_matches), axis=1)
    mask[res > residual_threshold] = False
    mask[res <= residual_threshold] = True
    res[res > residual_threshold] = residual_threshold
    H = find_homography(srt_matches[mask == True], dest_matches[mask == True])
    print("match=", len(matches[mask == True]))
    if return_matches:
        return ProjectiveTransform(H), matches[mask == True]
    else:
        return ProjectiveTransform(H)


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    # np.linalg.inv(forward_transforms[0])
    # your code here
    for i in range(center_index, image_count - 1):
        result[i + 1] = result[i] + ProjectiveTransform(forward_transforms[i]._inv_matrix)
    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i]

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    # your code here
    all_corners = list(get_corners(image_collection, simple_center_warps))
    arr = np.array(all_corners)
    corners = get_min_max_coords(arr)
    height = corners[1][1] - corners[0][1]
    width = corners[1][0] - corners[0][0]
    dest = [[0, 0], [height, 0], [0, width], [height, width]]
    src = [[corners[0][1], corners[0][0]],
           [corners[1][1], corners[0][0]],
           [corners[0][1], corners[1][0]],
           [corners[1][1], corners[1][0]]]
    src = np.array(src)
    dest = np.array(dest)
    transform = find_homography(src, dest)
    result = []
    for warm in simple_center_warps:
        result.append(warm + ProjectiveTransform(transform))
    return tuple(result), (int(round(height)), int(round(width)))


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform._inv_matrix[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    # your code here
    mask = np.ones_like(image[..., 0])
    sl0 = slice(0, mask.shape[0])
    sl1 = slice(0, mask.shape[1])
    new_image = np.zeros((output_shape[0], output_shape[1], 3))
    new_image[sl0, sl1] += image[sl0, sl1]
    new_mask = np.zeros((output_shape[0], output_shape[1]))
    new_mask[sl0, sl1] += mask[sl0, sl1]

    return warp(new_image, rotate_transform_matrix(transform)), \
           warp(new_mask, rotate_transform_matrix(transform)).astype('bool')


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    # your code here

    for img, warp in zip(image_collection, final_center_warps):
        cur_img, cur_mask = warp_image(img, warp, output_shape)
        cur_mask[result_mask == True] = False
        result_mask[cur_mask == True] = True
        msk = np.concatenate([cur_mask.reshape(output_shape + (1,))] * 3, axis=2)
        np.putmask(result, msk, cur_img)
    return result


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    # your code here
    result = []
    cur_image = image
    for i in range(n_layers):
        result.append(cur_image)
        cur_image = gaussian(cur_image, sigma=sigma)
    return tuple(result)


def get_laplacian_pyramid(image, n_layers=15, sigma=1.3):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    # your code here
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    result = []
    for i in range(n_layers - 1):
        result.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])
    result.append(gaussian_pyramid[n_layers - 1])
    return tuple(result)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=15, image_sigma=6,
                        merge_sigma=6):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    # your code here

    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    iter = 0
    for img, warp in zip(image_collection, final_center_warps):
        cur_img, cur_mask = warp_image(img, warp, output_shape)

        intersect = np.zeros(output_shape)
        np.putmask(intersect, cur_mask, result_mask.astype('float'))

        cnts = intersect.sum(axis=0)
        cnt = cnts.sum()
        cur_sum = 0
        i = 0
        print("in")
        while (cur_sum < cnt // 2):
            cur_sum += cnts[i]
            i += 1
        print("out")
        half = np.ones_like(intersect)
        half[..., i:] = 0

        la = get_laplacian_pyramid(result, n_layers, image_sigma)
        lb = get_laplacian_pyramid(cur_img, n_layers, image_sigma)
        gm = get_gaussian_pyramid(half, n_layers, merge_sigma)
        la = np.array(la)
        lb = np.array(lb)
        gm = np.array(gm)
        gm = np.concatenate([gm.reshape((n_layers,) + output_shape + (1,))] * 3, axis=3)
        gm_ = np.ones_like(gm) - gm
        inter_pyramid = la * gm + lb * gm_
        inter = inter_pyramid.sum(axis=0)

        msk = np.concatenate([intersect.reshape(output_shape + (1,))] * 3, axis=2).astype('bool')
        np.putmask(result, msk, inter)

        cur_mask[result_mask == True] = False
        result_mask[cur_mask == True] = True

        msk = np.concatenate([cur_mask.reshape(output_shape + (1,))] * 3, axis=2)
        np.putmask(result, msk, cur_img)
        print("Step ", iter)
        iter += 1
    return result
