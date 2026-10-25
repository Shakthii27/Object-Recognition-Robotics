
import cv2
import numpy as np


def undistort_image(image):
    height, width = image.shape[:2]

    focal_length = float(max(height, width))
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs
    )

    print("\nCamera Calibration Matrix:")
    print(camera_matrix)

    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())

    return undistorted, camera_matrix, dist_coeffs
