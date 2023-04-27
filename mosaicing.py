import cv2
import numpy as np
from homography import compute_homography

def mosaicing_images(images, homographies):
    # Compute mosaic dimensions
    corners = []
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        corners.append(np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32))
    corners = np.array(corners, dtype=object)
    corners = cv2.transform(corners, homographies)
    corners = np.concatenate(corners, axis=0)
    x_min, y_min = np.int32(corners.min(axis=0).ravel())
    x_max, y_max = np.int32(corners.max(axis=0).ravel())
    t = [-x_min, -y_min]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Create mosaic
    dst_shape = (y_max - y_min, x_max - x_min, images[0].shape[2])
    dst = np.zeros(dst_shape, dtype=np.uint8)
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        warped = cv2.warpPerspective(images[i], Ht.dot(homographies[i]), dst_shape[::-1])
        dst[t[1]:t[1]+h, t[0]:t[0]+w] = warped[t[1]:t[1]+h, t[0]:t[0]+w]

    return dst
