import os
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac

def homography(im1, im2):
    orb = ORB(n_keypoints=500, fast_threshold=0.05)
    orb.detect_and_extract(im1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(im2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    model_robust, inliers = ransac((keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]),
                                    transform.ProjectiveTransform, min_samples=4,
                                    residual_threshold=2, max_trials=300)

    return model_robust.params

def fun(x, y):
    x = x.reshape(3, 3)
    y = y.reshape(8, 1)
    y = y.reshape(-1, 2).T
    result = np.dot(x, np.vstack((y, np.ones((1, y.shape[1])))))
    return (result[:-1, :] / result[-1, :]).flatten()

def mosaic(img_out, img_in, H, x_offset, y_offset):
    img_in_warped = transform.warp(img_in, transform.ProjectiveTransform(matrix=H), output_shape=img_out.shape)
    mask_in = (img_in_warped > 0).astype(int)
    img_out = img_out * (1 - mask_in) + img_in_warped * mask_in
    return img_out

def main():
    input_folder = 'images'
    image_filenames = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    image_filenames.sort()

    im1 = io.imread(os.path.join(input_folder, image_filenames[0]))
    im1_gray = rgb2gray(im1)

    mosaic_img = im1.copy()
    x_min, y_min = 0, 0
    x_max, y_max = im1.shape[1], im1.shape[0]

    for i in range(1, len(image_filenames)):
        im2 = io.imread(os.path.join(input_folder, image_filenames[i]))
        im2_gray = rgb2gray(im2)

        H = homography(im1_gray, im2_gray)

        c = fun(H.ravel(), np.array([1, 1, im1.shape[1], 1, 1, im1.shape[0], im1.shape[1], im1.shape[0]]))
        x_min_new = min(x_min, c[0], c[2], c[4], c[6])
        x_max_new = max(x_max, c[0], c[2], c[4], c[6])
        y_min_new = min(y_min, c[1], c[3], c[5], c[7])
        y_max_new = max(y_max, c[1], c[3], c[5], c[7])

        width_new = int(x_max_new - x_min_new) + 1
        height_new = int(y_max_new - y_min_new) + 1

        mosaic_img_new = np.zeros((height_new, width_new, 3))
        mosaic_img_new = mosaic(mosaic_img_new, mosaic_img, np.eye(3), x_min_new - x_min, y_min_new - y_min)
        mosaic_img_new = mosaic(mosaic_img_new, im2, H, x_min_new - x_min, y_min_new - y_min)

        mosaic_img = mosaic_img_new
        x_min, y_min = x_min_new, y_min_new
        x_max, y_max = x_max_new, y_max_new
        im1_gray = im2_gray

    io.imsave("mosaic.jpg", (mosaic_img * 255).astype(np.uint8))
    print("Mosaic image saved as mosaic.jpg")

    
if __name__ == "__main__":
    main()
