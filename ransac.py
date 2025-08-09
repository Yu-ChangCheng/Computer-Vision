import numpy as np
from homography import compute_homography

def ransac_homography(points1, points2, max_iterations=1000, threshold=10):
    best_inliers = []
    best_H = None
    for i in range(max_iterations):
        # Sample random point correspondences
        idx = np.random.choice(len(points1), size=4, replace=False)
        src_pts = points1[idx]
        dst_pts = points2[idx]

        # Compute homography matrix
        H = compute_homography(src_pts, dst_pts)

        # Count inliers
        inliers = []
        for j in range(len(points1)):
            p1 = np.hstack((points1[j], 1))
            # Project the point using the homography and normalize
            p1_proj = np.dot(H, p1)
            p1_proj /= p1_proj[2]
            d = np.linalg.norm(points2[j] - p1_proj[:2])
            if d < threshold:
                inliers.append(j)

        # Update best homography if better inliers found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = compute_homography(points1[best_inliers], points2[best_inliers])

    return best_H
