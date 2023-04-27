import numpy as np
from scipy.optimize import least_squares

def compute_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    return H / H[2, 2]

def refine_homography(points1, points2, H):
    x0 = H.flatten()
    x1 = least_squares(fun, x0, method='lm', args=(points1, points2)).x
    return x1.reshape((3, 3))

def fun(x, points1, points2):
    H = x.reshape((3, 3))
    p1_hat = np.dot(H, np.hstack((points1, np.ones((len(points1), 1)))).T)
    p1_hat /= p1_hat[2, :]
    errors = (p1_hat[:2, :].T - points2)**2
    return errors.ravel()
