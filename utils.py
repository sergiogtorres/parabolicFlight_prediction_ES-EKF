import numpy as np
import cv2

from matplotlib import pyplot as plt

def reproject_points(image_points, object_points, rvec, tvec, k)
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, k, None)
    projected_points = projected_points.squeeze()

    plt.scatter(image_points[:, 0], image_points[:, 1], c='r', label="Image Points")
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='black', s = 5, label="Projected Points")
    plt.legend()
    plt.show()
    return