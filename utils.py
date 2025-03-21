import numpy as np
import cv2
import HSV_filter as hsv
import shapes

from matplotlib import pyplot as plt

black_color = (0, 0, 0)
no_detection_color = (245, 115, 115)[::-1]
detection_color = (72, 217, 101)[::-1]
org_1 = (20, 50)
org_2 = (20, 80)
def reproject_points(image_points, object_points, rvec, tvec, k):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, k, None)
    projected_points = projected_points.squeeze()

    plt.scatter(image_points[:, 0], image_points[:, 1], c='r', label="Image Points")
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='black', s = 5, label="Projected Points")
    plt.legend()
    plt.show()
    return

def print_outlined_text(img, text, org, scale, color, thickness_smaller):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, black_color, thickness_smaller + 1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness_smaller)
def get_circle_info(img_left, img_right, lower_bound, upper_bound, pixels_val_left, pixels_val_right, camera_parameters, print_on_screen = True):
    (cameraMatrixL, newCameraMatrixL, roiL, projMatrixL, distL, rectL,
     cameraMatrixR, newCameraMatrixR, roiR, projMatrixR, distR, rectR,
     rot, trans, essentialMatrix, fundamentalMatrix) = camera_parameters
    # HSV filtering:
    mask_left, hsv_left = hsv.get_HSV_mask(img_left, lower_bound, upper_bound, pixels_val_left)
    mask_right, hsv_right = hsv.get_HSV_mask(img_right, lower_bound, upper_bound, pixels_val_right)


    res_left = cv2.bitwise_and(img_left, img_left, mask=mask_left)
    res_right = cv2.bitwise_and(img_right, img_right, mask=mask_right)


    circles_left, circle_shape_left  = shapes.get_circles(img_left, mask_left)
    circles_right, circle_shape_right = shapes.get_circles(img_right, mask_right)

    is_detection_missing = np.all(circles_right) == None or np.all(circles_left) == None
    xyz_homogeneous_norm = None

    if is_detection_missing:
        if print_on_screen:
            print_outlined_text(img_right, "No detections", org_1, 0.7, no_detection_color, 2)
            print_outlined_text(img_left, "No detections", org_1, 0.7, no_detection_color, 2)

    else: # if ball detected in both images

        circles_left_np = np.array(circles_left, dtype=float)
        circles_right_np = np.array(circles_right, dtype=float)
        circles_left_corrected = cv2.undistortPoints(circles_left_np, cameraMatrixL, distL, rectL, newCameraMatrixL)
        circles_right_corrected = cv2.undistortPoints(circles_right_np, cameraMatrixR, distR, rectR, newCameraMatrixR)

        xyz_homogeneous = cv2.triangulatePoints(projMatrixL, projMatrixR, circles_left_corrected, circles_right_corrected)
        #xyz_homogeneous = cv2.triangulatePoints(projMatrixL, projMatrixR, circles_left_np, circles_right_np)
        xyz_homogeneous_norm = (xyz_homogeneous / xyz_homogeneous[-1]) / 1000     # in m
        xyz_homogeneous_norm_round = np.round(xyz_homogeneous_norm * 100, 1)  # in cm
        xyz_string = str(xyz_homogeneous_norm_round[0, 0]) + ", " + str(xyz_homogeneous_norm_round[1, 0]) + ", " + str(
            xyz_homogeneous_norm_round[2, 0])

        if print_on_screen:
            print_outlined_text(img_right, "Ball detected", org_1, 0.7, detection_color, 2)
            print_outlined_text(img_left, "Ball detections", org_1, 0.7, detection_color, 2)
            print_outlined_text(img_right, "pos: " + xyz_string, org_2, 0.7, detection_color, 2)
            print_outlined_text(img_left, "pos: " + xyz_string, org_2, 0.7, detection_color, 2)

    return is_detection_missing, xyz_homogeneous_norm, circles_left, circles_right, mask_right, mask_left, circle_shape_left, circle_shape_right