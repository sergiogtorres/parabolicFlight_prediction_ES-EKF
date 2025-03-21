import numpy as np
import cv2
import HSV_filter as hsv
import shapes

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

black_color = (0, 0, 0)
no_detection_color = (245, 115, 115)[::-1] #RGB to BGR
detection_color = (72, 217, 101)[::-1]
org_1 = (20, 50)
org_2 = (20, 80)
org_3 = (20, 110)
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
    xyz_homogeneous_norm_m = None

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
        xyz_homogeneous_norm_m = (xyz_homogeneous / xyz_homogeneous[-1]) / 1000     # in m
        xyz_homogeneous_norm_round_cm = np.round(xyz_homogeneous_norm_m * 100, 1)  # in cm
        xyz_string = str(xyz_homogeneous_norm_round_cm[0, 0]) + ", " + str(xyz_homogeneous_norm_round_cm[1, 0]) + ", " + str(
            xyz_homogeneous_norm_round_cm[2, 0])
        range_cm = np.linalg.norm(xyz_homogeneous_norm_round_cm[:3])
        range_string = str(np.round(range_cm, 1))

        if print_on_screen:
            print_outlined_text(img_right, "Ball detected", org_1, 0.7, detection_color, 2)
            print_outlined_text(img_left, "Ball detections", org_1, 0.7, detection_color, 2)
            print_outlined_text(img_right, "pos x, y, z [cm]: " + xyz_string, org_2, 0.7, detection_color, 2)
            print_outlined_text(img_left, "pos x, y, z [cm]: " + xyz_string, org_2, 0.7, detection_color, 2)
            print_outlined_text(img_right, "range [cm]: " + range_string, org_3, 0.7, detection_color, 2)
            print_outlined_text(img_left, "range [cm]: " + range_string, org_3, 0.7, detection_color, 2)

    return is_detection_missing, xyz_homogeneous_norm_m, circles_left, circles_right, mask_right, mask_left, circle_shape_left, circle_shape_right


class CircularArrayNP:
    def __init__(self, size, long_axis):
        if long_axis != 1:
            print ("long axis other than 1 Not implemented, check class CircularArrayNP!")
        self.size = size[long_axis]
        self.buffer = np.empty(size, dtype=float)  # Create an empty array
        self.index = 0  # Pointer to track the current position

    def __repr__(self):
        # Return a string representation of the buffer when printed
        return f"CircularArrayNP(buffer={self.buffer})"
    def add_data(self, data):
        self.buffer[:, self.index] = data
        self.index = (self.index + 1) % self.size  # Use modulo to wrap the index

    def get_latest(self):
        return self.buffer[(self.index - 1) % self.size]  # Most recent data

    def get_all_data(self):
        # Return all data (older data will appear first)
        return np.hstack((self.buffer[:, self.index:], self.buffer[:, :self.index]))



def visualize_trajectory(x, y, z):
    trajectory_fig = plt.figure()
    ax = trajectory_fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Estimated')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Estimated Trajectory - camera frame')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.zaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(loc=(0.62, 0.77))
    ax.view_init(elev=45, azim=-50)
    plt.show()
    while plt.fignum_exists(1):  # Keeps the plot window open until manually closed
        plt.pause(0.1)