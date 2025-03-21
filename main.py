import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt

# UTILITIES
import HSV_filter as hsv
import shapes as shape

# Read data from pickle file
with open('settings.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Access pickled variables
exposure_val_right = loaded_data['exposure_val_right']
exposure_val_left = loaded_data['exposure_val_left']
pixels_val_right = loaded_data['pixels_val_right']
pixels_val_left = loaded_data['pixels_val_left']
lower_bound = loaded_data['lower_bound']
upper_bound = loaded_data['upper_bound']

print(f"exposure_val_right: {exposure_val_right}")
print(f"exposure_val_left: {exposure_val_left}")
print(f"pixels_val_right: {pixels_val_right}")
print(f"pixels_val_left: {pixels_val_left}")
print(f"lower_bound: {lower_bound}")
print(f"upper_bound: {upper_bound}")


with open("data/data.pkl", "rb") as f:  # "rb" = read binary mode
    (cameraMatrixL, newCameraMatrixL, roiL, projMatrixL, distL, rectL,
     cameraMatrixR, newCameraMatrixR, roiR, projMatrixR, distR, rectR,
     rot, trans, essentialMatrix, fundamentalMatrix) = pickle.load(f)



# Open cameras
cap_right = cv2.VideoCapture(1)
cap_left =  cv2.VideoCapture(0)

# Set manual exposure if supported (some cameras require this)
# https://github.com/opencv/opencv/issues/9738
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)
no_detection_color = (245, 115, 115)[::-1]
detection_color = (72, 217, 101)[::-1]
black_color = (0, 0, 0)
org_1 = (20, 50)
org_2 = (20, 80)
# Sometimes, negative values expected, other times, decimals.
# I believe that negative numbers represent a value of 2^(-exposure) seconds
# While positive numbers might be the exposure time as a float (depends on camera)
# ctrl-f in link above:
#
# "I needed to set my exposure as an absolute value,
# 0.01 in my case, rather than using negative values.
# But that may be different for your camera."
#
# Also, check out:
# https://answers.opencv.org/question/233617/cv2cap_prop_exposure-is-different-in-ubuntu-and-windows-and-what-is-the-meaning-of-each-the-value-of-cv2cap_prop_exposure/


cap_right.set(cv2.CAP_PROP_EXPOSURE, -exposure_val_right)  # Negative values often work for webcams
cap_left.set(cv2.CAP_PROP_EXPOSURE, -exposure_val_left)  # Negative values often work for webcams
##

def print_outlined_text(img, text, org, scale, color, thickness_smaller):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, black_color, thickness_smaller+1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness_smaller)

while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, img_right = cap_right.read()
    succes_left, img_left = cap_left.read()




    # HSV filtering:
    mask_right, hsv_right = hsv.get_HSV_mask(img_right, lower_bound, upper_bound, pixels_val_right)
    mask_left, hsv_left = hsv.get_HSV_mask(img_left, lower_bound, upper_bound, pixels_val_left)


    res_right = cv2.bitwise_and(img_right, img_right, mask=mask_right)
    res_left = cv2.bitwise_and(img_left, img_left, mask=mask_left)


    circles_right, circle_shape_right = shape.get_circles(img_right, mask_right)
    circles_left, circle_shape_left  = shape.get_circles(img_left, mask_left)
    ###########################


    if np.all(circles_right) == None or np.all(circles_left) == None:
        print_outlined_text(img_right, "No detections", org_1, 0.7, no_detection_color, 2)
        print_outlined_text(img_left, "No detections", org_1, 0.7, no_detection_color, 2)

    else:
        print_outlined_text(img_right, "Ball detected", org_1, 0.7, detection_color, 2)
        print_outlined_text(img_left, "Ball detections", org_1, 0.7, detection_color, 2)

        circles_left_np = np.array(circles_left, dtype=float)
        circles_right_np = np.array(circles_right, dtype=float)
        circles_left_corrected = cv2.undistortPoints(circles_left_np, cameraMatrixL, distL, rectL, newCameraMatrixL)
        circles_right_corrected = cv2.undistortPoints(circles_right_np, cameraMatrixR, distR, rectR, newCameraMatrixR)

        xyz_homogeneous = cv2.triangulatePoints(projMatrixL, projMatrixR, circles_left_np, circles_right_np)
        xyz_homogeneous_norm = xyz_homogeneous/xyz_homogeneous[-1]
        xyz_homogeneous_norm_round = np.round(xyz_homogeneous_norm, 0)/10
        xyz_string = str(xyz_homogeneous_norm_round[0,0]) + ", " + str(xyz_homogeneous_norm_round[1,0]) + ", " + str(xyz_homogeneous_norm_round[2,0])

        print_outlined_text(img_right, "pos: " + xyz_string, org_2, 0.7, detection_color, 2)
        print_outlined_text(img_left, "pos: " + xyz_string, org_2, 0.7, detection_color, 2)


    # Show the frames
    cv2.imshow("img right", img_right)
    cv2.imshow("img left", img_left)
    cv2.imshow("mask right", mask_right)
    cv2.imshow("mask left", mask_left)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################



    # Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if cv2.waitKey(1) & 0xFF == ord('e'):



# Terminate
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
