import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
import utils

# UTILITIES
import HSV_filter as hsv

# Read data from pickle file
with open('data/camera_settings.pkl', 'rb') as f:
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
    camera_parameters = pickle.load(f)

(cameraMatrixL, newCameraMatrixL, roiL, projMatrixL, distL, rectL,
 cameraMatrixR, newCameraMatrixR, roiR, projMatrixR, distR, rectR,
 rot, trans, essentialMatrix, fundamentalMatrix) = camera_parameters

# Open cameras
cap_right = cv2.VideoCapture(1)
cap_left =  cv2.VideoCapture(0)

# Set manual exposure if supported (some cameras require this)
# https://github.com/opencv/opencv/issues/9738
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)
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







# Initializing Kalman filter variables
# Other constants
NN = 1000 # size of rolling array
p_est = utils.CircularArrayNP([3, NN], long_axis = 1)#np.zeros([NN, 3])

idx = 0
while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, img_right = cap_right.read()
    succes_left, img_left = cap_left.read()

    circle_data  = utils.get_circle_info(img_left, img_right, lower_bound, upper_bound, pixels_val_left, pixels_val_right, camera_parameters)
    is_detection_missing, xyz_homogeneous_norm, circles_left, circles_right, mask_right, mask_left, circle_shape_left, circle_shape_right = circle_data
    if not is_detection_missing:
        p_est.add_data(np.squeeze(xyz_homogeneous_norm[:3]))
    ###########################


    # Show the frames
    cv2.imshow("img right", img_right)
    cv2.imshow("img left", img_left)
    cv2.imshow("mask right", mask_right)
    cv2.imshow("mask left", mask_left)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################


    idx += 1

    # Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if cv2.waitKey(1) & 0xFF == ord('e'):


x, y, z = p_est.get_all_data()
utils.visualize_trajectory(x, y, z)

# Terminate
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
