import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt

# UTILITIES
import HSV_filter
import shapes as shape

def do_nothing(x):
    pass

# Create windows
cv2.namedWindow("HSV Adjustments")
cv2.namedWindow("Other Adjustments")

# Create trackbars for HSV limits
cv2.createTrackbar("H Min", "HSV Adjustments", 0, 179, do_nothing)
cv2.createTrackbar("H Max", "HSV Adjustments", 179, 179, do_nothing)
cv2.createTrackbar("S Min", "HSV Adjustments", 0, 255, do_nothing)
cv2.createTrackbar("S Max", "HSV Adjustments", 255, 255, do_nothing)
cv2.createTrackbar("V Min", "HSV Adjustments", 0, 255, do_nothing)
cv2.createTrackbar("V Max", "HSV Adjustments", 255, 255, do_nothing)

# Create exposure and morphological operation pixel radius trackbars
cv2.createTrackbar("Exposure Right", "Other Adjustments", 0, 150, do_nothing)  # Adjust range as needed
cv2.createTrackbar("Exposure Left", "Other Adjustments", 0, 150, do_nothing)  # Adjust range as needed
cv2.createTrackbar("Pixels Right", "Other Adjustments", 0, 1000, do_nothing)  # Adjust range as needed
cv2.createTrackbar("Pixels Left", "Other Adjustments", 0, 1000, do_nothing)  # Adjust range as needed




with open("data/data.pkl", "rb") as f:
    (cameraMatrixL, newCameraMatrixL, roi_L, projMatrixL, distL, rectL,
     cameraMatrixR, newCameraMatrixR, roi_R, projMatrixR, distR, rectR,
     rot, trans, essentialMatrix, fundamentalMatrix) = pickle.load(f)



# Open  cameras
cap_right = cv2.VideoCapture(1)
cap_left =  cv2.VideoCapture(0)

# Set manual exposure if supported (some cameras require this)
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual, 0.75 = Auto (varies by camera)

depth = 0
while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, img_right = cap_right.read()
    succes_left, img_left = cap_left.read()

    #########
    # Get trackbar positions
    exposure_val_right = (cv2.getTrackbarPos("Exposure Right", "Other Adjustments"))/10
    exposure_val_left = (cv2.getTrackbarPos("Exposure Left", "Other Adjustments"))/10

    pixels_val_right = (cv2.getTrackbarPos("Pixels Right", "Other Adjustments"))//10
    pixels_val_left = (cv2.getTrackbarPos("Pixels Left", "Other Adjustments"))//10

    h_min = cv2.getTrackbarPos("H Min", "HSV Adjustments")
    h_max = cv2.getTrackbarPos("H Max", "HSV Adjustments")
    s_min = cv2.getTrackbarPos("S Min", "HSV Adjustments")
    s_max = cv2.getTrackbarPos("S Max", "HSV Adjustments")
    v_min = cv2.getTrackbarPos("V Min", "HSV Adjustments")
    v_max = cv2.getTrackbarPos("V Max", "HSV Adjustments")

    # get lower and upper bounds for HSV filtering
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    ###########

    # set exposure (check comments in main scrip)
    cap_right.set(cv2.CAP_PROP_EXPOSURE, -exposure_val_right)
    cap_left.set(cv2.CAP_PROP_EXPOSURE, -exposure_val_left)
    ##


    # HSV filtering:
    mask_right, hsv_right = HSV_filter.get_HSV_mask(img_right, lower_bound, upper_bound, pixels_val_right)
    mask_left, hsv_left = HSV_filter.get_HSV_mask(img_left, lower_bound, upper_bound, pixels_val_left)


    res_right = cv2.bitwise_and(img_right, img_right, mask=mask_right)
    res_left = cv2.bitwise_and(img_left, img_left, mask=mask_left)

    circles_right = shape.get_circles(img_right, mask_right)
    circles_left  = shape.get_circles(img_left, mask_left)
    ##########################

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
    #    pass

data = {
    'exposure_val_right': exposure_val_right,
    'exposure_val_left': exposure_val_left,
    'pixels_val_right': pixels_val_right,
    'pixels_val_left': pixels_val_left,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound
}
# Write the data to a pickle file
with open('settings.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Data saved to settings.pkl")

# Terminate
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
