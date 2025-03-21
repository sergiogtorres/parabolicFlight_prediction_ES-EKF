import cv2
import numpy as np
import imutils

def get_circles(img, mask):

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    centroid = x = y = radius = None

    # if > one contour found
    if len(contours) > 0:

        # get largest contour
        contour = max(contours, key=cv2.contourArea)

        # get minimum enlosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # get centroid with moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            # Edge case. sometimes m00 may be zero
            centroid = (0, 0)  #
            print("Warning: m00 is zero, centroid cannot be calculated.")



        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(img, centroid, 5, (0, 0, 0), -1)

    circle_shape = x, y, radius

    return centroid, circle_shape
