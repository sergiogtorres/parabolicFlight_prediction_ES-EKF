import cv2
import numpy as np



def get_HSV_mask(frame, lower_bound, upper_bound, radius):

	# Blur for homogeneity. Otherwise mask is patchy
    blur = cv2.GaussianBlur(frame,(5,5),0)

    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Kernel controls size of morphological operations
    # Must be odd number
    kernel_size = radius*2+1
    kernel = np.ones((kernel_size, kernel_size))

    # Morphological Open (removes small patches,
    # e.g., background that was wrongly detected)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Morphological Close (removes, i.e., fills up,
    # undetected patches in detected object)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask, hsv_frame
