import cv2
import os
import numpy as np


path_left = "data/images/calibrationLeft/"
path_right = "data/images/calibrationRight/"

paths = [path_left, path_right]
for path in paths:
    if not os.path.exists(path):
       os.makedirs(path)

####################
chessboard_size = (9, 6)
frame_size = (640, 480)


# termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

chessboard_square_size_mm = 8.7#26
objp = objp * chessboard_square_size_mm
######################

cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)

img_idx = 0
while (cap_left.isOpened() and cap_right.isOpened()):

    succes_left, img_left = cap_left.read()
    succes_right, img_right = cap_right.read()

    k = cv2.waitKey(1)
    ##################
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    ret_left_chessboard, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right_chessboard, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
    img_left_show = img_left.copy()
    img_right_show = img_right.copy()
    cv2.drawChessboardCorners(img_left_show, chessboard_size, corners_left, ret_left_chessboard)
    cv2.drawChessboardCorners(img_right_show, chessboard_size, corners_right, ret_right_chessboard)


    ##################################
    if k == ord('q'):
        break

    elif k == ord('e'):
        exposure = float(input("please enter exposure to set: "))
        cap_left.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap_right.set(cv2.CAP_PROP_EXPOSURE, exposure)

    elif k == ord('s'):
        cv2.imwrite(path_left + 'imageL' + str(img_idx) + '.png', img_left)
        cv2.imwrite(path_right + 'imageR' + str(img_idx) + '.png', img_right)
        print(f"saved {img_idx+1}-st/nd/rd/th image")
        img_idx += 1

    cv2.imshow('Img L', img_left_show)
    cv2.imshow('Img R', img_right_show)

# Terminate
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()
