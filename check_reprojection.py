import numpy as np
import cv2 as cv2
import glob
import pickle


chessboard_size = (9, 6)
frame_size = (640, 480)


# termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

chessboard_square_size_mm = 8.7#26
objp = objp * chessboard_square_size_mm



with open("data/data.pkl", "rb") as f:  # "rb" = read binary mode
    camera_parameters = pickle.load(f)

(cameraMatrixL, newCameraMatrixL, roiL, projMatrixL, distL, rectL,
 cameraMatrixR, newCameraMatrixR, roiR, projMatrixR, distR, rectR,
 rot, trans, essentialMatrix, fundamentalMatrix) = camera_parameters

# Open cameras
cap_right = cv2.VideoCapture(1)
cap_left =  cv2.VideoCapture(0)
i = 0
#rvec_camL2R, _ = cv2.Rodrigues(rot)
trans_L2R = trans
while(cap_right.isOpened() and cap_left.isOpened()):
    print (i)
    i+=1
    succes_right, img_right = cap_right.read()
    succes_left, img_left = cap_left.read()

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret_left:

        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), termination_criteria)

        ret_chessboard, rvec_chessboard, tvec_chessboard = cv2.solvePnP(objp, corners_left, cameraMatrixL, distL)

        # Project the 3D chessboard points to Camera 2
        R_chessboard, _ = cv2.Rodrigues(rvec_chessboard)
        # Transform chessboard pose to Camera 2's coordinate frame
        R_chessboard_cam2 = rot @ R_chessboard
        t_chessboard_cam2 = rot @ tvec_chessboard + trans
        # Convert back to Rodrigues vector
        rvec_chessboard_cam2, _ = cv2.Rodrigues(R_chessboard_cam2)

        chessboard_proj, _ = cv2.projectPoints(objp, rvec_chessboard_cam2, t_chessboard_cam2, cameraMatrixR, distR)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        #cv2.imshow('img_left left', img_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, chessboard_proj, ret_chessboard)
        #cv2.imshow('img_left right', img_right)

    # Show the frames
    cv2.imshow("img right", img_right)
    cv2.imshow("img left", img_left)

    # Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break