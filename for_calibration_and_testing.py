import numpy as np
import cv2 as cv2
import glob
import pickle
import testing_functions

print("double check this implementation. It might be wrong!")

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

    check_reprojection.check_reprojection_function(img_left, img_right, chessboard_size,
                                termination_criteria, objp, cameraMatrixL, distL,
                                cameraMatrixR, distR, rot, trans)

    # Show the frames
    cv2.imshow("img right", img_right)
    cv2.imshow("img left", img_left)

    # Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break