import numpy as np
import cv2 as cv2
import glob
import pickle


############ check https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html ######

chessboard_size = (9, 6)
frame_size = (640, 480)


# termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

chessboard_square_size_mm = 8.7#26
objp = objp * chessboard_square_size_mm

# Arrays to store object points and image points from the images.
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane.
imgpoints_right = [] # 2d points in image plane.


img_files_left = sorted(glob.glob('data/images/calibrationLeft/*.png'))
img_files_right = sorted(glob.glob('data/images/calibrationRight/*.png'))

for img_file_left, img_file_right in zip(img_files_left, img_files_right):

    img_left = cv2.imread(img_file_left)
    img_right = cv2.imread(img_file_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret_left and ret_right == True:

        objpoints.append(objp)

        #corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), termination_criteria)
        imgpoints_left.append(corners_left)

        #corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), termination_criteria)
        imgpoints_right.append(corners_right)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.imshow('img_left left', img_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
        cv2.imshow('img_left right', img_right)
        cv2.waitKey(1000)


cv2.destroyAllWindows()




#######   Calibration   #######

ret_left, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, frame_size, None, None)
heightL, widthL, channelsL = img_left.shape
newCameraMatrixL, roiL = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

ret_right, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, frame_size, None, None)
heightR, widthR, channelsR = img_right.shape
newCameraMatrixR, roiR = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



#######   stereo calibration - get extrinsic matrices  #######

# fix intrinsic parameters, only calculate extrinsics
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

(retStereo, newCameraMatrixL, distL,
 newCameraMatrixR, distR, rot, trans,
 essentialMatrix, fundamentalMatrix) = (
    cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                        newCameraMatrixL, distL, newCameraMatrixR,
                        distR, gray_left.shape[::-1], criteria_stereo, flags))



########  get stereo rectification  ########
rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roiL, roiR = cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, gray_left.shape[::-1], rot, trans, rectifyScale, (0, 0))
stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, gray_left.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, gray_right.shape[::-1], cv2.CV_16SC2)

print("Saving stereo map")
cv_file = cv2.FileStorage('data/stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()
###############################


out = (cameraMatrixL, newCameraMatrixL, roiL, projMatrixL, distL, rectL,
       cameraMatrixR, newCameraMatrixR, roiR, projMatrixR, distR, rectR,
       rot, trans, essentialMatrix, fundamentalMatrix)

with open("data/data.pkl", "wb") as f:  # "wb" = write binary mode
    pickle.dump(out, f)