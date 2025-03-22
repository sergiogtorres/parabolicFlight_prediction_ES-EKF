import cv2
def check_reprojection_function(img_left, img_right, chessboard_size,
                                termination_criteria, objp, cameraMatrixL, distL,
                                cameraMatrixR, distR, rot, trans):
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