import cv2
import os


path_left = "data/images/calibrationLeft/"
path_right = "data/images/calibrationRight/"

paths = [path_left, path_right]
for path in paths:
    if not os.path.exists(path):
       os.makedirs(path)

cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)

img_idx = 0
while (cap_left.isOpened() and cap_right.isOpened()):

    succes_left, img_left = cap_left.read()
    succes_right, img_right = cap_right.read()

    k = cv2.waitKey(5)

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

    cv2.imshow('Img L', img_left)
    cv2.imshow('Img R', img_right)

# Terminate
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()
