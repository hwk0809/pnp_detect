# coding: UTF-8

import cv2

img = cv2.imread('/home/hwk/下载/1.jpg')

a = []
b = []


def get_button_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_button_point)
cv2.imshow("image", img)
cv2.waitKey(0)
print(a[0], b[0])
