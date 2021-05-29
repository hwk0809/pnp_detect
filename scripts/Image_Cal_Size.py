# coding: UTF-8

import cv2

img = cv2.imread('/home/hwk/下载/1.jpg')


def getImgWidthAndHeight(img):
    sp = img.shape
    return sp[0], sp[1]


[height, width, pixels] = img.shape
print (height)
print (width)
