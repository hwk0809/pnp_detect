import cv2

img = cv2.imread("/home/hwk/program/2021-1-competition_1/image_static/image_static1611648462.648835.png")
def getImgWidthAndHeight(img):
    sp = img.shape
    return sp[0],sp[1]


[height,width,pixels] = img.shape
print (height/2)
print (width/2)