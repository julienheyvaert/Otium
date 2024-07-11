from cannyAlgorithm import *
import cv2
image = getImage(r"C:\Users\julienn\Pictures\mona\5.jpg")
cv2.imwrite('rendered/5.jpg', image)
cv2.imwrite('rendered/5_c.jpg', canny(image))