from matplotlib import contour
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread(r"D:\Uni\Main_Repo\individualProject\Original.jpg")
edgeDetected = cv2.Canny(image,threshold1=150,threshold2=300, apertureSize=3)
# contours, hierachy = cv2.findContours(edgeDetected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# filteredContours = tuple(shape for shape in contours if cv2.contourArea(shape) > 2)
# for i in range(300,310):
#    print(cv2.contourArea(filteredContours[i]))
# cv2.drawContours(image, filteredContours, -1, (0,200,200), 2)
# cv2.imshow('Contours', edgeDetected)
# cv2.waitKey(0)


plt.imshow(edgeDetected, cmap='gray')
plt.show()