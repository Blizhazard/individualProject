from matplotlib import contour
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('Adaptive Gaussian.jpg')
edgeDetected = cv2.Canny(image,100,200)
contours, hierachy = cv2.findContours(edgeDetected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filteredContours = tuple(shape for shape in contours if cv2.contourArea(shape) > 2)
for i in range(300,310):
    print(cv2.contourArea(filteredContours[i]))
cv2.drawContours(image, filteredContours, -1, (0,200,200), 2)
cv2.imshow('Contours', image)
cv2.waitKey(0)