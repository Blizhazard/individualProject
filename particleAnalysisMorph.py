from matplotlib import contour, markers
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

image = cv2.imread('Adaptive Gaussian.jpg')
morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE , np.ones((2,2),np.uint8))
labeledImage = label(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY))
#labeledImage = label2rgb(labeledImage,image=morphed, bg_label=0)

'''
objects = regionprops(labeledImage)
print(f"Number of objects detected: {len(objects)}")
plt.figure(figsize=(1,1))
bigParticle = [i for i in objects if i.area > 100]
plt.imshow(bigParticle[200].image)
plt.show()

breakpoint()
'''


plt.figure(figsize=(8,8))
plt.subplot(231)
plt.imshow(labeledImage)
plt.subplot(232)
plt.imshow(morphed)
plt.subplot(233)
plt.imshow(image)



plt.subplot(234)
distTrans = cv2.distanceTransform(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY), cv2.DIST_L2,5) 
plt.imshow(distTrans)

'''
plt.subplot(234)
distance = ndi.distance_transform_edt(morphed)
breakpoint()
coords = peak_local_max(distance, min_distance=1,labels=morphed)
masks = np.zeros(distance.shape, dtype=bool)
masks[tuple(coords.T)] = True

markers, _ = ndi.label(masks)
labelsWS = watershed(-distance, markers, mask=morphed)
plt.imshow(labelsWS)
#plt.subplot(235)
'''

plt.show()