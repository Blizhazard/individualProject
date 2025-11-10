from sqlite3 import connect
from matplotlib import contour, markers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import filters, morphology, segmentation, measure, feature
import random

image = cv2.imread('Adaptive Gaussian edited.png')
image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
morphed = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE , np.ones((2,2),np.uint8))
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN , np.ones((2,2),np.uint8))
morphed = cv2.threshold(morphed, 100, 255, cv2.THRESH_BINARY)[1]
labeledImage, count = label(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY), connectivity=2, return_num=True)
print(count)
objects = regionprops(labeledImage)
objects = [obj for obj in objects if obj.area > 100]
for i in range(5):
    pick = random.randint(0,len(objects)-1)
    print(objects[pick].area)
    print(objects[pick])
    plt.imshow(objects[pick].image.astype(np.uint8)*255)
    plt.show()
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


plt.figure(figsize=(9,9))
plt.subplot(231)
plt.imshow(labeledImage)
plt.subplot(232)
plt.imshow(morphed)
plt.subplot(233)
plt.imshow(image)


'''
plt.subplot(234)
distTrans = cv2.distanceTransform(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY), cv2.DIST_L2,0) 
plt.imshow(distTrans)
plt.subplot(235)
distThresh = np.uint8(distTrans)
_, labels = cv2.connectedComponents(distThresh)

labels = np.int32(labels)
labels = cv2.watershed(morphed, labels) 
plt.imshow(labels)
'''

'''
plt.subplot(234)
distance = ndi.distance_transform_edt(morphed)
coords = peak_local_max(distance, min_distance=1,labels=morphed)
masks = np.zeros(distance.shape, dtype=bool)
masks[tuple(coords.T)] = True

markers, _ = ndi.label(masks)
labelsWS = watershed(-distance, markers, mask=morphed)
plt.imshow(labelsWS)
'''


'''
img = skimage.io.imread('Adaptive Gaussian edited.png')
binary = img > 0.7
dist = ndi.distance_transform_edt(binary)

coords = feature.peak_local_max(dist, min_distance=1, labels=binary)
mark = np.zeros(dist.shape, dtype=int)
mark[tuple(coords.T)] = np.arange(1,len(coords)+1)

labels = segmentation.watershed(-dist, mark, mask=binary)

plt.subplot(234)
plt.imshow(labels)
'''

plt.subplot(234)
sureBG = cv2.dilate(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
plt.imshow(sureBG)
plt.subplot(235)
distTrans = cv2.distanceTransform(cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY), cv2.DIST_L2,3)
plt.imshow(distTrans)
plt.subplot(236)
_, sureFG = cv2.threshold(distTrans, 0.15*distTrans.max(), 255, cv2.THRESH_BINARY)   
sureFG = np.uint8(sureFG)
plt.imshow(sureFG)
plt.subplot(337)
unknown = cv2.subtract(sureBG,sureFG)
plt.imshow(unknown)





plt.show()