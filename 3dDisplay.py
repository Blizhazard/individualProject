import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

filename = r"individualProject\20250314_HMX_4523_ZZ_316L_300mic_2000x2000x2000x8bit.raw"
#filename = "CylinderCropped.raw"
width, height, depth = 2000, 2000, 2000
dtype = np.uint8

slice_index = 1500  # choose any slice (0–1999)
bytes_per_voxel = np.dtype(dtype).itemsize
slice_size = width * height * bytes_per_voxel
offset = slice_index * slice_size


with open(filename, 'rb') as f:
    f.seek(offset)
    slice_data = np.fromfile(f, dtype=dtype, count=width*height*100)
slice_images = slice_data.reshape((100, height, width)) 


k = np.ones((3,3),np.uint8)

for i in range(100):
    cropMask = cv2.threshold(slice_images[i], 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    slice_images[i] = cv2.adaptiveThreshold(slice_images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)  
    #slice_images[i] = cv2.bitwise_not(slice_images[i])
    
    cropMask = cv2.dilate(cropMask, k, iterations=20)
    cropMask = cv2.erode(cropMask, k, iterations=20)
    outputImg = cv2.bitwise_and(cropMask, slice_images[i])
    slice_images[i] = outputImg
    # if i == 1:
    #     plt.imshow(outputImg)
    #     plt.show()
    #     plt.imshow(cropMask)
    #     plt.show()


plt.imshow(slice_images[1], cmap='gray')
plt.imshow(slice_images[99], cmap='gray')
plt.show()

imageio.volsave('output_volume.tiff', slice_images)
