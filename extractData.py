import numpy as np
import matplotlib.pyplot as plt
import cv2

filename = "20250314_HMX_4523_ZZ_316L_300mic_2000x2000x2000x8bit.raw"
width, height, depth = 2000, 2000, 2000
dtype = np.uint8

slice_index = 1500  # choose any slice (0–1999)
bytes_per_voxel = np.dtype(dtype).itemsize
slice_size = width * height * bytes_per_voxel
offset = slice_index * slice_size

# open file and read only one slice
with open(filename, 'rb') as f:
    f.seek(offset)
    slice_data = np.fromfile(f, dtype=dtype, count=width*height)

slice_img = slice_data.reshape((height, width))

thresh_global = cv2.threshold(slice_img, 100, 255, cv2.THRESH_BINARY)[1]
thresh_otsu = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
thresh_adaptive_mean = cv2.adaptiveThreshold(slice_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 10)
thresh_adaptive_gauss = cv2.adaptiveThreshold(slice_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

titles = ['Original', 'Global (t=100)', 'Otsu', 'Adaptive Mean', 'Adaptive Gaussian']
images = [slice_img, thresh_global, thresh_otsu, thresh_adaptive_mean, thresh_adaptive_gauss]

plt.figure(figsize=(14, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap = 'gray')
    plt.title(titles[i])
    plt.axis('off')
    cv2.imwrite(titles[i] + ".jpg",images[i])
plt.tight_layout()
plt.show()
