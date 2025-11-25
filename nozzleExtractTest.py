'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

filename = 
width, height, depth = 1350, 1438, 4300
dtype = np.float32

slice_index = 1500  # choose any slice (0–1999)
bytes_per_voxel = np.dtype(dtype).itemsize
slice_size = width * height * bytes_per_voxel
offset = slice_index * slice_size

# open file and read only one slice
with open(filename, 'rb') as f:
    f.seek(offset)
    slice_data = np.fromfile(f, dtype=dtype, count=width*height)

slice_img = slice_data.reshape((height, width))
plt.figure(figsize=(14, 6))
plt.imshow(slice_img, cmap = 'gray')
plt.show()
'''
import numpy as np
import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cv2

width  = 1350
height = 1458
slices = 4300
dtype  = np.float32

def to_uint8(img):
    imin = np.nanmin(img)
    imax = np.nanmax(img)
    if imax == imin:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - imin) / (imax - imin)
    return (norm * 255).astype(np.uint8)

volume = np.memmap(
    r'D:\Uni\1st scan\20251118_HMX_4936_JP_Sample_1_CROPPED_1350x1438x4300_32bit',
    dtype='<f4',
    mode='r',
    shape=(slices, height, width)
)
os.makedirs("axial", exist_ok=True)



for i in range(2000, slices-2000):
    arr = volume[i]
    cv2.imshow("slice", arr)
    cv2.waitKey(0)
    cv2.imwrite(f"axial/axial_{i:04d}.jpg", to_uint8(arr))
    image = cv2.imread(f"axial/axial_{i:04d}.jpg")
    cv2.imshow("", image)
    cv2.waitKey(0)

