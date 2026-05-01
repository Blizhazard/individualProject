import numpy as np
import skimage
import napari
import pandas as pd
import dask.array as da

filename = r"E:\HT1762_B\20260213_HMX_4936_JP_HT1762_B_ALL_2000x2000x5748x32bit"
width, height, depth = 2000, 2000, 5748
# dtype = np.uint8
dtype = np.float32

slice_index = 4748  
bytes_per_voxel = np.dtype(dtype).itemsize
slice_size = width * height * bytes_per_voxel
offset = slice_index * slice_size



with open(filename, 'rb') as f:
    f.seek(offset)
    slice_data = np.fromfile(f, dtype=dtype, count=width*height*500)
img = slice_data.reshape((500, height, width)) 


viewer = napari.Viewer()
viewer.add_image(img, name='3D Volume', scale=(3,3,3))
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'  
viewer.scale_bar.font_size = 40 
viewer.scale_bar.position = 'bottom_center'
napari.run()