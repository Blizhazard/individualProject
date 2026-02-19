import numpy as np
import skimage
import napari
import pandas as pd

filename = "20250314_HMX_4523_ZZ_Inconel_625_300mic_2000x2000x2000x8bit.raw"
width, height, depth = 2000, 2000, 2000
dtype = np.uint8

slice_index = 1500  
bytes_per_voxel = np.dtype(dtype).itemsize
slice_size = width * height * bytes_per_voxel
offset = slice_index * slice_size



with open(filename, 'rb') as f:
    f.seek(offset)
    slice_data = np.fromfile(f, dtype=dtype, count=width*height*100)
img = slice_data.reshape((100, height, width)) 


viewer = napari.Viewer()
viewer.add_image(img, name='3D Volume', visible=False, scale=(3,3,3))
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'  
viewer.scale_bar.font_size = 40 
viewer.scale_bar.position = 'bottom_center'
napari.run()