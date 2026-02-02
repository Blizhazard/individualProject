import numpy as np
import skimage
import napari
import scipy.ndimage as ndi
import pandas as pd

img = skimage.io.imread("output_volume.tiff")
img = skimage.morphology.remove_small_holes(img, area_threshold=10, connectivity=1)
ball = skimage.morphology.ball(1.8)
morphedLabel = skimage.morphology.dilation(img, ball)
labels = skimage.measure.label(morphedLabel == 0, connectivity=1)
largeObjects = skimage.morphology.remove_small_objects(labels, min_size=4000000, connectivity=1)
labels = labels ^ largeObjects

filteredLabels = skimage.morphology.remove_small_objects(labels, min_size=100)
props = skimage.measure.regionprops_table(filteredLabels, properties=['label', 'area_filled', 'axis_major_length', 'axis_minor_length', 'feret_diameter_max'])
df = pd.DataFrame(props)
df.to_csv("3d_analysis_results.csv", index=False)
breakpoint()
# transformed = ndi.distance_transform_edt(labels)
# maxima = skimage.morphology.local_maxima(transformed)
viewer = napari.Viewer()
viewer.add_image(img, name='3D Volume', visible=False)
viewer.add_labels(filteredLabels, name='Segmented Labels', visible=False)
viewer.add_labels(morphedLabel, name='Morphed Labels')
# viewer.add_points(np.transpose(np.nonzero(maxima)), name='Local Maxima')
napari.run()