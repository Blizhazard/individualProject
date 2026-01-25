import numpy as np
import skimage
import napari
import scipy.ndimage as ndi

img = skimage.io.imread("output_volume.tiff")
img = skimage.morphology.remove_small_holes(img, area_threshold=10, connectivity=1)
ball = skimage.morphology.ball(1.8)
morphedLabel = skimage.morphology.dilation(img, ball)
labels = skimage.measure.label(morphedLabel == 0, connectivity=1)
largeObjects = skimage.morphology.remove_small_objects(labels, min_size=4000000, connectivity=1)
labels = labels ^ largeObjects
# transformed = ndi.distance_transform_edt(labels)
# maxima = skimage.morphology.local_maxima(transformed)
viewer = napari.Viewer()
viewer.add_image(img, name='3D Volume', visible=False)
viewer.add_labels(labels, name='Segmented Labels', visible=False)
viewer.add_labels(morphedLabel, name='Morphed Labels')
# viewer.add_points(np.transpose(np.nonzero(maxima)), name='Local Maxima')
napari.run()