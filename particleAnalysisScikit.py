import numpy as np
import skimage
import napari
import pandas as pd

img = skimage.io.imread("718_20mic_output_volume.tiff")
img = skimage.morphology.remove_small_holes(img, area_threshold=10, connectivity=1)
img = np.invert(img)
# ball = skimage.morphology.ball(2)
# morphedLabel = skimage.morphology.dilation(img, ball)
# morphedLabel = skimage.morphology.dilation(img, ball)
# morphedLabel = skimage.morphology.dilation(img, ball)
morphedLabel = skimage.morphology.isotropic_dilation(img, radius=4)

labels = skimage.measure.label(morphedLabel == 0, connectivity=1)

filteredLabels = skimage.morphology.remove_small_objects(labels, min_size=100)
props = skimage.measure.regionprops_table(filteredLabels, properties=['label', 'area_filled', 'area_convex' ,'axis_major_length', 'axis_minor_length', 'feret_diameter_max'])
df = pd.DataFrame(props)
df.to_csv("718_20mic_particle_analysis_results.csv", index=False)


viewer = napari.Viewer()
viewer.add_image(img, name='3D Volume', visible=False, scale=(3,3,3))
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'  
viewer.scale_bar.font_size = 40 
viewer.scale_bar.position = 'bottom_center'
viewer.add_labels(labels, name='Segmented Labels', visible=False, scale=(3,3,3))
viewer.add_labels(morphedLabel, name='Morphed Labels', scale=(3,3,3), visible=False)
napari.run()