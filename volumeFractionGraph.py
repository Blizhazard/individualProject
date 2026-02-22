import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('volume.csv')

col1, col2, col3 = df.columns

data1 = pd.to_numeric(df[col1], errors='coerce').dropna().values
data2 = pd.to_numeric(df[col2], errors='coerce').dropna().values
data3 = pd.to_numeric(df[col3], errors='coerce').dropna().values

all_data = np.concatenate([data1, data2, data3])
bin_width = 5
min_edge = np.floor(all_data.min() / bin_width) * bin_width
max_edge = np.ceil(all_data.max() / bin_width) * bin_width
bins = np.arange(min_edge, max_edge + bin_width, bin_width)

def volume_fraction(data, bins):
    bin_indices = np.digitize(data, bins) - 1
    vol_per_bin = np.zeros(len(bins) - 1)

    for i, val in zip(bin_indices, data):
        if 0 <= i < len(vol_per_bin):
            vol_per_bin[i] += val

    total_volume = np.sum(data)
    return vol_per_bin / total_volume

vf1 = volume_fraction(data1, bins)
vf2 = volume_fraction(data2, bins)
vf3 = volume_fraction(data3, bins)
bin_centers = bins[:-1] + bin_width / 2

plt.figure(figsize=(8, 6))

plt.bar(bin_centers, vf1, width=bin_width, alpha=0.5, label=col1)
plt.bar(bin_centers, vf2, width=bin_width, alpha=0.5, label=col2)
plt.bar(bin_centers, vf3, width=bin_width, alpha=0.5, label=col3)

# counts1, _, patches1 = plt.hist(data1, bins=bins, alpha=0.5, label=col1)
# counts2, _, patches2 = plt.hist(data2, bins=bins, alpha=0.5, label=col2)
# counts3, _, patches3 = plt.hist(data3, bins=bins, alpha=0.5, label=col3)

# def add_labels(counts, patches):
#     for count, patch in zip(counts, patches):
#         if count > 0:
#             x = patch.get_x() + patch.get_width() / 2
#             y = patch.get_height()
#             plt.text(x, y, int(count), ha='center', va='bottom', fontsize=8)


# add_labels(counts1, patches1)
# add_labels(counts2, patches2)
# add_labels(counts3, patches3)


plt.xticks(bins, rotation=90)
plt.xlabel('cubic root volume (um)')
plt.ylabel('Volume Fraction')
plt.title('Overlapping Volume Fraction Histograms of cubic root volumes')
plt.legend()

plt.tight_layout()
plt.show()