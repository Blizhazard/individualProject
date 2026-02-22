import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('poreFerretDiameter.csv')

col1, col2, col3 = df.columns

data1 = pd.to_numeric(df[col1], errors='coerce').dropna().values
data2 = pd.to_numeric(df[col2], errors='coerce').dropna().values
data3 = pd.to_numeric(df[col3], errors='coerce').dropna().values

all_data = np.concatenate([data1, data2, data3])
bin_width = 5
min_edge = np.floor(all_data.min() / bin_width) * bin_width
max_edge = np.ceil(all_data.max() / bin_width) * bin_width
bins = np.arange(min_edge, max_edge + bin_width, bin_width)


plt.figure(figsize=(8, 6))


counts1, _, patches1 = plt.hist(data1, bins=bins, alpha=0.5, label=col1)
counts2, _, patches2 = plt.hist(data2, bins=bins, alpha=0.5, label=col2)
counts3, _, patches3 = plt.hist(data3, bins=bins, alpha=0.5, label=col3)

def add_labels(counts, patches):
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            plt.text(x, y, int(count), ha='center', va='bottom', fontsize=8)


# add_labels(counts1, patches1)
# add_labels(counts2, patches2)
# add_labels(counts3, patches3)


plt.xticks(bins, rotation=90)
plt.xlabel('diameter (um)')
plt.ylabel('Frequency')
plt.title('Overlapping Histograms of Ferret Diameters of pores')
plt.legend()

plt.tight_layout()
plt.show()