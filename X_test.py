from simple_multi_unet_model import multi_unet_model
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# n_classes = 4

# def get_model():
#     return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=256, IMG_WIDTH=512, IMG_CHANNELS=1)

# model = get_model()
# model.load_weights('/bess25/jskim/semantic_segmentation/U-net_colab/230705_cityscape_all.hdf5')

source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230705source/'
X_test = np.load(source_path + 'X_test.npy')
y_test = np.load(source_path + 'y_test.npy')
with open('names_test.pkl', 'rb') as f:
    names_test = pickle.load(f)

print(X_test.min(), X_test.max())

source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230706source/'
X_test_rgb = np.load(source_path + 'X_test.npy')
y_test = np.load(source_path + 'y_test.npy')
with open('names_test.pkl', 'rb') as f:
    names_test = pickle.load(f)

print(X_test_rgb.min(), X_test_rgb.max())
print(X_test_rgb)
