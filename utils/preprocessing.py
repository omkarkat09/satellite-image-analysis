# utils/preprocessing.py

import os
import numpy as np
import rasterio

def load_tif_images(directory):
    """Load all .tif images from a directory into a list of arrays."""
    images = []
    file_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            path = os.path.join(directory, filename)
            with rasterio.open(path) as src:
                image = src.read()  # shape: (bands, height, width)
                images.append(image)
                file_paths.append(path)
    return images, file_paths

def flatten_bands(image):
    """
    Flattens a multi-band image into 2D: (pixels, bands).
    Input: image.shape = (bands, height, width)
    Output: (num_pixels, bands)
    """
    bands, height, width = image.shape
    image_reshaped = image.reshape((bands, -1)).T
    return image_reshaped  # shape: (height*width, bands)

def scale_features(features):
    """Normalize the features across all bands."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features)
