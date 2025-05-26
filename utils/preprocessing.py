import os
import numpy as np
import rasterio


def load_tif_images(directory, max_images=None):
    """Load `.tif` images from a directory into arrays."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    images, file_paths = [], []
    for i, filename in enumerate(os.listdir(directory)):
        if max_images and i >= max_images:
            break
        if filename.endswith(".tif"):
            path = os.path.join(directory, filename)
            try:
                with rasterio.open(path) as src:
                    images.append(src.read())
                    file_paths.append(path)
            except rasterio.RasterioIOError:
                print(f"Failed to read {filename}")
    return images, file_paths


def flatten_bands(image):
    """Flatten 2D (H,W) or 3D (C,H,W) images into (pixels, features)."""
    if image.ndim == 2:
        return image.reshape(-1, 1)  # (H*W, 1)
    elif image.ndim == 3:
        return image.reshape(image.shape[0], -1).T  # (H*W, C)
    else:
        raise ValueError("Input must be 2D (H,W) or 3D (C,H,W)")


def scale_features(features):
    """Normalize features using StandardScaler."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required. Install with `pip install scikit-learn`")
    return StandardScaler().fit_transform(features)
