import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def visualize_batch(x_batch, y_batch, output_batch, **kwargs):
    """Converts x_batch and y_batch to numpy arrays and plots them in a grid"""
    x_batch = x_batch.numpy()
    y_batch = y_batch.numpy()
    if not isinstance(output_batch, np.ndarray):
        output_batch = output_batch.numpy()
    fig, axes = plt.subplots(3, 4, figsize=(10, 5))
    for i in range(x_batch.shape[0]):
        print(x_batch[i], y_batch[i])
        axes[0, i].imshow(x_batch[i].astype("float32"))
        axes[1, i].imshow(y_batch[i].astype("float32"))
        axes[2, i].imshow(output_batch[i].astype("float32"))
    plt.show()
