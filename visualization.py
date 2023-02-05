import tensorflow as tf
from matplotlib import pyplot as plt


def visualize_batch(x_batch, y_batch):
    """Converts x_batch and y_batch to numpy arrays and plots them in a grid"""
    x_batch = x_batch.numpy()
    y_batch = y_batch.numpy()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(x_batch.shape[0]):
        axes[0, i].imshow(x_batch[i].astype("float32"))
        axes[1, i].imshow(y_batch[i].astype("float32"))
    plt.show()

