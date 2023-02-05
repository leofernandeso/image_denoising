import tensorflow as tf

from typing import Callable

NoiseAdderCallable = Callable[[tf.Tensor], tf.Tensor]


def add_gaussian_noise(
    tensor: tf.Tensor, sigma: float, normalize: float = 1 / 255.0
) -> tf.Tensor:
    """
    Adds gaussian noise to tensor and returns the result
    """
    noise = tf.random.normal(tensor.shape, stddev=sigma * normalize)
    return tensor + noise
