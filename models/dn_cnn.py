from typing import Tuple, Optional

import tensorflow as tf


class ConvRELUBlock(tf.keras.Model):
    def __init__(
        self,
        filters: Optional[int] = 64,
        kernel_size: Optional[Tuple[int, int]] = (3, 3),
        batch_norm_kwargs: Optional[dict] = None,
    ):
        super(ConvRELUBlock, self).__init__()
        self._batch_norm_kwargs = batch_norm_kwargs
        self._kernel_initializer = tf.keras.initializers.Orthogonal()
        self._conv2d = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            kernel_initializer=self._kernel_initializer,
        )
        if self._batch_norm_kwargs:
            self._batch_norm = tf.keras.layers.BatchNormalization(
                **self._batch_norm_kwargs
            )

    def call(self, X):
        conv_result = tf.nn.relu(self._conv2d(X))
        if self._batch_norm_kwargs:
            return self._batch_norm(conv_result)
        return conv_result


def DnCNN(
    depth: int,
    image_channels: int = 3,
    batch_norm_kwargs: Optional[dict] = {},
) -> tf.keras.models.Model:
    inputs = tf.keras.Input(shape=(None, None, image_channels))
    x = ConvRELUBlock()(inputs)

    batch_norm_kwargs = {"axis": 3, "momentum": 0.0, "epsilon": 0.0001}
    for _ in range(depth - 2):
        x = ConvRELUBlock(batch_norm_kwargs=batch_norm_kwargs)(x)

    final_conv_layer = tf.keras.layers.Conv2D(
        filters=image_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=tf.keras.initializers.Orthogonal(),
        use_bias=False,
    )

    out = final_conv_layer(x)
    y = tf.keras.layers.subtract([out, inputs])
    return tf.keras.models.Model(inputs=inputs, outputs=y)
