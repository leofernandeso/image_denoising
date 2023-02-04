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


class DnCNN(tf.keras.Model):
    def __init__(
        self,
        depth: int,
        image_channels: Optional[int] = 1,
        batch_norm_kwargs: Optional[dict] = None,
    ):
        super(DnCNN, self).__init__()
        self._depth = depth
        self._image_channels = image_channels
        self._batch_norm_kwargs = batch_norm_kwargs if batch_norm_kwargs else {}
        self._construct()

    def _construct(self) -> None:
        self._model = tf.keras.Sequential()

        self._model.add(ConvRELUBlock())

        for _ in range(self._depth - 2):
            self._model.add(ConvRELUBlock(**self._batch_norm_kwargs))

        self.final_conv_layer = tf.keras.layers.Conv2D(
            filters=self._image_channels,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            use_bias=False,
        )
        self._model.add(self.final_conv_layer)

    def call(self, X):
        out = self._model(X)
        return X - out


if __name__ == "__main__":
    model = DnCNN(depth=64, image_channels=3)
