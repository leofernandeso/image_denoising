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
        image_shape: Tuple,
        batch_norm_kwargs: Optional[dict] = None,
    ):
        super(DnCNN, self).__init__()
        self._depth = depth
        self._image_shape = image_shape
        self._image_channels = self._image_shape[-1]

        self._batch_norm_kwargs = batch_norm_kwargs if batch_norm_kwargs else {}
        self._build()

    def _build(self) -> None:
        self._layers = [ConvRELUBlock()]

        for _ in range(self._depth - 2):
            self._layers.append(ConvRELUBlock(**self._batch_norm_kwargs))

        final_conv_layer = tf.keras.layers.Conv2D(
            filters=self._image_channels,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            use_bias=False,
        )
        self._layers.append(final_conv_layer)
        self.build((None, *self._image_shape))

    def _forward(self, X) -> tf.Tensor:
        out = tf.identity(X)  # keeping the input alive so we can subtract it later
        for layer in self._layers:
            out = layer(out)
        return out

    def summary(self):
        # Workaround: tensorflow does not print the correct tensor shapes when using model.build(). Therefore, a dummy Model() is instantiated.
        x = tf.keras.Input(shape=self._image_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()

    def call(self, X):
        return X - self._forward(X)
