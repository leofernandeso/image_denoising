import random
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional

from noise_models import NoiseAdderCallable


class ImageDenoisingDataGenerator:
    def __init__(
        self,
        base_folder: str,
        noise_adder_callable: NoiseAdderCallable,
        resize_shape: Optional[Tuple[int, int]] = None,
    ):
        self._base_folder = base_folder
        self._noise_adder_callable = noise_adder_callable
        self._image_filepaths = list(Path(self._base_folder).iterdir())
        self._length = len(self._image_filepaths)
        self._resize_shape = resize_shape

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.keras.preprocessing.image.load_img(self._image_filepaths[idx])
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        if self._resize_shape:
            image_arr = tf.image.resize(image_arr, self._resize_shape)
        x = image_arr / 255.0  # assuming only 8-bit images
        y = self._noise_adder_callable(x)
        return x, y

    def __call__(self):
        for i in range(self._length):
            yield self.__getitem__(i)

            if i == self._length - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self._image_filepaths)
