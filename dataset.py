import random
import tensorflow as tf
from pathlib import Path
from typing import Tuple

from noise_models import NoiseAdderCallable


class ImageDenoisingDataGenerator:
    def __init__(
        self,
        base_folder: str,
        noise_adder_callable: NoiseAdderCallable,
        augmentation_pipeline=None,
        color_mode="grayscale",
    ):
        self._base_folder = base_folder
        self._noise_adder_callable = noise_adder_callable
        self._image_filepaths = list(Path(self._base_folder).iterdir())
        self._length = len(self._image_filepaths)
        self._augmentation_pipeline = augmentation_pipeline
        self._color_mode = color_mode

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.keras.preprocessing.image.load_img(self._image_filepaths[idx], color_mode=self._color_mode)
        image = tf.keras.preprocessing.image.img_to_array(image)
        if self._augmentation_pipeline:
            image = self._augmentation_pipeline(image=image)
            image = image["image"]
        y = image / 255.0
        x = tf.clip_by_value(self._noise_adder_callable(y), 0.0, 1.0)
        return x, y

    def __call__(self):
        for i in range(self._length):
            yield self.__getitem__(i)

            if i == self._length - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self._image_filepaths)
