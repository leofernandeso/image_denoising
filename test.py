import tensorflow as tf

model = tf.keras.models.load_model("saved_models/dn_cnn")

import tensorflow as tf
from functools import partial

from dataset import ImageDenoisingDataGenerator
from noise_models import add_gaussian_noise
from visualization import visualize_batch


params = {
    "data": {
        "image_shape": (128, 128, 3),
    },
    "training": {
        "epochs": 400,
        "batch_size": 4,
        "visualize": True,
        "visualization_frequency_in_epochs": 50,
        "loss_logging_frequency": 5,
    },
}

IMAGE_SIZE = params["data"]["image_shape"][:2]
NCHANNELS = params["data"]["image_shape"][-1]
LOSS_LOGGING_FREQUENCY = params["training"]["loss_logging_frequency"]

if __name__ == "__main__":
    # Loading model
    model = tf.keras.models.load_model("saved_models/dn_cnn")

    add_gaussian_noise_ = partial(add_gaussian_noise, sigma=25.0)
    data_generator = ImageDenoisingDataGenerator(
        base_folder="data/train",
        noise_adder_callable=add_gaussian_noise_,
        resize_shape=IMAGE_SIZE,
    )
    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=params["data"]["image_shape"], dtype=tf.float32),
            tf.TensorSpec(shape=params["data"]["image_shape"], dtype=tf.float32),
        ),
    ).batch(8)

    for xs, ys in train_dataset:
        output = model.predict(xs)
        visualize_batch(xs, ys, output)
