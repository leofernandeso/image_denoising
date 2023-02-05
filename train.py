import tensorflow as tf
from functools import partial

from models.dn_cnn import DnCNN
from dataset import ImageDenoisingDataGenerator
from noise_models import add_gaussian_noise
from visualization import visualize_batch


params = {
    "data": {
        "image_shape": (512, 512, 3),
    },
    "training": {"epochs": 10, "batch_size": 4, "visualize": True},
}

IMAGE_SIZE = params["data"]["image_shape"][:2]
NCHANNELS = params["data"]["image_shape"][-1]

if __name__ == "__main__":
    # Initializing model
    model = DnCNN(depth=64, image_shape=params["data"]["image_shape"])

    # Creating train dataset
    add_gaussian_noise_ = partial(add_gaussian_noise, sigma=25.0)
    data_generator = ImageDenoisingDataGenerator(
        base_folder="data",
        noise_adder_callable=add_gaussian_noise_,
        resize_shape=IMAGE_SIZE,
    )
    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=params["data"]["image_shape"], dtype=tf.float16),
            tf.TensorSpec(shape=params["data"]["image_shape"], dtype=tf.float16),
        ),
    )

    for epoch in range(params["training"]["epochs"]):
        print(f"Starting epoch {epoch+1}")
        for step, (x_batch, y_batch) in enumerate(
            train_dataset.batch(params["training"]["batch_size"])
        ):
            if params["training"]["visualize"]:
                visualize_batch(x_batch, y_batch)
            print(step, x_batch.shape, y_batch.shape)
