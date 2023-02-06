import tensorflow as tf
from functools import partial

from models.dn_cnn import DnCNN
from dataset import ImageDenoisingDataGenerator
from noise_models import add_gaussian_noise


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


def lr_schedule(epoch, learning_rate):
    if epoch < 100:
        return learning_rate
    elif epoch % 50 == 0:
        return learning_rate / 10
    else:
        return learning_rate


if __name__ == "__main__":
    # Initializing model
    model = DnCNN(depth=17)

    # Creating train dataset
    num_epochs = params["training"]["epochs"]
    batch_size = params["training"]["batch_size"]

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
    ).batch(batch_size)

    # Optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Callbacks
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule)]

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)
    model.save("saved_models/dn_cnn")
