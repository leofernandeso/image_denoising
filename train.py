import tensorflow as tf
import albumentations as A
from functools import partial

from models.dn_cnn import DnCNN
from dataset import ImageDenoisingDataGenerator
from noise_models import add_gaussian_noise
from visualization import visualize_batch


params = {
    "data": {
        "image_shape": (128, 128, 1),
    },
    "training": {
        "epochs": 200,
        "batch_size": 4,
        "visualize": True,
        "visualization_frequency_in_epochs": 50,
        "loss_logging_frequency": 5,
    },
}

IMAGE_SIZE = params["data"]["image_shape"][:2]
NCHANNELS = params["data"]["image_shape"][-1]
LOSS_LOGGING_FREQUENCY = params["training"]["loss_logging_frequency"]

AUGMENTATION_PIPELINE = A.Compose([
    A.RandomCrop(128, 128),
    A.HorizontalFlip(p=0.5)
])

if __name__ == "__main__":
    # Initializing model
    model = DnCNN(depth=17, image_channels=1)

    # Creating train dataset
    num_epochs = params["training"]["epochs"]
    batch_size = params["training"]["batch_size"]

    add_gaussian_noise_ = partial(add_gaussian_noise, sigma=15.0)
    data_generator = ImageDenoisingDataGenerator(
        base_folder="data/train",
        noise_adder_callable=add_gaussian_noise_,
        augmentation_pipeline=AUGMENTATION_PIPELINE
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
    callbacks = [
        # tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_models/checkpoint",
            save_weights_only=False,
            save_freq=10,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)
    model.save("saved_models/dn_cnn")
