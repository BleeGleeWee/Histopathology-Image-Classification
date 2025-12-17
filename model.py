import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(96, 96, 3)):
    """
    Build a CNN model using ResNet50 backbone.
    """
    base_model = tf.keras.applications.ResNet50(
        weights=None,      # Set to "imagenet" for pretrained weights
        include_top=False,
        input_shape=input_shape
    )

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model