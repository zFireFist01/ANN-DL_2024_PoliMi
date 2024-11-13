import numpy as np
import tensorflow as tf
import keras_cv

# Load the dataset
data = np.load('training_set.npz')
images = data['images']
labels = data['labels']




# Define advanced augmentation pipeline for blood cell images
# Adjust scale_factor to only scale up
# Modifica il layer JitteredResize per mantenere le dimensioni originali
data_augmentation = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.AutoContrast(value_range=(0, 255)),
        keras_cv.layers.AugMix(severity=0.5, value_range=(0, 255)),
        keras_cv.layers.ChannelShuffle(),
        keras_cv.layers.CutMix(),
        keras_cv.layers.FourierMix(),
        keras_cv.layers.GridMask(),
        # Rimuovi o modifica il layer di ridimensionamento
        # Se desideri mantenere le dimensioni, puoi rimuovere completamente questo layer
        # oppure impostare target_size a (96, 96)
        keras_cv.layers.JitteredResize(target_size=(96, 96), scale_factor=(1.0, 1.2)),
        keras_cv.layers.MixUp(),
        keras_cv.layers.RandAugment(magnitude=0.5, value_range=(0, 255)),
        keras_cv.layers.RandomAugmentationPipeline(
            layers=[
                keras_cv.layers.RandomChannelShift(value_range=(0, 255), factor=0.1),
                keras_cv.layers.RandomColorDegeneration(factor=0.5),
                keras_cv.layers.RandomCutout(height_factor=0.2, width_factor=0.2),
                keras_cv.layers.RandomHue(factor=0.2, value_range=(0, 255)),
                keras_cv.layers.RandomSaturation(factor=0.2),
                keras_cv.layers.RandomSharpness(factor=0.2, value_range=(0, 255)),
                keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2),
                keras_cv.layers.Solarization(value_range=(0, 255))
            ],
            augmentations_per_image=4
        ),
    ]
)


# Apply augmentations
augmented_images = []
augmented_labels = []

for image, label in zip(images, labels):
    # Prepare inputs as a dictionary for CutMix compatibility
    input_data = {"images": tf.expand_dims(image, axis=0), "labels": tf.expand_dims(label, axis=0)}
    augmented_data = data_augmentation(input_data)
    augmented_images.append(tf.squeeze(augmented_data["images"], axis=0).numpy())
    augmented_labels.append(tf.squeeze(augmented_data["labels"], axis=0).numpy())

# Convert lists to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Save augmented dataset
np.savez('augmented_set3.npz', images=augmented_images, labels=augmented_labels)
