import numpy as np
import tensorflow as tf
from keras import layers as tfkl

# Load the dataset
data = np.load('training_set.npz')
images = data['images']
labels = data['labels']

# Define advanced augmentation pipeline for blood cell images
# Adjust scale_factor to only scale up
# Modifica il layer JitteredResize per mantenere le dimensioni originali
augmentation = tf.keras.Sequential([
    tfkl.RandomCrop(height=96, width=96),  # Adjust crop size if necessary
    tfkl.RandomFlip("horizontal_and_vertical"),
    tfkl.RandomRotation(0.3),
    tfkl.Dropout(0.1),
    tfkl.Dropout(0.2),
    tfkl.RandomContrast(0.3),
    tfkl.RandomZoom(0.15),
    tfkl.RandomBrightness(0.6),
], name='advanced_preprocessing')

# Apply augmentations
augmented_images = []
augmented_labels = []

for image, label in zip(images, labels):
    # Add batch dimension to image
    image = tf.expand_dims(image, axis=0)
    # Apply augmentation
    augmented_image = augmentation(image)
    # Remove batch dimension and convert to numpy array
    augmented_images.append(tf.squeeze(augmented_image, axis=0).numpy())
    augmented_labels.append(label)  # No need to augment labels

# Convert lists to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Save augmented dataset
np.savez('augmented_set4.npz', images=augmented_images, labels=augmented_labels)
