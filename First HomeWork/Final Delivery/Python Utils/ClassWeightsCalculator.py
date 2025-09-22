import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Load the .npz dataset
data = np.load("removed_dataset.npz")

# Assuming the labels are stored in 'labels'
labels = data['labels'][:11959]  # Replace 'labels' with the actual key for the labels in your file

# Get unique classes and their counts
unique_classes, class_counts = np.unique(labels, return_counts=True)

# Compute class weights (inverse proportional to class frequency)
class_weights = {cls: 1.0 / count for cls, count in zip(unique_classes, class_counts)}

# Optionally normalize weights to ensure the sum of weights is constant
total_weight = sum(class_weights.values())
class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

print("Class Weights:", class_weights)

