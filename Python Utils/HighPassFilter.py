import numpy as np
import cv2
import os
from scipy.fft import fft2, fftshift, ifft2

# Load the dataset
input_file = 'augmented_set3.npz'
data = np.load(input_file)
images = data['images']
labels = data['labels']

# Directory to save filtered images
filtered_dir = '/home/zfirefist/Desktop/ANN-DL_2024_PoliMi/filtered_images'
os.makedirs(filtered_dir, exist_ok=True)

# Function to create a high-pass filter based on adaptive threshold
def adaptive_high_pass_filter(f_transform, threshold_ratio=0.2):
    """
    Generates an adaptive high-pass filter mask based on the frequency spectrum.

    Parameters:
    - f_transform: The Fourier transform of the image.
    - threshold_ratio: Ratio of the average magnitude to set as the cutoff.

    Returns:
    - mask: High-pass filter mask.
    """
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform)
    
    # Compute adaptive threshold based on mean magnitude
    adaptive_threshold = magnitude_spectrum.mean() * threshold_ratio
    
    # Create a mask where frequencies below the threshold are set to 0
    mask = magnitude_spectrum > adaptive_threshold
    return mask

# Check if the filtered image is almost black
def is_almost_black(image, threshold_ratio=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_zero = cv2.countNonZero(gray)
    total_pixels = gray.size
    return (non_zero / total_pixels) < threshold_ratio

# Initialize variables
filtered_images = []
processed_indices = []
skipped_count = 0

# Process each color channel independently
filtered_images = []
processed_indices = []
skipped_count = 0

# Process each image
for idx, image in enumerate(images):
    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    try:
        # Prepare an empty list to store the filtered channels
        filtered_channels = []
        
        # Process each color channel separately
        for channel in cv2.split(image):
            # Fourier transform for the channel
            f_transform = fft2(channel)
            f_transform_shifted = fftshift(f_transform)
            
            # Apply adaptive high-pass filter
            adaptive_filter = adaptive_high_pass_filter(f_transform_shifted, threshold_ratio=0.2)
            filtered_transform = f_transform_shifted * adaptive_filter
            
            # Convert back to spatial domain
            filtered_channel = np.abs(ifft2(np.fft.ifftshift(filtered_transform)))
            filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Append the filtered channel
            filtered_channels.append(filtered_channel)
        
        # Merge channels back into a color image
        filtered_image = cv2.merge(filtered_channels)

        # Check for meaningful content
        if is_almost_black(filtered_image):
            skipped_count += 1
            print(f"Skipped image {idx}: no significant features.")
            continue

        # Save filtered image
        filtered_images.append(filtered_image)
        processed_indices.append(idx)

        filtered_path = os.path.join(filtered_dir, f"image_{idx}_filtered.png")
        cv2.imwrite(filtered_path, filtered_image)

        if idx % 100 == 0:
            print(f"Processed {idx} images...")
    except Exception as e:
        print(f"Error processing image index {idx}: {e}")


# Save filtered dataset
filtered_images = np.array(filtered_images)

if filtered_images.size > 0:
    filtered_labels = labels[processed_indices]
    np.savez(
        'filtered_dataset.npz',
        images=filtered_images,
        labels=filtered_labels
    )
    print(f"Filtered dataset saved as 'filtered_dataset.npz'")
    print(f"Filtered images saved in '{filtered_dir}'")
    print(f"Total images processed: {len(images)}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Total images filtered: {len(filtered_images)}")
else:
    print("No filtered images to save.")
