import numpy as np
import cv2
import os

def segment_nucleus(image):
    """
    Segments nuclei in the given image.

    Parameters:
    - image (numpy.ndarray): Input image array (H, W, C) in BGR format.

    Returns:
    - segmented_only (numpy.ndarray): Image containing only the segmented nuclei.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply stronger noise reduction with median filter
    denoised = cv2.medianBlur(gray, 5)

    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Use morphological operations to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for segmented nuclei
    mask = np.zeros_like(gray)
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjust area filter based on expected nucleus size
        if 200 < area < 5000:  # These thresholds can be tuned
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image to extract only the nuclei
    segmented_only = cv2.bitwise_and(image, image, mask=mask)

    return segmented_only

# Load the augmented dataset
augmented_data = np.load('augmented_set3.npz')
augmented_images = augmented_data['images']  # Assuming shape (N, H, W, C)
augmented_labels = augmented_data['labels']

# Initialize list to store segmented images
segmented_images = []
processed_indices = []  # To keep track of which images were processed

# Directory to save segmented images
segmented_dir = '/home/zfirefist/Desktop/ANN-DL_2024_PoliMi/segmented_images'
os.makedirs(segmented_dir, exist_ok=True)

# Threshold for determining if an image is "almost black"
# For example, less than 1% of pixels are non-zero
def is_almost_black(segmented_image, threshold_ratio=0.01):
    """
    Determines if the segmented image is almost black based on the ratio of non-zero pixels.

    Parameters:
    - segmented_image (numpy.ndarray): The segmented image.
    - threshold_ratio (float): The ratio below which the image is considered almost black.

    Returns:
    - bool: True if the image is almost black, False otherwise.
    """
    # Convert to grayscale to count non-zero pixels
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    non_zero = cv2.countNonZero(gray)
    total_pixels = gray.size
    ratio = non_zero / total_pixels
    return ratio < threshold_ratio

# Iterate over each image and apply segmentation
skipped_count = 0
for idx, image in enumerate(augmented_images):
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # If image has a single channel, convert to BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        segmented = segment_nucleus(image)

        # Check if the segmented image is almost black
        if is_almost_black(segmented):
            skipped_count += 1
            print(f"Skipped image {idx} as it contains no detectable nuclei.")
            continue  # Skip saving and appending

        # If not skipped, append to the list and save
        segmented_images.append(segmented)
        processed_indices.append(idx)

        # Save segmented image to disk
        segmented_path = os.path.join(segmented_dir, f"image_{idx}_segmented.png")
        cv2.imwrite(segmented_path, segmented)

        if idx % 100 == 0:
            print(f"Processed {idx} images...")
    except Exception as e:
        print(f"Error processing image index {idx}: {e}")

# Convert list to numpy array
segmented_images = np.array(segmented_images)

# Save the segmented dataset only if there are segmented images
if segmented_images.size > 0:
    # If you wish to save only the labels corresponding to processed images
    # you can filter augmented_labels using processed_indices
    segmented_labels = augmented_labels[processed_indices]

    np.savez(
        'segmented_dataset.npz',
        images=segmented_images,
        labels=segmented_labels  # Labels corresponding to processed images
    )

    print("Segmented dataset saved as 'segmented_dataset.npz'")
    print(f"Segmented images saved in '{segmented_dir}'")
    print(f"Total images processed: {len(augmented_images)}")
    print(f"Total images skipped (no nuclei detected): {skipped_count}")
    print(f"Total images segmented: {len(segmented_images)}")
else:
    print("No segmented images to save.")

