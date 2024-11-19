import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def segment_sample_nucleus(image_path, save_path):
    # Check if the image exists
    if not os.path.isfile(image_path):
        print(f"Error: File does not exist at {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply stronger noise reduction with median filter
    denoised = cv2.medianBlur(gray, 5)

    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Use morphological operations to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw only relevant contours on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjust area filter based on expected nucleus size
        if 200 < area < 5000:  # These thresholds can be tuned
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    segmented_image_path = os.path.join(save_path, f"{base_name}_segmented.png")
    visualization_path = os.path.join(save_path, f"{base_name}_visualization.png")

    cv2.imwrite(segmented_image_path, image)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(thresh, cmap='gray'), plt.title('Thresholded Image')
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Segmented Nuclei')
    plt.savefig(visualization_path)

    print(f"Segmented image saved at {segmented_image_path}")
    print(f"Visualization saved at {visualization_path}")

# Path to the input image and output directory
input_image = '/home/zfirefist/Desktop/ANN-DL_2024_PoliMi/exported_images/image_10.png'  # Replace with your sample image
output_directory = '/home/zfirefist/Desktop/ANN-DL_2024_PoliMi/segmented_sample'
os.makedirs(output_directory, exist_ok=True)

segment_sample_nucleus(input_image, output_directory)
 