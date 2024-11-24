import numpy as np
import cv2
import os
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("segmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def segment_nucleus(
    image,
    median_blur_kernel=5,
    adaptive_thresh_block_size=11,
    adaptive_thresh_C=2,
    morph_kernel_size=(5, 5),
    area_threshold=(200, 5000)
):
    """
    Segments nuclei in the given image using configurable parameters.

    Parameters:
    - image (numpy.ndarray): Input image array (H, W, C) in BGR format.
    - median_blur_kernel (int): Kernel size for median blur.
    - adaptive_thresh_block_size (int): Block size for adaptive thresholding.
    - adaptive_thresh_C (int): Constant subtracted from the mean in adaptive thresholding.
    - morph_kernel_size (tuple): Kernel size for morphological operations.
    - area_threshold (tuple): (min_area, max_area) to filter contours based on area.

    Returns:
    - segmented_only (numpy.ndarray): Image containing only the segmented nuclei.
    """
    if image is None:
        raise ValueError("Input image is None")

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply median blur to reduce noise
        denoised = cv2.medianBlur(gray, median_blur_kernel)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_thresh_block_size,
            adaptive_thresh_C
        )

        # Morphological operations to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for segmented nuclei
        mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area_threshold[0] < area < area_threshold[1]:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image to extract only the nuclei
        segmented_only = cv2.bitwise_and(image, image, mask=mask)

        return segmented_only

    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return None

def is_almost_black(segmented_image, threshold_ratio=0.01):
    """
    Determines if the segmented image is almost black based on the ratio of non-zero pixels.

    Parameters:
    - segmented_image (numpy.ndarray): The segmented image.
    - threshold_ratio (float): The ratio below which the image is considered almost black.

    Returns:
    - bool: True if the image is almost black, False otherwise.
    """
    if segmented_image is None:
        return True  # Treat None as almost black

    # Convert to grayscale to count non-zero pixels
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    non_zero = cv2.countNonZero(gray)
    total_pixels = gray.size
    ratio = non_zero / total_pixels
    return ratio < threshold_ratio

def process_image(
    idx,
    image,
    config,
    segmented_dir
):
    """
    Processes a single image: segmentation, checking, saving.

    Parameters:
    - idx (int): Index of the image.
    - image (numpy.ndarray): Image array.
    - config (dict): Configuration parameters.
    - segmented_dir (str): Directory to save segmented images.

    Returns:
    - tuple: (processed, idx) where 'processed' is the segmented image or None.
    """
    try:
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # If image has a single channel, convert to BGR
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Segment the nucleus
        segmented = segment_nucleus(
            image,
            median_blur_kernel=config['median_blur_kernel'],
            adaptive_thresh_block_size=config['adaptive_thresh_block_size'],
            adaptive_thresh_C=config['adaptive_thresh_C'],
            morph_kernel_size=config['morph_kernel_size'],
            area_threshold=config['area_threshold']
        )

        # Check if the segmented image is almost black
        if is_almost_black(segmented, threshold_ratio=config['black_threshold_ratio']):
            logger.info(f"Skipped image {idx} as it contains no detectable nuclei.")
            return (False, idx, None)

        # Save segmented image to disk
        segmented_path = os.path.join(segmented_dir, f"image_{idx}_segmented.png")
        cv2.imwrite(segmented_path, segmented)

        return (True, idx, segmented)

    except Exception as e:
        logger.error(f"Error processing image index {idx}: {e}")
        return (False, idx, None)

def main():
    # Configuration parameters
    config = {
        'median_blur_kernel': 5,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 2,
        'morph_kernel_size': (5, 5),
        'area_threshold': (200, 5000),
        'black_threshold_ratio': 0.01  # 1%
    }

    # Load the augmented dataset
    try:
        augmented_data = np.load('augmented_set3.npz')
        augmented_images = augmented_data['images']  # Shape: (N, H, W, C)
        augmented_labels = augmented_data['labels']
        logger.info(f"Loaded augmented dataset with {len(augmented_images)} images.")
    except Exception as e:
        logger.error(f"Failed to load augmented dataset: {e}")
        return

    # Initialize lists to store segmented images and their indices
    segmented_images = []
    processed_indices = []

    # Directory to save segmented images
    segmented_dir = '/home/zfirefist/Desktop/ANN-DL_2024_PoliMi/segmented_images'
    os.makedirs(segmented_dir, exist_ok=True)
    logger.info(f"Segmented images will be saved to '{segmented_dir}'.")

    # Partial function for multiprocessing
    partial_process_image = partial(
        process_image,
        config=config,
        segmented_dir=segmented_dir
    )

    # Use multiprocessing Pool
    num_processes = cpu_count()
    logger.info(f"Starting segmentation using {num_processes} processes.")

    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(partial_process_image, enumerate(augmented_images)),
            total=len(augmented_images),
            desc="Segmenting Images"
        ))

    # Process results
    skipped_count = 0
    for result in results:
        processed, idx, segmented = result
        if processed:
            segmented_images.append(segmented)
            processed_indices.append(idx)
        else:
            skipped_count += 1

    # Convert list to numpy array
    if segmented_images:
        segmented_images = np.array(segmented_images)
        # Filter labels based on processed_indices
        segmented_labels = augmented_labels[processed_indices]

        # Save the segmented dataset
        try:
            np.savez(
                'segmented_dataset.npz',
                images=segmented_images,
                labels=segmented_labels
            )
            logger.info("Segmented dataset saved as 'segmented_dataset.npz'.")
        except Exception as e:
            logger.error(f"Failed to save segmented dataset: {e}")
    else:
        logger.warning("No segmented images to save.")

    # Summary
    logger.info(f"Total images processed: {len(augmented_images)}")
    logger.info(f"Total images skipped (no nuclei detected): {skipped_count}")
    logger.info(f"Total images segmented: {len(segmented_images)}")

if __name__ == "__main__":
    main()
