import numpy as np
import os

# Load the .npz dataset
npz_file_path = 'training_set.npz'
data = np.load(npz_file_path)

# Create a directory to save exported images
output_dir = 'exported_images_clean'
os.makedirs(output_dir, exist_ok=True)

# Extract images and labels (assuming your dataset has 'images' and 'labels' keys)
images = data['images']
labels = data.get('labels', None)  # Optional, if labels are included

# Loop through and save each image
for idx, image in enumerate(images):
    image_path = os.path.join(output_dir, f'image_{idx}.png')
    # Save the image (convert to uint8 if needed)
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    from PIL import Image
    Image.fromarray(image_uint8).save(image_path)
    print(f'Saved {image_path}')

    # Optionally, save a text file with labels if labels are present
    if labels is not None:
        with open(os.path.join(output_dir, 'labels.txt'), 'a') as label_file:
            label_file.write(f'image_{idx}.png: {labels[idx]}\n')

print("All images exported successfully!")
