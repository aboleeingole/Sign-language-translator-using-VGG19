import os
import random

# Set the path to the augmented images directory
aug_dir = 'augmented_images'

# Shuffle the images in each subdirectory
for label in os.listdir(aug_dir):
    label_dir = os.path.join(aug_dir, label)

    if not os.path.isdir(label_dir):
        continue

    images = os.listdir(label_dir)
    random.shuffle(images)

    for i, img_name in enumerate(images):
        src_path = os.path.join(label_dir, img_name)
        dst_path = os.path.join(label_dir, f'{i+1}.jpg')
        os.rename(src_path, dst_path)
