import os
import cv2

# Define the input and output directories
input_dir = 'test'
output_dir = 'output_directory'
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories in the augmented directory
for label in os.listdir(input_dir):
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Define the unsharp mask parameters
kernel_size = (5, 5)
sigma = 1.0
amount = 1.5
threshold = 0

# Iterate through each file in the input directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        # Load the image
        input_path = os.path.join(dirpath, filename)
        output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = cv2.imread(input_path)

        # Check if the image is empty
        if img is None:
            print(f"Error: {input_path} is empty")
            continue

        # Apply unsharp masking to enhance the image clarity
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        sharpened = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_TOZERO)[1]

        # Save the enhanced image to the output directory
        cv2.imwrite(output_path, sharpened)
