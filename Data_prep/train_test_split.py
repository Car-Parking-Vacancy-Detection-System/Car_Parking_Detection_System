import os
import random
import shutil

# Define paths
image_train_dir = "./Final Merge/final/images/train"
image_val_dir = "./Final Merge/final/images/val"
label_train_dir = "./Final Merge/final/labels/train"
label_val_dir = "./Final Merge/final/labels/val"

# Ensure val directories exist
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Get list of image files (assuming .jpg, modify if needed)
image_files = [f for f in os.listdir(image_train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Determine number of files to move
num_to_move = int(0.2 * len(image_files))
files_to_move = random.sample(image_files, num_to_move)

# Move images and corresponding labels
for image_file in files_to_move:
    image_path = os.path.join(image_train_dir, image_file)
    label_path = os.path.join(label_train_dir, image_file.replace(os.path.splitext(image_file)[1], ".txt"))
    
    # Move image
    shutil.move(image_path, os.path.join(image_val_dir, image_file))
    
    # Move label if it exists
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(label_val_dir, os.path.basename(label_path)))

print(f"Moved {num_to_move} images and corresponding labels to validation set.")
