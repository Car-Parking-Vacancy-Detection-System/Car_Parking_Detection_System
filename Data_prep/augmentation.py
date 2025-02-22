import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from tqdm import tqdm

# Define a more balanced augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ToTensorV2()
])

# Input and output directories
parent_folder = "distinct_frames"  # Parent folder containing video frame folders
output_folder = "new_augmented_dataset"
summary_file = "new_augmentation_summary.txt"  # Summary file

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each video folder
video_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

total_augmented = 0
summary_data = []

for video_folder in tqdm(video_folders, desc="Processing Video Folders"):
    video_path = os.path.join(parent_folder, video_folder)
    save_path = os.path.join(output_folder, video_folder)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    frames = [f for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png")]
    num_original = len(frames)
    num_augmented = 0

    for frame_name in frames:
        frame_path = os.path.join(video_path, frame_name)
        image = cv2.imread(frame_path)

        if image is None:
            continue  # Skip unreadable images

        # Generate 5 augmentations per frame instead of 10
        for i in range(5):  
            augmented = augmentations(image=image)['image']
            augmented_np = augmented.numpy().transpose(1, 2, 0)  # Convert to OpenCV format
            
            aug_filename = f"{os.path.splitext(frame_name)[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(save_path, aug_filename), augmented_np)

            total_augmented += 1
            num_augmented += 1

    summary_data.append(f"Video: {video_folder}\nOriginal Frames: {num_original}\nAugmented Frames: {num_augmented}\n-----------------------------\n")

# Write summary file
with open(summary_file, "w") as f:
    f.write("Augmentation Summary\n=====================\n")
    f.writelines(summary_data)
    f.write(f"\nTotal Augmented Images: {total_augmented}\n")

print(f"✅ Augmentation Complete! Total Augmented Images: {total_augmented}")
print(f"📜 Summary saved to {summary_file}")
