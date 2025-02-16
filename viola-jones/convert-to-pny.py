import numpy as np
from PIL import Image
import glob
import os

def convert_images_to_npy(image_folder, output_file):
    images = []
    total_images = 0
    print(f"Checking folder: {image_folder}")
    for extension in ('*.jpg', '*.png'):  # Vérifiez les extensions .jpg et .png
        files = glob.glob(os.path.join(image_folder, extension))
        print(f"Found {len(files)} files with extension {extension} in {image_folder}")
        total_images += len(files)
        for filename in files:
            print(f"Processing image: {filename}")
            img = Image.open(filename).convert('L')  # Convert to grayscale
            img = img.resize((19, 19))  # Resize to 19x19
            img = np.array(img)
            images.append(img)
            print(f"Loaded image: {filename}, shape: {img.shape}")
    images = np.array(images)
    np.save(output_file, images)
    print(f"Saved {len(images)} images to {output_file}")
    print(f"Total images processed from {image_folder}: {total_images}")

def verify_npy_files(file_paths):
    for file_path in file_paths:
        data = np.load(file_path)
        print(f"{file_path} shape: {data.shape}")
        if data.shape[0] == 0:
            raise ValueError(f"File {file_path} is empty. Please check the images in the corresponding folder.")

# Convert training and test sets using images from testdata/faces and testdata/nofaces
convert_images_to_npy('C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/testdata/faces', 'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_train_faces_19x19g.npy')
convert_images_to_npy('C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/testdata/nonfaces', 'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_train_nofaces_19x19g.npy')
convert_images_to_npy('C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/testdata/faces', 'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_test_faces_19x19g.npy')
convert_images_to_npy('C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/testdata/nonfaces', 'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_test_nofaces_19x19g.npy')

# Verify the shapes of the .npy files
verify_npy_files([
    'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_train_faces_19x19g.npy',
    'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_train_nofaces_19x19g.npy',
    'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_test_faces_19x19g.npy',
    'C:/Users/Skyzo/Desktop/M2 IA2/kernel/viola-jones/datasets/cbcl_test_nofaces_19x19g.npy'
])