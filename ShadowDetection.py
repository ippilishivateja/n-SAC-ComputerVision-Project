#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """ Load image in grayscale. """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def detect_shadows(image):
    """ Detects shadows while ignoring dark object parts. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, shadow_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY_INV)
    
    edges = cv2.Canny(image, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    refined_shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(edges_dilated))
    
    return shadow_mask, refined_shadow_mask

def remove_object_mask(image, refined_shadow_mask):
    """ Removes object mask while keeping the shadow highlighted. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, object_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    object_mask = cv2.dilate(object_mask, np.ones((5,5), np.uint8), iterations=2)
    shadow_only = cv2.bitwise_and(refined_shadow_mask, cv2.bitwise_not(object_mask))
    highlighted_image = image.copy()
    highlighted_image[shadow_only > 0] = [0, 0, 255]
    return shadow_only, highlighted_image

def save_results(output_folder, image_name, shadow_mask, refined_mask, shadow_only, highlighted_image):
    """ Save the processed images. """
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_shadow_mask.png"), shadow_mask)
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_refined_mask.png"), refined_mask)
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_shadow_only.png"), shadow_only)
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_highlighted.png"), highlighted_image)

def visualize_results(original, shadow_mask, refined_mask, highlighted_image, image_name):
    """ Display images side by side for comparison. """
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(shadow_mask, cmap='gray')
    plt.title("Raw Shadow Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(refined_mask, cmap='gray')
    plt.title("Refined Shadow Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title("Mask Removed - Shadow Highlighted")
    plt.axis("off")
    
    plt.suptitle(image_name)
    plt.show()

def process_images_in_folder(folder_path, output_folder):
    """ Process all images in a given folder and save results. """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print("No image files found in the folder.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            original_image = load_image(image_path)
            shadow_mask, refined_shadow_mask = detect_shadows(original_image)
            shadow_only, highlighted_image = remove_object_mask(original_image, refined_shadow_mask)
            save_results(output_folder, os.path.splitext(image_file)[0], shadow_mask, refined_shadow_mask, shadow_only, highlighted_image)
            visualize_results(original_image, shadow_mask, refined_shadow_mask, highlighted_image, image_file)
        except ValueError as e:
            print(e)

# Specify folder paths
folder_path = '/Users/vyakhyaverma/Downloads/TEST'  # Replace with your folder path
output_folder = '/Users/vyakhyaverma/Downloads/TEST_2'  # Replace with your output folder path
process_images_in_folder(folder_path, output_folder)

