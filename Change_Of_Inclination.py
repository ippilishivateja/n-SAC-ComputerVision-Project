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
folder_path = '/Users/vyakhyaverma/Downloads/Test'  # Replace with your folder path
output_folder = '/Users/vyakhyaverma/Downloads/Test 2'  # Replace with your output folder path
process_images_in_folder(folder_path, output_folder)


# In[ ]:


import cv2
import numpy as np

def extract_shadow_mask(image):
    """Extracts the mask for red-highlighted shadows."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range (shadows)
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    # Create mask for red regions
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    return mask1 + mask2  # Combine both masks

def extract_object_edges(image, shadow_mask):
    """Extracts object edges while ignoring the red shadow."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[shadow_mask > 0] = 0  # Mask out shadow regions

    # Apply GaussianBlur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def calculate_inclination(edges):
    """Finds the inclination angle using Probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is not None:
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))  # Compute inclination
            angles.append(angle)
        
        # Filter out extreme angles
        angles = np.array(angles)
        filtered_angles = angles[(angles > -80) & (angles < 80)]  # Remove near-horizontal/vertical angles

        if len(filtered_angles) > 0:
            return np.mean(filtered_angles)  # Return robust average angle

    return None

def process_images(image_paths):
    """Processes 3 images, extracts inclination angles, and computes rate of change."""
    shadow_angles, object_angles = [], []

    for path in image_paths:
        image = cv2.imread(path)
        shadow_mask = extract_shadow_mask(image)

        shadow_angle = calculate_inclination(shadow_mask)
        object_edges = extract_object_edges(image, shadow_mask)
        object_angle = calculate_inclination(object_edges)

        if shadow_angle is not None:
            shadow_angles.append(shadow_angle)
        if object_angle is not None:
            object_angles.append(object_angle)

    if len(shadow_angles) < 3 or len(object_angles) < 3:
        print("Not enough valid frames to compute rate of change.")
        return

    # Compute rate of change using numpy.diff()
    shadow_rate = np.diff(shadow_angles)
    object_rate = np.diff(object_angles)

    # Classify based on rate change consistency
    threshold = 5  # Adjust threshold as needed
    shadow_class = "OBJECT" if np.all(np.abs(shadow_rate) < threshold) else "SHADOW"
    object_class = "OBJECT" if np.all(np.abs(object_rate) < threshold) else "SHADOW"

    print(f"Shadow Classification: {shadow_class}")
    print(f"Object Classification: {object_class}")
    print(f"Shadow Angle Changes: {shadow_angles}")
    print(f"Object Angle Changes: {object_angles}")
    print(f"Shadow Rate of Change: {shadow_rate}")
    print(f"Object Rate of Change: {object_rate}")

# Example usage
image_files = [
    "/Users/vyakhyaverma/Downloads/Test 2/1.png",
    "/Users/vyakhyaverma/Downloads/Test 2/2.png",
    "/Users/vyakhyaverma/Downloads/Test 2/3.png"
]
process_images(image_files)


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_angle_line(image, angle, color, label):
    """
    Draws an inclination line based on the given angle.
    """
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    
    angle_rad = np.radians(angle + 90)
    
    length = 200
    x1 = int(center[0] - length * np.cos(angle_rad))
    y1 = int(center[1] - length * np.sin(angle_rad))
    x2 = int(center[0] + length * np.cos(angle_rad))
    y2 = int(center[1] + length * np.sin(angle_rad))

    cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    cv2.putText(image, f"{label}: {angle:.2f}°", (50, 50 if color == (255, 0, 0) else 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

def visualize_results(image_paths, shadow_angles, object_angles):
    """
    Overlays detected angles on images and plots angle change over time.
    """
    images = []
    
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        
        if shadow_angles[i] is not None:
            image = draw_angle_line(image, shadow_angles[i], (0, 0, 255), "Shadow Angle")  # Red line
        
        if object_angles[i] is not None:
            image = draw_angle_line(image, object_angles[i], (255, 0, 0), "Object Angle")  # Blue line

        images.append(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, path in zip(axes, images, image_paths):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.split("/")[-1])
        ax.axis("off")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(shadow_angles)), shadow_angles, 'ro-', label="Shadow Angle")
    plt.plot(range(len(object_angles)), object_angles, 'bo-', label="Object Angle")
    plt.xlabel("Frame Number")
    plt.ylabel("Inclination Angle (degrees)")
    plt.legend()
    plt.title("Angle Change Over Time")
    plt.grid()
    plt.show()


# In[ ]:


shadow_angles = [-10.802963573194388, 25.74691869026442, 3.714753352701254]  # Replace with actual computed angles
object_angles = [0.31273860458331826, 0.6491273211768825, 2.008933713845934]  # Replace with actual computed angles


# In[ ]:




