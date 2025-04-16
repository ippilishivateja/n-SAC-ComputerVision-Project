#!/usr/bin/env python
# coding: utf-8

# In[63]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def dark_channel_prior(img, window_size=15):
    """
    Compute the dark channel prior for an image.
    """
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(img, dark_channel):
    """
    Estimate the atmospheric light (A) from the dark channel.
    """
    flat_dark = dark_channel.flatten()
    flat_img = img.reshape(-1, 3)
    top_percentile = 0.1
    num_pixels = int(flat_dark.shape[0] * top_percentile)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    atmospheric_light = np.max(flat_img[indices], axis=0)
    return atmospheric_light

def estimate_transmission(img, atmospheric_light, omega=0.95, window_size=15):
    """
    Estimate the transmission map for the image.
    """
    normalized_img = img / atmospheric_light
    dark_channel = dark_channel_prior(normalized_img, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

def dehaze_image(img, window_size=15, omega=0.95, t0=0.1):
    """
    Dehaze an image using the Dark Channel Prior (DCP) algorithm.
    """
    img = img.astype(np.float32) / 255.0

    dark_channel = dark_channel_prior(img, window_size)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel)

    transmission = estimate_transmission(img, atmospheric_light, omega, window_size)

    transmission = cv2.bilateralFilter(transmission, d=9, sigmaColor=75, sigmaSpace=75)

    transmission = np.clip(transmission, t0, 1.0)

    scene_radiance = np.zeros_like(img)
    for i in range(3):
        scene_radiance[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

    scene_radiance = np.clip(scene_radiance, 0, 1)
    scene_radiance = (scene_radiance * 255).astype(np.uint8)

    return scene_radiance

def enhance_image_comprehensive(img):
    """
    Comprehensive image enhancement with dehazing for foggy images.
    """
    if len(img.shape) == 3:
        img = dehaze_image(img)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(img, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    img_clahe1 = clahe1.apply(denoised)
    img_clahe2 = clahe2.apply(denoised)
    img_clahe = cv2.addWeighted(img_clahe1, 0.4, img_clahe2, 0.6, 0)
    img_bilateral = cv2.bilateralFilter(img_clahe, d=3, sigmaColor=10, sigmaSpace=10)
    min_val = np.percentile(img_bilateral, 2)
    max_val = np.percentile(img_bilateral, 98)
    img_contrast = cv2.normalize(img_bilateral, None, min_val, max_val, cv2.NORM_MINMAX)
    enhanced = cv2.addWeighted(denoised, 0.7, img_contrast, 0.3, 0)

    return enhanced

def process_and_display(image_path, output_path):
    """
    Process image and display/save results.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image file: {image_path}")
        return

    enhanced_img = enhance_image_comprehensive(img)

    plt.figure(figsize=(20,8))
    plt.subplot(121)
    plt.title('Original Image', fontsize=14, pad=10)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    plt.title('Enhanced Image\n(Dehazing + Environmental Correction)', fontsize=14, pad=10)
    plt.imshow(enhanced_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite(output_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Enhanced image saved as: {output_path}")

input_image = 'C:\\Users\\User\\Downloads\\gray_3091.jpg'  # Your input image
output_image = 'C:\\Users\\User\\Downloads\\enhanced_image1.jpg'  # Output filename
process_and_display(input_image, output_image)


# In[ ]:




