# n-SAC-ComputerVision-Project
Validation of n-Sphere Anti-Falsing Calculation (n-SAC) for overcoming shadows in computer vision


Introduction
The Geometry-Based Anti-Falsing for Computer Vision project is a cutting-edge initiative that addresses one of the most pressing challenges in modern automated systems: improving the reliability and robustness of computer vision (CV) systems in complex environments. Automated systems, such as self-driving cars, heavily rely on CV to interpret and analyze their surroundings. However, environmental factors such as shadows, reflections, and atmospheric conditions like humidity significantly hinder the performance of CV algorithms. These factors can introduce false positives and negatives, potentially leading to critical errors in real-world applications.
This project aims to overcome these challenges by leveraging the innovative n-Sphere Anti-Falsing Calculation (n-SAC) method. The n-SAC technique focuses on geometric principles to mitigate the impact of shadows and lighting inconsistencies, providing a computationally efficient and adaptable solution. The ultimate goal is to validate the n-SAC method’s effectiveness in real-world scenarios by using video data from moving platforms under varying conditions. Through this project, the team seeks to enhance the accuracy and reliability of CV systems, contributing to the broader goal of advancing autonomous technologies and supporting critical applications in security and defense.

Problem Statement and Goals:
The Automated systems relying on CV often face inaccuracies due to shadows and environmental factors, especially in real-time, 2D and 3D spaces. This project aims to validate the n-SAC method as a solution to these issues. The specific goals include:
•	Testing the n-SAC method for recognizing known objects, such as buildings, in video data.
•	Evaluating its robustness under varying lighting and environmental conditions.
•	Developing a working prototype that demonstrates the application of n-SAC.


In this project, we aim to solve the problem of shadow misclassification in computer vision. This is an object classification problem in which we identify whether a detected shape in a video frame is a real object or just a shadow.

Problem Statement
Shadows in images can lead to major misclassifications in autonomous systems like self-driving cars, drones, and surveillance cameras. These shadows—especially fractal shadows—often resemble object edges and confuse vision systems, causing issues like phantom braking or navigation errors.
This project automates the process of shadow detection and classification using a geometry-based approach called n-Sphere Anti-Falsing Calculation (n-SAC). This method evaluates how object angles change over time, helping us differentiate between real objects and shadows even in complex environments.

Objective
The goal is to build a pipeline that can process video data, detect shadows, and use geometric motion analysis to classify what is a shadow and what is not. This helps reduce false positives in real-time vision systems.

About Dataset
We used two datasets for training and testing:
* Synthetic Dataset: Generated using ChatGPT Sora, simulating drone and aerial views with lighting variations.
* Dashcam Footage: Real-world scenarios like overpasses, airplanes, and urban environments.
Each dataset contains multiple videos, which were converted to image frames for processing and analysis.

Data Description
The frames include:
* Daylight and low-light conditions
* Scenarios with only shadows (no visible object)
* Fractal shadow patterns from buildings and moving aircraft
* Real-world cases like phantom braking
Each frame goes through preprocessing, shadow detection, and n-SAC-based classification.

Models Used
We used a custom pipeline consisting of:
* Grayscale Conversion: Simplifies processing
* Noise Reduction: Enhances clarity under fog, rain, and light changes
* Shadow Detection: HSV masking and morphological filtering
* n-SAC Classification: Calculates angle changes to classify object vs. shadow

Results
We tested the model on multiple complex cases:
* Fractal shadows were detected and filtered effectively
* Airplane shadow-only footage was handled accurately
* Phantom braking in dashcam footage was minimized by filtering rapid shadow changes

Demo
Watch output videos and demos on our YouTube channel: 👉 Fractal Shadows – Demo Channel

Requirements
* Python 3.x
* OpenCV
* NumPy
* Matplotlib
* scikit-image
You can install all dependencies using:
bashCopyEditpip install -r requirements.txt

Reference
* Project: DAEN 690 Capstone – George Mason University
* Video Generation: ChatGPT Sora
* YouTube Guide: Fractal Shadows
* Concept inspired by real-world CV failures and phantom braking case studies
