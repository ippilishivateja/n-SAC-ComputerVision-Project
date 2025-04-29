# Geometry-Based Anti-Falsing for Computer Vision

**Introduction**
The Geometry-Based Anti-Falsing for Computer Vision project is a cutting-edge initiative that addresses one of the most pressing challenges in modern automated systems: improving the reliability and robustness of computer vision (CV) systems in complex environments. Automated systems, such as self-driving cars, heavily rely on CV to interpret and analyze their surroundings. However, environmental factors such as shadows, reflections, and atmospheric conditions like humidity significantly hinder the performance of CV algorithms. These factors can introduce false positives and negatives, potentially leading to critical errors in real-world applications.
This project aims to overcome these challenges by leveraging the innovative n-Sphere Anti-Falsing Calculation (n-SAC) method. The n-SAC technique focuses on geometric principles to mitigate the impact of shadows and lighting inconsistencies, providing a computationally efficient and adaptable solution. The ultimate goal is to validate the n-SAC method’s effectiveness in real-world scenarios by using video data from moving platforms under varying conditions. Through this project, the team seeks to enhance the accuracy and reliability of CV systems, contributing to the broader goal of advancing autonomous technologies and supporting critical applications in security and defense.

**Problem Statement and Goals**
The Automated systems relying on CV often face inaccuracies due to shadows and environmental factors, especially in real-time, 2D and 3D spaces. This project aims to validate the n-SAC method as a solution to these issues. The specific goals include:
•	Testing the n-SAC method for recognizing known objects, such as buildings, in video data.
•	Evaluating its robustness under varying lighting and environmental conditions.
•	Developing a working prototype that demonstrates the application of n-SAC.

**Dataset**
* Synthetic Dataset: Generated using ChatGPT Sora, simulating drone and aerial views with lighting variations.
* Dashcam Footage: Real-world scenarios like overpasses, airplanes, and urban environments.
Each dataset contains multiple videos, which were converted to image frames for processing and analysis.

**Models**
*Grayscale Conversion: Converts colored frames to grayscale to simplify feature detection and reduce noise.
*Noise Reduction: Applies dehazing and filtering techniques to enhance image clarity and highlight object boundaries.
*Shadow Detection: Uses HSV thresholding and morphological operations to isolate shadow regions from frames.
*n-SAC Classification: Tracks changes in inclination angles across frames to differentiate real objects from shadows based on motion consistency.

**Results**
The proposed system effectively detected and filtered complex shadow patterns, improving classification accuracy under varied lighting conditions. It demonstrated reliable performance across synthetic and real-world datasets, reducing false positives and correctly identifying shadow-only scenarios.

**Conclusion**
The n-Sphere Anti-Falsing Calculation (n-SAC) method presents a novel approach to enhancing object detection accuracy in computer vision systems. By leveraging the rate of change in inclination angles, n-SAC distinguishes real objects from shadows without relying on deep learning or labeled data. Through a hybrid detection pipeline combining classical image processing techniques with geometric analysis, the method achieved strong performance across standard and challenging environments, including cases with limited visibility of real objects.

**Requirements**
* Python 3.x
* OpenCV
* NumPy
* Matplotlib
* scikit-image
You can install all dependencies using:
bashCopyEditpip install -r requirements.txt

**Reference**
* Project: DAEN 690 Capstone – George Mason University
* Video Generation: ChatGPT Sora
* YouTube Guide: Fractal Shadows
* Concept inspired by real-world CV failures and phantom braking case studies
