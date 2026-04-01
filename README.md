# NeuroScan.ai - Brain Tumor Detection 

## Overview

**NeuroScan.ai** is a platform designed for the detection and classification of brain tumors from MRI scans. Leveraging state-of-the-art deep learning models and ensemble methods, NeuroScan.ai provides accurate and reliable predictions to support diagnostic workflows in healthcare settings.

### Key Features

- 🧠 **Accurate Tumor Detection**: Identifies brain tumors with high precision using ensemble deep learning models
- 📊 **Multi-Model Ensemble**: Combines DenseNet121, ResNet18, and MobileNetV2 for robust predictions
- 🔍 **Grad-CAM Visualization**: Visual explanations showing which regions of the MRI scan influenced the prediction
- 📈 **Confidence Scores**: Provides confidence levels for each prediction
- 🏥 **Medical-Grade Accuracy**: Designed to serve as a valuable second opinion tool for healthcare professionals
- 💻 **User-Friendly Interface**: Intuitive Streamlit-based web application

---

## What is a Brain Tumor?

A brain tumor is an abnormal mass or growth of cells in the brain. Because the skull is rigid and cannot expand, any growth inside it can cause significant health complications. Brain tumors can originate from brain cells (primary tumors) or spread from cancer cells in other parts of the body (secondary tumors).


## Screenshots

### About Page
- Platform overview and capabilities
- Brain tumor education
- Model architecture information
- <img width="1600" height="701" alt="image" src="https://github.com/user-attachments/assets/351ef15a-c274-43f5-8386-477cd66f0cac" />

### Tumor Detection Interface
- MRI image upload and display
- Real-time predictions with confidence scores
- Detailed tumor information and symptoms
- <img width="1600" height="742" alt="image" src="https://github.com/user-attachments/assets/00c5fb3a-c7c6-4c93-a375-334346416bdb" />


### Grad-CAM Visualization
- Original MRI scan alongside Grad-CAM heatmaps
- Class-specific attention maps for all ensemble models
- Heatmap legend for interpretation
- <img width="1600" height="725" alt="image" src="https://github.com/user-attachments/assets/fb76e918-30a9-492e-9c56-ca304062dde8" />



## Models

NeuroScan.ai uses an ensemble approach combining three state-of-the-art deep learning architectures:
 
### ResNet-18

ResNet-18 is a deep convolutional neural network that uses residual connections to avoid vanishing gradient problems.
It enables stable and efficient training even with deeper architectures.
In this project, it provides reliable and high-performing predictions for tumor classification.

### DenseNet-121

DenseNet-121 connects each layer to every other layer, improving feature reuse and gradient flow.
This architecture captures complex patterns and fine details in MRI images.
It enhances the model’s ability to detect subtle tumor features.

### MobileNetV2

MobileNetV2 is a lightweight and efficient neural network designed for fast computation.
It uses depthwise separable convolutions to reduce complexity while maintaining performance.
In this project, it enables quick and efficient predictions suitable for real-time applications.

### Ensemble Strategy

The three models work together using an ensemble voting mechanism:
1. Each model independently analyzes the MRI scan
2. Individual predictions are combined using weighted averaging
3. Final prediction includes confidence scores for each tumor class
4. Increased robustness and reliability compared to single models
   

## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV, NumPy
- **Model Visualization**: Grad-CAM
- **Pre-trained Models**: ResNet, DenseNet, MobileNet (ImageNet)


---


## Important Disclaimer

⚠️ **MEDICAL DISCLAIMER**: NeuroScan.ai is designed as a **supplementary diagnostic tool** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. This tool is intended to assist radiologists and healthcare providers as a second opinion in their diagnostic workflows.

---

