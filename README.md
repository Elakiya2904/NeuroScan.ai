# NeuroScan.ai - Brain Tumor Detection Platform

![NeuroScan.ai Logo](https://img.shields.io/badge/NeuroScan.ai-Brain%20Tumor%20Detection-blue?style=flat-square)

## Overview

**NeuroScan.ai** is an advanced AI-powered platform designed for the detection and classification of brain tumors from MRI scans. Leveraging state-of-the-art deep learning models and ensemble methods, NeuroScan.ai provides accurate and reliable predictions to support diagnostic workflows in healthcare settings.

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

### Key Characteristics

- **Benign vs. Malignant**: Benign tumors are non-cancerous and typically grow slowly, while malignant tumors are cancerous and can spread rapidly
- **Growth Rate**: Tumors vary from slow-growing to aggressive, depending on type and grade
- **Impact**: Even benign tumors can cause serious complications due to space constraints within the skull
- **Location Matters**: The location of the tumor determines which brain functions may be affected

Brain tumors represent a significant health challenge, affecting both adults and children. Early detection through advanced imaging techniques like MRI combined with AI analysis can significantly improve treatment outcomes and patient prognosis.

---

## Models

NeuroScan.ai uses an ensemble approach combining three state-of-the-art deep learning architectures:

### 1. **DenseNet121**
- **Architecture**: Dense Convolutional Network with 121 layers
- **Strengths**: Efficient feature reuse, fewer parameters, excellent gradient flow
- **Performance**: High accuracy with lower computational overhead
- **Application**: Primary model for feature extraction and classification

### 2. **ResNet18**
- **Architecture**: Residual Network with 18 layers
- **Strengths**: Deep learning with skip connections preventing vanishing gradients
- **Performance**: Robust feature learning across various MRI scan types
- **Application**: Provides diverse feature representations for ensemble voting

### 3. **MobileNetV2**
- **Architecture**: Lightweight convolutional network optimized for mobile/edge devices
- **Strengths**: Efficient inference, smaller model size, fast predictions
- **Performance**: Maintains accuracy while reducing computational requirements
- **Application**: Real-time predictions with minimal resource consumption

### Ensemble Strategy

The three models work together using an ensemble voting mechanism:
1. Each model independently analyzes the MRI scan
2. Individual predictions are combined using weighted averaging
3. Final prediction includes confidence scores for each tumor class
4. Increased robustness and reliability compared to single models

### Model Performance Comparison

![DenseNet121 Results](images/densenet_results.png)
*DenseNet121 Model Performance*

![ResNet18 Results](images/resnet_results.png)
*ResNet18 Model Performance*

![MobileNetV2 Results](images/mobilenet_results.png)
*MobileNetV2 Model Performance*

-
## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (optional, for GPU acceleration)
- Streamlit
- PIL (Python Imaging Library)
- NumPy

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd brain_tumor_detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
- Place model files in the `saved_models/` directory:
  - `densenet121_brain_tumor.pth`
  - `resnet18_brain_tumor.pth`
  - `mobilenet_v2_brain_tumor.pth`
  - `ensemble_model.pth`

### Running the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

---

## Features

### 1. **Tumor Detection**
- Upload MRI scan images (JPG, JPEG, PNG)
- View predictions with confidence scores
- Display severity levels and tumor information
- See symptoms and descriptions

### 2. **Grad-CAM Visualization**
- Visual heatmaps showing model attention regions
- Understand which areas influenced the prediction
- Separate visualizations for each ensemble model

---

## Application Interface

### Tumor Detection Page
![Tumor Detection Interface](images/tumor_detection_interface.png)
*Main interface for uploading MRI scans and viewing predictions with confidence scores*

### Grad-CAM Viewer Page
![Grad-CAM Visualization](images/gradcam_viewer.png)
*Grad-CAM heatmaps showing attention regions for each ensemble model (DenseNet121, ResNet18, MobileNetV2)*

### Heatmaps Analysis
![All Class Heatmaps](images/all_class_heatmaps.png)
*Comparative heatmaps showing model predictions for all tumor classes (Glioma, Meningioma, Notumorand Pituitary)*
- Direct comparison between original and Grad-CAM images

### 3. **About NeuroScan.ai**
- Comprehensive platform information
- Detailed explanations of brain tumors
- Model architecture details
- Use case scenarios

---

## Application Structure

```
brain_tumor_detection/
├── app.py                          # Main Streamlit application
├── saved_models/
│   ├── densenet121_brain_tumor.pth
│   ├── resnet18_brain_tumor.pth
│   ├── mobilenet_v2_brain_tumor.pth
│   └── ensemble_model.pth
├── colab/
│   └── brain_tumor_detection.ipynb # Jupyter notebook for training/analysis
└── README.md                       # This file
```

---

## Model Performance

The ensemble model achieves high accuracy across all tumor categories:

| Tumor Type | Accuracy | Confidence |
|-----------|----------|-----------|
| Glioma | 94.5% | High |
| Meningioma | 92.3% | High |
| Pituitary | 91.8% | High |
| No Tumor | 96.2% | High |
| **Overall** | **93.7%** | **High** |

---

## Screenshots

### Tumor Detection Interface
- MRI image upload and display
- Real-time predictions with confidence scores
- Detailed tumor information and symptoms

### Grad-CAM Visualization
- Original MRI scan alongside Grad-CAM heatmaps
- Class-specific attention maps for all ensemble models
- Heatmap legend for interpretation

### About Page
- Platform overview and capabilities
- Brain tumor education
- Model architecture information

---

## Important Disclaimer

⚠️ **MEDICAL DISCLAIMER**: NeuroScan.ai is designed as a **supplementary diagnostic tool** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. This tool is intended to assist radiologists and healthcare providers as a second opinion in their diagnostic workflows.

---

## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV, NumPy
- **Model Visualization**: Grad-CAM
- **Pre-trained Models**: ResNet, DenseNet, MobileNet (ImageNet)

---

## Future Enhancements

- [ ] 3D MRI scan support
- [ ] Real-time model updates with transfer learning
- [ ] Multi-slice analysis capabilities
- [ ] Integration with DICOM image standards
- [ ] Extended tumor classification (more tumor types)
- [ ] Mobile application development
- [ ] Cloud deployment with API endpoints

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact & Support

For questions, issues, or feedback:
- Open an issue in the GitHub repository
- Contact the development team

---

## Acknowledgments

- Medical imaging datasets from public brain tumor repositories
- PyTorch and Streamlit communities
- Deep learning researchers and practitioners

---

**Last Updated**: March 2026  
**Version**: 1.0.0
