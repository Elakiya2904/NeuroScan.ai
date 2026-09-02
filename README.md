# NeuroScan.ai: Brain Tumor Detection and Functional Impact Assessment

## Abstract

NeuroScan.ai is a computer-vision decision-support application for analyzing brain MRI images. The system classifies an image into glioma, meningioma, no tumor, or pituitary tumor using an ensemble of trained convolutional neural networks. It combines ResNet18 and DenseNet121 predictions at runtime, provides a class-specific Grad-CAM explanation, estimates a tumor mask with a U-Net segmentation model, extracts the mask centroid and bounding box, and maps the estimated location to a coarse brain region and potential functional impacts. The application is implemented in Python with PyTorch, OpenCV, PIL, and Streamlit. The saved checkpoint metadata reports 77.38% test accuracy for ResNet18, 77.75% for DenseNet121, and 77.69% for MobileNetV2; no accuracy value is stored for the ensemble checkpoint. Higher values in the analysis report are retained below as unverified analysis summaries and should not be presented as validated performance.

## Keywords

Brain tumor classification; MRI; deep learning; ensemble learning; ResNet18; DenseNet121; MobileNetV2; U-Net; Grad-CAM; explainable AI; tumor localization; functional brain mapping; Streamlit.

## I. Introduction

### Background

Magnetic resonance imaging is widely used to inspect brain tissue and characterize suspected tumors. Automated image analysis can support radiologists by providing consistent first-pass classification, localization, and visual explanations.

### Problem Statement

Manual interpretation is time-intensive, while a classification label alone does not explain which image region influenced a prediction or how a lesion may relate to brain function. A practical prototype therefore needs classification, visual explanation, segmentation, localization, and an accessible interface in one workflow.

### Motivation

NeuroScan.ai was developed to combine model prediction with interpretable image evidence and an educational functional-region assessment. It is a supplementary decision-support prototype, not a diagnostic device.

### Contributions

- Four-class MRI classification using trained ResNet18 and DenseNet121 checkpoints.
- Runtime ensemble prediction by averaging the two model logits.
- Class-specific Grad-CAM visualization using the final ResNet18 convolutional block.
- U-Net-based binary tumor segmentation with area, centroid, and bounding-box output.
- Coarse centroid-to-lobe mapping and potential functional impact descriptions.
- Streamlit interface for image upload and end-to-end result presentation.

## II. Related Work

### Existing Brain Tumor Classification Methods

Transfer learning with CNN architectures such as ResNet, DenseNet, and MobileNet is a common approach for MRI classification. Residual connections help optimize deeper networks, dense connectivity encourages feature reuse, and MobileNet reduces computation through lightweight convolutional blocks.

### Tumor Segmentation Approaches

U-Net and its variants are widely used for biomedical segmentation because encoder features can be combined with decoder features through skip connections. The implementation uses a custom U-Net with four encoder and four decoder stages and a one-channel output mask.

### Explainable AI / Grad-CAM

Grad-CAM uses gradients flowing into a convolutional feature layer to create a coarse class-specific importance map. In this project, the map is generated from ResNet18 layer `layer4[-1].conv2` and resized to the 224×224 classification image.

### Brain Functional Mapping Approaches

The project uses a rule-based spatial heuristic rather than a neuroanatomically registered atlas. A normalized 2-D centroid is assigned to a broad region such as frontal, parietal, temporal, occipital, cerebellar, or pituitary/sellar.

### Research Gap

Many prototypes focus on classification only. NeuroScan.ai demonstrates a unified workflow that combines prediction, explanation, segmentation, localization, and functional-impact context, while also making the limitations of coarse 2-D mapping explicit.

## III. Proposed Methodology

### A. System Overview

An uploaded image is stored in Streamlit session state and passed through classification. The predicted class selects the Grad-CAM target. For tumor classes other than no tumor, the image is also segmented, localized, and mapped to a functional region.

### B. Dataset Description

The project analysis report describes 3,596 RGB images at 256×256 resolution: 898 glioma, 898 meningioma, 898 pituitary, and 902 no-tumor images. The report also describes approximately 85% training, 7.5% validation, and 7.5% test partitions and a five-fold evaluation. These dataset details come from project analysis artifacts, not from metadata enforced by the application code, and should be independently verified before publication.

### C. Image Preprocessing

Classification images are converted to RGB, resized to 224×224 using Lanczos resampling, scaled to [0, 1], normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`, and converted to a channel-first PyTorch tensor. Segmentation uses an RGB 256×256 image scaled to [0, 1].

### D. Brain Tumor Classification

The class order used by the application is glioma, meningioma, no tumor, and pituitary.

#### DenseNet121

The DenseNet121 checkpoint uses a four-class custom classifier head. Dense connectivity provides feature reuse and complementary representations for the ensemble.

#### MobileNetV2

The repository contains a four-class MobileNetV2 checkpoint whose saved metadata reports 77.69% test accuracy. It is a lightweight comparison/deployment option; the current `load_ensemble_model()` runtime path loads the ResNet18 and DenseNet121 entries stored in `ensemble_model.pth`.

#### ResNet18

The ResNet18 checkpoint uses a four-class custom classifier head. Its final convolutional block is also used as the Grad-CAM target layer.

#### Ensemble Model

The saved ensemble checkpoint contains ResNet18 and DenseNet121 state dictionaries. The application reconstructs both architectures, loads their weights, computes logits for the same preprocessed image, averages the logits, and applies softmax to obtain class probabilities.

### E. Explainable AI Using Grad-CAM

For the selected class, the application performs a gradient-enabled forward pass through ResNet18, captures activations and gradients from `layer4[-1].conv2`, computes channel weights from spatially averaged gradients, forms a rectified weighted activation map, and resizes it to 224×224. The interface displays the original MRI beside the class-specific model-focus overlay.

### F. Tumor Segmentation Using U-Net

The custom U-Net has encoder channels 3→32→64→128→256, a 512-channel bottleneck, and decoder channels 256→128→64→32. A one-channel sigmoid output is thresholded at 0.5 to form a binary mask. The red segmentation overlay blends the mask with the resized MRI image.

### G. Tumor Localization and Centroid Extraction

The system calculates tumor area as the percentage of positive pixels in the 256×256 mask. It obtains the bounding box from the minimum and maximum positive x/y coordinates and computes the centroid as the midpoint of that box. If no positive pixels are found, the center of the image is used and the bounding box is zeroed.

### H. Functional Brain Region Mapping

The centroid is normalized by the 256-pixel segmentation size. Vertical and horizontal thresholds assign a broad region. Pituitary predictions are directly assigned to the pituitary/sellar region because tumor class is used as an additional rule.

### I. Potential Functional Impact Assessment

Each mapped region has descriptive functions and possible impacts, including executive and motor functions for the frontal lobe, sensory and spatial processing for the parietal lobe, memory and language for the temporal lobe, visual processing for the occipital lobe, balance for the cerebellum, vital functions for the brainstem, and hormonal/visual-pathway functions for the pituitary region.

## IV. System Architecture and Implementation

### A. Overall System Workflow

```text
MRI upload → RGB conversion and resize → ensemble classification
          → predicted class and probabilities
          → ResNet18 Grad-CAM visualization
          → U-Net segmentation (tumor classes only)
          → area, bounding box, centroid
          → coarse functional mapping and impact assessment
```

### B. Classification Module

`prepare_image_for_classification()`, `load_ensemble_model()`, and `run_classification()` implement preprocessing, cached checkpoint loading, inference, probability calculation, and final class selection.

### C. Segmentation Module

`UNet`, `load_seg_model()`, and `run_segmentation()` implement the segmentation architecture, cached checkpoint loading, mask thresholding, overlay generation, and geometric measurements.

### D. Functional Mapping Module

`functional_mapping.py` contains the region descriptions, `infer_lobe_from_location()` heuristic, and `get_functional_impact()` lookup.

### E. Streamlit-Based Application

`app.py` exposes a Home upload page and an Analysis Results page. The application uses session state for navigation and cached resources for model reuse. Supported uploads include JPG, JPEG, PNG, GIF, and BMP files.

## V. Experimental Setup

### A. Hardware and Software Environment

The application requires Python 3.8 or later, PyTorch, torchvision, Streamlit, NumPy, PIL/Pillow, Matplotlib, OpenCV, and the project checkpoints. It selects CUDA when available and otherwise uses CPU. The included development environment uses a project-local `.venv`.

### B. Training Configuration

Training code and notebook exports are separate from the deployed inference path. The analysis report describes training up to 50 epochs, convergence near epoch 35, and early stopping. Optimizer, learning-rate schedule, augmentation, and seed settings are not enforced by `app.py` and should be taken from the original training records when reproducing the experiment.

### C. Evaluation Metrics

The relevant metrics are overall accuracy, per-class accuracy, precision, recall, F1-score, confusion matrix, segmentation area/mask quality, and qualitative Grad-CAM usefulness. The current repository does not calculate precision, recall, F1, IoU, or Dice during application inference.

## VI. Results and Discussion

### A. Classification Results

The saved classifier checkpoints contain these verified metadata values:

| Model | Reported accuracy |
|---|---:|
| DenseNet121 | 77.75% |
| ResNet18 | 77.38% |
| MobileNetV2 | 77.69% |
| Ensemble | Not stored |

The analysis report separately claims 94.2% for DenseNet121, 93.8% for ResNet18, 91.5% for MobileNetV2, and 95.8% ± 1.2% for an ensemble. These values do not match the checkpoint metadata and cannot be verified from the current executable evaluation code. It also reports 739 correct predictions out of 766, or 96.5%; that confusion-matrix result is arithmetically valid but uses a different or undocumented evaluation summary.

### Chart-Ready Reported Metrics

The following tables consolidate the percentages and accuracy values in `RESULTS_ANALYTICS.md`. They are suitable for bar charts, radar charts, ROC/AUC charts, confusion-matrix summaries, and dataset-distribution graphs. They are reported analysis values and are not recalculated by the Streamlit application.

#### Analysis-Report Model Accuracy Comparison (Unverified)

| Model | Accuracy | Relative improvement over baseline |
|---|---:|---:|
| DenseNet121 | 94.2% | 0.0% |
| ResNet18 | 93.8% | -0.4% |
| MobileNetV2 | 91.5% | -2.7% |
| Ensemble | 95.8% | +1.6% |

#### Ensemble Per-Class Metrics

| Class | Accuracy | Precision | Recall / sensitivity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Glioma | 96.4% | 95.8% | 97.1% | 96.4% | 0.989 |
| Meningioma | 95.3% | 94.2% | 95.8% | 95.0% | 0.982 |
| Pituitary | 96.1% | 95.1% | 96.8% | 95.9% | 0.988 |
| No tumor | 95.0% | 92.4% | 94.7% | 93.5% | 0.983 |
| Ensemble summary | 95.8% | 94.5% | 96.2% | 95.3% | 0.985 |

#### Ensemble Summary Statistics

| Metric | Reported value |
|---|---:|
| Accuracy | 95.8% ± 1.2% |
| Precision / PPV | 94.5% ± 1.5% |
| Recall / sensitivity | 96.2% ± 0.9% |
| F1-score | 95.3% ± 1.1% |
| Specificity | 94.7% |
| NPV | 96.1% |
| ROC-AUC | 0.985 |
| Diagnostic odds ratio | 157.8 |

#### Confusion-Matrix and Error Chart Data

| Measure | Count | Percentage |
|---|---:|---:|
| Test samples | 766 | 100.0% |
| Correct predictions | 739 | 96.5% |
| Misclassifications | 27 | 3.5% |

| True class | Correct / total | Class accuracy |
|---|---:|---:|
| Glioma | 185 / 192 | 96.4% |
| Meningioma | 181 / 189 | 95.8% |
| Pituitary | 179 / 185 | 96.8% |
| No tumor | 194 / 200 | 97.0% |

| Error category | Errors | Share of errors |
|---|---:|---:|
| Similar pathology | 12 | 44% |
| Low image quality | 8 | 30% |
| Edge cases | 5 | 19% |
| Labeling ambiguity | 2 | 7% |

#### Dataset Distribution

| Category | Images | Share |
|---|---:|---:|
| Glioma | 898 | 25.0% |
| Meningioma | 898 | 25.0% |
| Pituitary | 898 | 25.0% |
| No tumor | 902 | 25.0% |
| Total | 3,596 | 100.0% |

| Partition | Images | Share |
|---|---:|---:|
| Training | 3,064 | 85.0% |
| Validation | 269 | 7.5% |
| Test | 266 | 7.5% |

The reported partition counts sum to 3,599 rather than 3,596, so the dataset split must be checked before producing a final distribution chart. The 766-sample confusion-matrix evaluation is a separate reported evaluation summary.

#### Training and Cross-Validation Indicators

| Indicator | Reported value |
|---|---:|
| Final training accuracy | 96.8% |
| Final validation accuracy | 95.8% |
| Training-validation gap | 1.0% |
| Cross-validation mean accuracy | 95.8% ± 1.2% |
| Minimum fold accuracy | 93.9% |
| Maximum fold accuracy | 97.1% |
| Approximate convergence | Epoch 35 of 50 |

#### Ensemble and Agreement Analysis

| Comparison | Reported improvement or value |
|---|---:|
| Majority voting improvement | +0.8% |
| Weighted voting improvement | +1.2% |
| Probability averaging improvement | +1.4% |
| Final optimization improvement | +1.6% |
| Model diversity score | 0.825 |
| Error correlation | 0.18 |

| Agreement case | Frequency | Final accuracy |
|---|---:|---:|
| All three models agree, confidence >95% | 89.7% (687/766) | 98.4% |
| Two of three models agree | 9.3% (71/766) | 84.5% |
| All three models disagree | 1.0% (8/766) | 62.5% |

#### Binary Detection Indicators

| Metric | Reported value |
|---|---:|
| Average sensitivity | 96.2% |
| Average specificity | 94.7% |
| Average PPV | 94.5% |
| Average NPV | 96.1% |
| False-negative rate | 3.8% |
| False-positive rate | 5.3% |

These values support chart creation, but they should not be interpreted as independently verified clinical performance. The project contains no executable evaluation script that reproduces all of these metrics from labeled predictions.

### B. Segmentation Results

The application reports tumor area percentage, binary mask, overlay, bounding box, and centroid. No verified Dice, IoU, sensitivity, or specificity value is produced by the current code, so a numerical segmentation accuracy claim is not made here.

### C. Grad-CAM Visualization Results

Grad-CAM is generated for the predicted class from the ResNet18 final convolutional block. The result is a coarse model-focus overlay, not a tumor boundary and not a validated anatomical localization.

### D. Functional Brain Mapping Results

The centroid is converted into a broad functional region using fixed 2-D thresholds. The pituitary class receives a dedicated pituitary/sellar mapping. The interface presents associated functions and possible impacts as contextual information.

### E. Comparative Analysis

The analysis-only ensemble accuracy is reported as 1.6 percentage points higher than the analysis-only DenseNet121 value. The verified checkpoint metadata does not contain an ensemble accuracy, and the individual model values are approximately 77.38% to 77.75%. Comparisons are meaningful only when all models use the same data split and evaluation protocol.

### F. Discussion

The system demonstrates how classification, explanation, segmentation, and functional context can be presented in one workflow. Its strongest engineering feature is the separation between model prediction and supplementary interpretation. Its reported performance is promising for a prototype, but the metric inconsistencies and lack of validated segmentation metrics prevent clinical conclusions.

## VII. Limitations

- **Dataset limitations:** Dataset provenance, patient-level separation, preprocessing history, and exact training configuration are not fully enforced or documented in the application code.
- **Localization limitations:** Grad-CAM is coarse and the segmentation checkpoint may produce imperfect masks. Centroid and bounding-box calculations depend entirely on the predicted mask.
- **Functional mapping limitations:** The mapping uses a 2-D heuristic grid, not 3-D registration, neuroimaging atlas alignment, or validated clinical anatomy.
- **Clinical validation requirements:** The system has not been established as safe or effective for diagnosis. External validation, calibrated probabilities, reader studies, bias analysis, uncertainty estimates, and regulatory review are required.

## VIII. Conclusion and Future Work

NeuroScan.ai provides an end-to-end prototype for four-class brain MRI classification with ensemble inference, Grad-CAM explanation, U-Net segmentation, geometric localization, and coarse functional impact assessment. Verified checkpoint metadata reports 77.75% for DenseNet121, 77.38% for ResNet18, and 77.69% for MobileNetV2; no ensemble accuracy is stored. The analysis artifact contains higher, incompatible accuracy claims and a separate 96.5% confusion-matrix calculation, all of which require reconciliation.

Future work should establish a reproducible patient-level dataset split, expose the MobileNetV2 model consistently in the runtime ensemble, add Dice and IoU evaluation for segmentation, calibrate confidence scores, support DICOM and multi-slice/3-D input, replace heuristic mapping with registered anatomical localization, and conduct external clinical validation.

## References

1. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” *MICCAI*, 2015.
2. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *CVPR*, 2016.
3. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” *CVPR*, 2017.
4. M. Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *CVPR*, 2018.
5. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *ICCV*, 2017.
6. F. Isensee et al., “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” *Nature Methods*, 2021.

## Medical Disclaimer

NeuroScan.ai is a research and educational prototype. It is not a substitute for a qualified radiologist, neurologist, oncologist, or other medical professional. Predictions, visualizations, segmentation results, and functional-impact descriptions must not be used alone for diagnosis or treatment decisions.
