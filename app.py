import os
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["toolbar"] = "none"

from functional_mapping import infer_lobe_from_location, get_functional_impact

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan.ai",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    background-color: #ffffff !important;
    color: #1a1a1a;
    font-family: "Source Sans Pro", sans-serif;
}
header { background: #ffffff !important; }
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
    border-right: none !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.5rem; }
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
.nav-label {
    font-size: 0.85rem; font-weight: 600; color: #444 !important;
    margin-bottom: 0.6rem; letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio button,
[data-testid="stSidebar"] .stRadio [role="radio"] {
    font-size: 1rem !important; font-weight: 400 !important;
    padding: 8px 10px !important; color: #1a1a1a !important;
    cursor: pointer; border-radius: 10px !important;
    transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
}
[data-testid="stSidebar"] .stRadio button:hover,
[data-testid="stSidebar"] .stRadio label:hover,
[data-testid="stSidebar"] .stRadio [role="radio"]:hover { background: #e8f2ff !important; }
[data-testid="stSidebar"] .stRadio button[aria-checked="true"],
[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"],
[data-testid="stSidebar"] .stRadio label[selected] {
    background: #3a8fe8 !important; color: #ffffff !important;
    border-color: #3a8fe8 !important;
}
[data-testid="stSidebar"] .stRadio button[aria-checked="true"] svg,
[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] svg { fill: #ffffff !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { font-size: 1rem !important; }
.main .block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1100px; }
.page-title { font-size: 2.8rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.5rem; line-height: 1.2; }
.page-desc { font-size: 1.05rem; color: #444; margin-bottom: 1.8rem; line-height: 1.6; }
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #1a1a1a;
    margin: 1.2rem 0 0.5rem 0; padding-top: 0.5rem; border-top: 1px solid #eee;
}
.body-text { font-size: 1rem; color: #333; line-height: 1.7; margin-bottom: 1rem; }
.bullet-list { font-size: 1rem; color: #333; line-height: 2; padding-left: 1.2rem; }
.upload-label { font-size: 1rem; color: #444; margin-bottom: 0.4rem; }
[data-testid="stFileUploader"] {
    border-radius: 8px !important; background: #f9f9f9 !important; padding: 2rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #f0f0f0 !important; border-radius: 8px !important; padding: 2rem !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #333 !important; }
[data-testid="stFileUploader"] { color: #1a1a1a !important; }
[data-testid="stFileUploader"] * { color: #1a1a1a !important; }
[data-testid="stFileUploader"] label { color: #1a1a1a !important; }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div { color: #1a1a1a !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploadButton"] button {
    background: #3a8fe8 !important; color: #ffffff !important;
    border: 1px solid #3a8fe8 !important; padding: 0.8rem 1.3rem !important;
    border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(58,143,232,0.18) !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploadButton"] button:hover {
    background: #2f78ce !important; border-color: #2f78ce !important;
}
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] svg {
    background: transparent !important; color: #1a1a1a !important; fill: #1a1a1a !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button svg,
[data-testid="stFileUploadButton"] button,
[data-testid="stFileUploadButton"] button svg {
    background: #3a8fe8 !important; color: #ffffff !important; fill: #ffffff !important;
}
.result-card {
    background: #f8f9fa; border-radius: 10px; padding: 1.5rem 2rem;
    border-left: 5px solid #e05c5c; margin: 1.2rem 0;
}
.result-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
.result-value { font-size: 1.6rem; font-weight: 700; color: #1a1a1a; text-transform: capitalize; }
.result-conf { font-size: 0.95rem; color: #555; margin-top: 0.3rem; }
.summary-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem; margin-top: 1rem;
}
.summary-card, .symptom-card {
    background: #ffffff; border: 1px solid #e7e7e7; border-radius: 14px;
    padding: 1.2rem 1.3rem; box-shadow: 0 1px 6px rgba(20,30,60,0.05);
}
.summary-title, .symptom-title { font-size: 0.95rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.65rem; }
.summary-text, .symptom-text { font-size: 0.95rem; color: #424242; line-height: 1.75; }
.symptom-card ul { margin: 0.5rem 0 0 1.1rem; padding: 0; }
.symptom-card li { margin-bottom: 0.45rem; color: #333; }
.func-card {
    background: #f0f7ff; border: 1px solid #c3daf9; border-radius: 14px;
    padding: 1.2rem 1.3rem; box-shadow: 0 1px 6px rgba(20,30,60,0.05);
}
.func-title { font-size: 0.95rem; font-weight: 700; color: #1a4a8a; margin-bottom: 0.65rem; }
.func-text { font-size: 0.95rem; color: #1a3a6a; line-height: 1.75; }
.func-card ul { margin: 0.5rem 0 0 1.1rem; padding: 0; }
.func-card li { margin-bottom: 0.45rem; color: #1a3a6a; }
.probability-panel {
    background: #ffffff; border: 1px solid #e7e7e7; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.8rem 0 1.2rem;
}
.probability-heading {
    color: #1a1a1a; font-size: 0.95rem; font-weight: 700;
    margin-bottom: 0.9rem;
}
.probability-row { margin-bottom: 0.75rem; }
.probability-row:last-child { margin-bottom: 0; }
.probability-label {
    display: flex; justify-content: space-between; align-items: center;
    color: #333; font-size: 0.88rem; margin-bottom: 0.3rem;
}
.probability-value { color: #1a1a1a; font-weight: 700; }
.probability-track {
    height: 8px; background: #edf0f4; border-radius: 999px; overflow: hidden;
}
.probability-fill { height: 100%; border-radius: 999px; }
.disclaimer {
    background: #fff8e1; border-left: 4px solid #ffc107; border-radius: 6px;
    padding: 0.9rem 1.2rem; font-size: 0.88rem; color: #5d4037; margin-top: 1.5rem;
}
.func-disclaimer {
    background: #e8f4fd; border-left: 4px solid #3a8fe8; border-radius: 6px;
    padding: 0.9rem 1.2rem; font-size: 0.85rem; color: #1a3a6a; margin-top: 1rem;
}
[data-testid="stImage"] { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stImage"] img {
    display: block !important; width: 100% !important; max-width: 300px !important;
    max-height: 300px !important; height: auto !important; object-fit: contain !important;
    margin: 0 auto !important;
}
[data-testid="stButton"] button,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploadButton"] button {
    padding: 0.45rem 0.9rem !important;
    min-height: 2.25rem !important;
}
[data-testid="stButton"] button {
    background: #3a8fe8 !important; color: #ffffff !important;
    border: 1px solid #3a8fe8 !important; border-radius: 8px !important;
}
[data-testid="stButton"] button:hover {
    background: #2f78ce !important; border-color: #2f78ce !important;
}
[data-testid="stButton"] button:focus {
    color: #ffffff !important; box-shadow: 0 0 0 2px rgba(58,143,232,0.25) !important;
}
[data-testid="stImage"] button,
[data-testid="stImage"] svg,
[data-testid="stImage"] [role="button"],
[data-testid="stPlotContainer"] button,
[data-testid="stPlotContainer"] svg,
[data-testid="stPlotContainer"] [role="button"] {
    display: none !important; opacity: 0 !important;
    visibility: hidden !important; pointer-events: none !important;
}
[data-testid="column"] { background: transparent !important; }
[data-testid="stElementContainer"] { background: transparent !important; color: #1a1a1a !important; }
[data-testid="stElementContainer"] * { color: #1a1a1a !important; }
.info-block { border-bottom: 1px solid #eee; padding: 1rem 0; }
[data-testid="stMarkdownContainer"] { color: #1a1a1a !important; }
[data-testid="stMarkdownContainer"] * { color: #1a1a1a !important; }
.footer { position: fixed; bottom: 1rem; left: 0; width: 260px; text-align: center; font-size: 0.78rem; color: #aaa; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CLASS_COLORS = {
    "glioma":     "#e05c5c",
    "meningioma": "#e8913a",
    "no tumor":   "#3aad6e",
    "pituitary":  "#3a8fe8",
}
CLASS_INFO = {
    "glioma": {
        "severity": "High",
        "desc": "A tumor that originates in the glial cells of the brain or spine. Gliomas account for about 33% of all brain tumors and are the most common malignant primary brain tumors.",
        "symptoms": "Headaches, seizures, memory loss, personality changes, vision or speech problems.",
    },
    "meningioma": {
        "severity": "Moderate",
        "desc": "Tumors that arise from the meninges — the membranes surrounding the brain and spinal cord. Most meningiomas are benign and grow slowly.",
        "symptoms": "Headaches, weakness in arms or legs, seizures, personality changes, vision problems.",
    },
    "pituitary": {
        "severity": "Low–Moderate",
        "desc": "Tumors that form in the pituitary gland at the base of the brain. Most pituitary tumors are benign (non-cancerous) adenomas.",
        "symptoms": "Headaches, vision problems, hormonal imbalance, fatigue, unexplained weight changes.",
    },
    "no tumor": {
        "severity": "None",
        "desc": "No tumor detected in the MRI scan. The brain tissue appears normal with no signs of abnormal growth.",
        "symptoms": "N/A",
    },
}
IMG_SIZE = 224
SEG_SIZE = 256

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Device Setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── UNet Architecture ───────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """UNet segmentation model matching segmentation_model.pth checkpoint."""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder blocks
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoder with upsampling
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        # Output layer
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        enc4_out = self.enc4(x)
        x = self.pool4(enc4_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up4(x)
        x = torch.cat([x, enc4_out], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        return x

# ─── Model Loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_seg_model():
    """Load trained segmentation model from checkpoint."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join(BASE_DIR, "saved_models", "segmentation_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    return model

@st.cache_resource
def load_ensemble_model():
    """Load trained ensemble model from checkpoint."""
    model_path = os.path.join(BASE_DIR, "saved_models", "ensemble_model.pth")
    if os.path.exists(model_path):
        ensemble = torch.load(model_path, map_location=device)
        if isinstance(ensemble, dict):
            resnet = models.resnet18(weights=None)
            resnet.fc = nn.Sequential(nn.Identity(), nn.Linear(512, len(CLASS_NAMES)))
            resnet.load_state_dict(ensemble["resnet"])
            resnet = resnet.to(device).eval()

            densenet = models.densenet121(weights=None)
            densenet.classifier = nn.Sequential(nn.Identity(), nn.Linear(1024, len(CLASS_NAMES)))
            densenet.load_state_dict(ensemble["densenet"])
            densenet = densenet.to(device).eval()

            return {"resnet": resnet, "densenet": densenet}
        return ensemble.to(device).eval()
    return None

def prepare_image_for_classification(image):
    """Prepare image for classification model (224x224, ImageNet norm)."""
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = ((img_array - mean) / std).astype(np.float32)
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img

def run_classification(image):
    """Run ensemble classification on input image."""
    img_tensor, processed_img = prepare_image_for_classification(image)
    
    model = load_ensemble_model()
    if model is None:
        # Fallback: return placeholder
        return {
            "class": "unknown",
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in CLASS_NAMES}
        }
    
    with torch.no_grad():
        if isinstance(model, dict):
            outputs = [classifier(img_tensor) for classifier in model.values()]
            output = torch.stack(outputs).mean(dim=0)
        else:
            output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": prob_dict
    }

def generate_gradcam(image, model, class_idx=None):
    """Generate a Grad-CAM heatmap from the ResNet feature extractor."""
    img_tensor, _ = prepare_image_for_classification(image)

    if isinstance(model, dict):
        model = model.get("resnet")
    if model is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    feature_maps = []
    gradients = []

    def save_features(_, __, output):
        feature_maps.append(output)

    def save_gradients(_, __, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1].conv2
    forward_handle = target_layer.register_forward_hook(save_features)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        output = model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

    forward_handle.remove()
    backward_handle.remove()

    activations = feature_maps[0].detach()
    layer_gradients = gradients[0].detach()
    weights = layer_gradients.mean(dim=(2, 3), keepdim=True)
    heatmap = (weights * activations).sum(dim=1).squeeze(0).relu().cpu().numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    return heatmap

# ─── Segmentation Inference ─────────────────────────────────────────────────
def run_segmentation(image):
    """
    Run segmentation on input image.
    
    Preprocessing: 256x256 (separate from classification preprocessing).
    Returns: binary mask, overlay, tumor area %, bbox, centroid.
    """
    model = load_seg_model()
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to 256x256 for segmentation
    seg_img = image.resize((SEG_SIZE, SEG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(seg_img, dtype=np.float32)
    
    # Normalize to [0, 1]
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).cpu().numpy()
    
    # Threshold at 0.5
    binary_mask = (mask[0, 0] > 0.5).astype(np.uint8) * 255
    
    # Calculate tumor area percentage
    tumor_pixels = np.sum(binary_mask > 0)
    total_pixels = SEG_SIZE * SEG_SIZE
    tumor_area_pct = (tumor_pixels / total_pixels) * 100
    
    # Find bounding box
    points = np.where(binary_mask > 0)
    if len(points[0]) > 0:
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        bbox = (x_min, y_min, x_max, y_max)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
    else:
        bbox = (0, 0, 0, 0)
        cx, cy = SEG_SIZE / 2, SEG_SIZE / 2
    
    # Create overlay
    overlay = np.array(seg_img, dtype=np.float32)
    mask_color = np.array([255, 0, 0], dtype=np.float32)  # Red
    overlay[binary_mask > 0] = overlay[binary_mask > 0] * 0.5 + mask_color * 0.5
    overlay = overlay.astype(np.uint8)
    
    return {
        "mask": binary_mask,
        "overlay": overlay,
        "area_pct": tumor_area_pct,
        "bbox": bbox,
        "centroid": (cx, cy),
    }

# ─── Main Streamlit App ──────────────────────────────────────────────────────
def page_home():
    """Home page: Upload MRI scan for analysis."""
    st.markdown('<h1 class="page-title">NeuroScan.ai</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">AI-powered brain tumor detection and analysis</p>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-title">Upload MRI Scan</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="Choose MRI image",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Store image in session state and switch page
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.session_state.current_page = "Grad-CAM Viewer"
        st.rerun()

def page_gradcam():
    """Grad-CAM Viewer: Display analysis results."""
    st.markdown('<h1 class="page-title">Analysis Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Detailed tumor analysis and functional impact assessment</p>', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Upload", key="back_to_home"):
        st.session_state.current_page = "Home"
        st.rerun()
    
    st.divider()
    
    if "uploaded_image" not in st.session_state:
        st.warning("No image uploaded. Please go to Home and upload an MRI scan.")
        return
    
    image = st.session_state.uploaded_image
    
    # Display original image
    st.markdown('<h2 class="section-title">Original MRI Scan</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(image, width="stretch")
    
    # Run classification
    with st.spinner("🧠 Analyzing tumor classification..."):
        classification_result = run_classification(image)
    
    pred_class = classification_result["class"]
    confidence = classification_result["confidence"]
    probabilities = classification_result["probabilities"]
    
    # Display classification result
    st.markdown('<h2 class="section-title">Tumor Classification</h2>', unsafe_allow_html=True)
    
    color = CLASS_COLORS.get(pred_class, "#999999")
    result_html = f"""
    <div class="result-card" style="border-left-color: {color};">
        <div class="result-label">Predicted Class</div>
        <div class="result-value" style="color: {color};">{pred_class.upper()}</div>
        <div class="result-conf">Confidence: {confidence*100:.1f}%</div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
    
    # Display classification probabilities as a compact visual summary
    probability_rows = "".join(
        f'''<div class="probability-row">
            <div class="probability-label">
                <span>{class_name.capitalize()}</span>
                <span class="probability-value">{prob * 100:.1f}%</span>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: {prob * 100:.1f}%; background: {CLASS_COLORS.get(class_name, "#3a8fe8")};"></div>
            </div>
        </div>'''
        for class_name, prob in probabilities.items()
    )
    st.markdown(
        f'''<div class="probability-panel">
            <div class="probability-heading">Classification Probabilities</div>
            {probability_rows}
        </div>''',
        unsafe_allow_html=True,
    )
    
    with st.spinner("Generating Grad-CAM visualization..."):
        gradcam_heatmap = generate_gradcam(
            image,
            load_ensemble_model(),
            class_idx=CLASS_NAMES.index(pred_class),
        )

    st.markdown('<h2 class="section-title">Grad-CAM Visualization</h2>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(image)
    axes[0].set_title("Original MRI", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(gradcam_heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Model Focus", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    plt.tight_layout()
    st.pyplot(fig, width="content")
    
    # Run segmentation only if not "no tumor"
    if pred_class != "no tumor":
        with st.spinner("🔍 Running tumor segmentation..."):
            seg_result = run_segmentation(image)
        
        st.markdown('<h2 class="section-title">Tumor Segmentation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(seg_result["overlay"], width="stretch", caption="Segmentation Overlay (Red)")
        
        with col2:
            st.metric("Tumor Area", f"{seg_result['area_pct']:.2f}%")
            cx, cy = seg_result["centroid"]
            st.write(f"**Centroid:** ({cx:.2f}, {cy:.2f})")
            st.write(f"**Bounding Box:** {seg_result['bbox']}")
        
        # Functional mapping
        lobe = infer_lobe_from_location("unknown", pred_class, cx, cy, img_size=256)
        func_impact = get_functional_impact(lobe)
        
        st.markdown('<h2 class="section-title">Functional Impact Assessment</h2>', unsafe_allow_html=True)
        
        st.markdown(f"### {func_impact['functional_area']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Associated Functions:")
            for func in func_impact['functions']:
                st.write(f"• {func}")
        
        with col2:
            st.markdown("#### Potential Impacts:")
            for impact in func_impact['potential_impacts']:
                st.write(f"• {impact}")
        
        st.markdown('<div class="func-disclaimer"><b>Clinical Note:</b> This assessment is AI-generated. Always consult with qualified medical professionals for diagnosis and treatment planning.</div>', unsafe_allow_html=True)
    
    else:
        st.success("✅ No tumor detected. Brain tissue appears normal.")
    
    # Tumor information
    if pred_class in CLASS_INFO:
        st.markdown('<h2 class="section-title">Tumor Information</h2>', unsafe_allow_html=True)
        info = CLASS_INFO[pred_class]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="summary-title">Severity Level</div>
                <div class="summary-text">{info['severity']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="summary-card">
                <div class="summary-title">Description</div>
                <div class="summary-text">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="symptom-card">
            <div class="symptom-title">Common Symptoms</div>
            <div class="symptom-text">{info['symptoms']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="disclaimer"><b>⚠️ Medical Disclaimer:</b> This AI system is a diagnostic aid only and should not be used as a substitute for professional medical judgment. All results must be reviewed by a qualified radiologist or oncologist.</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application with single-page conditional rendering."""
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation - just display current page
    st.sidebar.markdown('<p class="nav-label">NAVIGATION</p>', unsafe_allow_html=True)
    st.sidebar.markdown(f"**Current: {st.session_state.current_page}**")
    st.sidebar.divider()
    
    # Render appropriate page
    if st.session_state.current_page == "Home":
        page_home()
    else:
        page_gradcam()

# ─── Run App ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
