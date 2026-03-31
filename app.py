import os
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan.ai",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS — matches PneumoScan style exactly ───────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
html, body, [class*="css"], .stApp {
    background-color: #ffffff !important;
    color: #1a1a1a;
    font-family: "Source Sans Pro", sans-serif;
}

/* Streamlit header/toolbar */
header {
    background: #ffffff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
    border-right: none !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.5rem;
}
[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}

/* Nav label "Navigation" */
.nav-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #444 !important;
    margin-bottom: 0.6rem;
    letter-spacing: 0.02em;
}

/* Radio buttons — match PneumoScan style */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem !important;
    font-weight: 400 !important;
    padding: 4px 0 !important;
    color: #1a1a1a !important;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 1rem !important;
}

/* ── Main content area ── */
.main .block-container {
    padding: 3rem 4rem 3rem 4rem;
    max-width: 1100px;
}

/* ── Page title (big bold like PneumoScan.ai) ── */
.page-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

/* ── Subtitle / description text ── */
.page-desc {
    font-size: 1.05rem;
    color: #444;
    margin-bottom: 1.8rem;
    line-height: 1.6;
}

/* ── Section headings ── */
.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 1.8rem 0 0.6rem 0;
}

/* ── Body text ── */
.body-text {
    font-size: 1rem;
    color: #333;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* ── Bullet list ── */
.bullet-list {
    font-size: 1rem;
    color: #333;
    line-height: 2;
    padding-left: 1.2rem;
}

/* ── Upload label ── */
.upload-label {
    font-size: 1rem;
    color: #444;
    margin-bottom: 0.4rem;
}

/* ── File uploader — match drag-drop box ── */
[data-testid="stFileUploader"] {
    border-radius: 8px !important;
    background: #f9f9f9 !important;
    padding: 2rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f0f0f0 !important;
    border-radius: 8px !important;
    padding: 2rem !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #333 !important;
}

/* File uploader container and text */
[data-testid="stFileUploader"] {
    color: #1a1a1a !important;
}

[data-testid="stFileUploader"] * {
    color: #1a1a1a !important;
}

[data-testid="stFileUploader"] label {
    color: #1a1a1a !important;
}

[data-testid="stFileUploader"] p, 
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {
    color: #1a1a1a !important;
}

/* File upload "Browse" button (inside uploader container) */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploadButton"] button {
    background: #3a8fe8 !important;
    color: #ffffff !important;
    border: 1px solid #3a8fe8 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* ── Result card ── */
.result-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    border-left: 5px solid #e05c5c;
    margin: 1.2rem 0;
}
.result-label {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.result-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1a1a;
    text-transform: capitalize;
}
.result-conf {
    font-size: 0.95rem;
    color: #555;
    margin-top: 0.3rem;
}

/* ── Probability bar labels ── */
.prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #333;
    margin-bottom: 2px;
}

/* ── Disclaimer box ── */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #5d4037;
    margin-top: 1.5rem;
}

/* ── Image container — remove background boxes ── */
[data-testid="stImage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stImage"] img {
    display: block !important;
    width: 100% !important;
    height: auto !important;
}

/* Remove background from all containers */
[data-testid="column"] {
    background: transparent !important;
}

/* Remove element container backgrounds */
[data-testid="stElementContainer"] {
    background: transparent !important;
    color: #1a1a1a !important;
}

/* Ensure all text is visible */
[data-testid="stElementContainer"] * {
    color: #1a1a1a !important;
}

/* ── Info card (about page) ── */
.info-block {
    border-bottom: 1px solid #eee;
    padding: 1rem 0;
}

/* Markdown containers */
[data-testid="stMarkdownContainer"] {
    color: #1a1a1a !important;
}

[data-testid="stMarkdownContainer"] * {
    color: #1a1a1a !important;
}

/* ── Made with footer ── */
.footer {
    position: fixed;
    bottom: 1rem;
    left: 0;
    width: 260px;
    text-align: center;
    font-size: 0.78rem;
    color: #aaa;
}

/* Hide streamlit branding */
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

# ─── Model ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_model(name: str, num_classes: int) -> nn.Module:
    """
    Match the Colab architecture:
    - Load ImageNet pretrained backbone
    - Freeze backbone parameters
    - Replace final classifier head with Dropout + Linear(num_classes)
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.fc.in_features, num_classes),
        )
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier.in_features, num_classes),
        )
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model architecture: {name}")
    return m

def discover_weight_files() -> list:
    """
    Discover available weight files from project directories.
    Prefer `saved_models/` if present, otherwise fall back to `models/`.
    """
    candidates = [
        os.path.join(BASE_DIR, "saved_models"),
        os.path.join(BASE_DIR, "models"),
    ]

    all_files = []
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".pth"):
                    all_files.append(os.path.join(d, f))

    brain_tumor_files = [
        p for p in all_files if "_brain_tumor" in os.path.basename(p).lower()
    ]

    preferred = sorted(brain_tumor_files) if brain_tumor_files else []
    return preferred if preferred else sorted(all_files)

@st.cache_resource
def load_model(path: str):
    """
    Returns: (model, model_name, class_names)
    """
    default_model_name = "resnet18"
    default_class_names = CLASS_NAMES

    model_name = default_model_name
    class_names = default_class_names
    num_classes = len(default_class_names)

    # Build something deterministic even if weights are missing
    m = build_model(model_name, num_classes)

    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        filename = os.path.basename(path).lower()

        def infer_model_name_from_filename() -> str:
            if "mobilenet" in filename:
                return "mobilenet_v2"
            if "densenet" in filename:
                return "densenet121"
            if "resnet" in filename:
                return "resnet18"
            return default_model_name

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt.get("state_dict")
            model_name = ckpt.get("model_name") or infer_model_name_from_filename()
            class_names = ckpt.get("class_names", class_names)
            num_classes = int(ckpt.get("num_classes", len(class_names)))
        else:
            state_dict = ckpt
            model_name = infer_model_name_from_filename()

        m = build_model(model_name, num_classes)
        m.load_state_dict(state_dict)

    m.eval()
    return m, model_name, class_names

def preprocess(img: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)

# Grad-CAM
def _get_gradcam_target_layer(model: nn.Module, model_name: str):
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    return model.layer4[-1]

def gradcam(model, tensor, class_idx, model_name: str):
    tl = _get_gradcam_target_layer(model, model_name)
    grads, acts = [], []
    fh = tl.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    bh = tl.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach()))
    for p in model.parameters():
        p.requires_grad = True
    model.zero_grad()
    out = model(tensor)
    out[0, class_idx].backward()
    fh.remove(); bh.remove()
    w   = grads[0][0].mean(dim=(1, 2))
    c   = torch.relu((w[:, None, None] * acts[0][0]).sum(0)).numpy()
    return (c - c.min()) / (c.max() - c.min() + 1e-8)

def overlay(img, cam):
    base = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    camr = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    
    # Apply jet colormap without matplotlib to avoid figure artifacts
    jet_colors = np.array([
        [0, 0, 1], [0, 0.5, 1], [0, 1, 1], [0.5, 1, 0.5],
        [1, 1, 0], [1, 0.5, 0], [1, 0, 0]
    ])
    idx = (camr * 6).astype(int).clip(0, 5)
    frac = (camr * 6) - idx
    
    jet_cam = np.zeros((*camr.shape, 3))
    for i in range(camr.shape[0]):
        for j in range(camr.shape[1]):
            c1 = jet_colors[idx[i, j]]
            c2 = jet_colors[min(idx[i, j] + 1, 6)]
            jet_cam[i, j] = c1 * (1 - frac[i, j]) + c2 * frac[i, j]
    
    return np.clip(0.55 * base + 0.45 * jet_cam, 0, 1)

# ─── Ensemble Model ───────────────────────────────────────────────────────────
def load_ensemble_model():
    """
    Load all three models and return them as a list.
    """
    model_paths = discover_weight_files()
    models = []

    for path in model_paths:
        model, model_name, _ = load_model(path)
        models.append((model, model_name))

    return models

def ensemble_predict(models, tensor):
    """
    Perform ensemble prediction by averaging softmax probabilities.

    Args:
        models (list): List of (model, model_name) tuples.
        tensor (torch.Tensor): Preprocessed input image tensor.

    Returns:
        tuple: (predicted_class, confidence_score)
    """
    softmax = nn.Softmax(dim=1)
    ensemble_probs = None

    for model, _ in models:
        with torch.no_grad():
            logits = model(tensor)
            probs = softmax(logits)

            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs

    # Average the probabilities
    ensemble_probs /= len(models)

    # Get the predicted class and confidence score
    confidence, predicted_class = torch.max(ensemble_probs, dim=1)
    return predicted_class.item(), confidence.item()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=["About NeuroScan.ai", "Tumor Detection", "Grad-CAM Viewer"],
        label_visibility="collapsed"
    )

# Ensemble loading (no model settings UI)
weight_files = discover_weight_files()
densenet_path = next((p for p in weight_files if "densenet121" in os.path.basename(p).lower() or "densenet" in os.path.basename(p).lower()), "")
mobilenet_path = next((p for p in weight_files if "mobilenet_v2" in os.path.basename(p).lower() or "mobilenet" in os.path.basename(p).lower()), "")

if not densenet_path:
    st.error("Could not find `densenet121` weights in `models/` or `saved_models/`.")
    st.stop()

model_a, model_a_name, loaded_class_names = load_model(densenet_path)
model_b = None
model_b_name = None

if mobilenet_path and os.path.exists(mobilenet_path):
    model_b, model_b_name, _ = load_model(mobilenet_path)

CLASS_NAMES = loaded_class_names


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
if page == "About NeuroScan.ai":

    st.markdown("<div class='page-title'>NeuroScan.ai</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='page-desc'>
        Welcome to NeuroScan.ai, a platform designed to assist in the detection and classification of brain tumors from MRI scans. 
        Leveraging state-of-the-art deep learning models and ensemble methods, NeuroScan.ai analyzes medical imaging data to identify potential tumor presence with high accuracy. 
        Our technology combines multiple neural networks including DenseNet, ResNet, and MobileNet to provide robust, reliable predictions. 
        Whether you're a healthcare professional, radiologist, or researcher, NeuroScan.ai serves as a valuable second opinion tool to support diagnostic workflows and improve patient outcomes.
    </div>
    """, unsafe_allow_html=True)

    # ── What is a Brain Tumor? ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>What is a Brain Tumor?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='body-text'>
        A brain tumor is an abnormal mass or growth of cells in the brain. Because the skull is rigid and cannot expand, 
        any growth inside it can cause significant health complications. Brain tumors can originate from brain cells (primary tumors) 
        or spread from cancer cells in other parts of the body (secondary tumors).
        <br><br>
        <strong>Key Characteristics:</strong>
        <ul style='margin-left: 1.2rem; line-height: 1.8;'>
            <li><strong>Benign vs. Malignant:</strong> Benign tumors are non-cancerous and typically grow slowly, while malignant tumors are cancerous and can spread rapidly.</li>
            <li><strong>Growth Rate:</strong> Tumors vary from slow-growing to aggressive, depending on type and grade.</li>
            <li><strong>Impact:</strong> Even benign tumors can cause serious complications due to space constraints within the skull.</li>
            <li><strong>Location Matters:</strong> The location of the tumor determines which brain functions may be affected.</li>
        </ul>
        Brain tumors represent a significant health challenge, affecting both adults and children. Early detection through advanced imaging techniques like MRI 
        combined with AI analysis can significantly improve treatment outcomes and patient prognosis.
    </div>
    """, unsafe_allow_html=True)



    # ── Tumor Types ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Tumor Types Detected</div>", unsafe_allow_html=True)

    sev_color = {"High": "#e05c5c", "Moderate": "#e8913a",
                 "Low–Moderate": "#3a8fe8", "None": "#3aad6e"}

    for cls, info in CLASS_INFO.items():
        col = sev_color[info["severity"]]
        st.markdown(f"""
        <div class='info-block'>
            <div style='font-size:1.05rem; font-weight:700; color:#1a1a1a; margin-bottom:0.2rem;'>
                {cls.title()}
                <span style='font-size:0.78rem; font-weight:600; color:{col};
                             background:{col}18; padding:2px 10px; border-radius:20px;
                             margin-left:8px;'>Severity: {info["severity"]}</span>
            </div>
            <div style='font-size:0.95rem; color:#444; line-height:1.6;'>{info["desc"]}</div>
            <div style='font-size:0.85rem; color:#888; margin-top:0.3rem;'>
                <strong>Symptoms:</strong> {info["symptoms"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <strong>Disclaimer:</strong> NeuroScan.ai is intended for educational and research
        purposes only. It is <em>not</em> a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TUMOR DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tumor Detection":

    st.markdown("<div class='page-title'>Tumor Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Upload an MRI scan to detect brain tumors using our ensemble model.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        img = Image.open(uploaded_file)

        # Preprocess the image
        tensor = preprocess(img)

        # Load ensemble models
        models = load_ensemble_model()

        # Perform ensemble prediction
        predicted_class, confidence = ensemble_predict(models, tensor)

        # Display image and results side by side
        class_name = CLASS_NAMES[predicted_class]
        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.image(img, width="stretch")

        with col2:
            st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='result-card' style='margin-top:0.5rem;'>
                <div class='result-label'>Prediction</div>
                <div class='result-value'>{class_name}</div>
                <div class='result-conf'>Confidence: {confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Ensure 'info' is defined before use
            if class_name in CLASS_INFO:
                info = CLASS_INFO[class_name]
                st.markdown(f"""
                <div style='font-size:0.95rem; color:#333; line-height:1.7; margin-top:1rem;'>
                    <strong>Severity:</strong> {info['severity']}<br><br>
                    <strong>Description:</strong> {info['desc']}<br><br>
                    <strong>Symptoms:</strong> {info['symptoms']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Class information not found.")

    else:
        st.markdown("""
        <div style='color:#aaa; margin-top:3rem; font-size:0.95rem;'>
            Upload an MRI image above to generate Grad-CAM heatmaps.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Grad-CAM Viewer":

    st.markdown("<div class='page-title'>Grad-CAM Viewer</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='page-desc'>
        Upload an MRI image to visualise which regions of the brain the model focused on
        when making its prediction. Warmer colours (red/yellow) indicate higher importance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='upload-label'>Choose an image…</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="gcam"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Computing Grad-CAM…"):
            with torch.no_grad():
                tensor = preprocess(image)
                probs_a = torch.softmax(model_a(tensor), dim=1).numpy()[0]
                if model_b is not None:
                    probs_b = torch.softmax(model_b(tensor), dim=1).numpy()[0]
                    probs = (probs_a + probs_b) / 2.0
                else:
                    probs = probs_a
            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]

        # Original vs best-class CAM side by side
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='font-weight:600; margin-bottom:0.4rem;'>Original MRI</div>",
                        unsafe_allow_html=True)
            st.image(image.resize((IMG_SIZE, IMG_SIZE)), width="stretch")

        with c2:
            st.markdown(
                f"<div style='font-weight:600; margin-bottom:0.4rem;'>"
                f"Grad-CAM → {pred_class.title()} ({probs[pred_idx]*100:.1f}%)</div>",
                unsafe_allow_html=True
            )
            try:
                t   = preprocess(image)
                cam_a = gradcam(model_a, t, pred_idx, model_a_name)
                if model_b is not None:
                    cam_b = gradcam(model_b, t, pred_idx, model_b_name)
                    cam = (cam_a + cam_b) / 2.0
                else:
                    cam = cam_a
                ov  = overlay(image, cam)
                plt.close('all')
                st.image(ov, width="stretch")
            except Exception as e:
                st.error(f"Could not compute Grad-CAM: {e}")

        # All 4 class heatmaps
        st.markdown(
            "<div style='font-size:1rem; font-weight:600; margin:1.5rem 0 0.6rem;'>"
            "Heatmaps for All Classes</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        for i, (cls, col) in enumerate(zip(CLASS_NAMES, cols)):
            with col:
                try:
                    t   = preprocess(image)
                    cam_a = gradcam(model_a, t, i, model_a_name)
                    if model_b is not None:
                        cam_b = gradcam(model_b, t, i, model_b_name)
                        cam = (cam_a + cam_b) / 2.0
                    else:
                        cam = cam_a
                    ov  = overlay(image, cam)
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(ov)
                    ax.set_title(
                        f"{cls.title()}\n{probs[i]*100:.1f}%",
                        fontsize=9,
                        fontweight="bold" if i == pred_idx else "normal",
                        color="#e05c5c" if i == pred_idx else "#333"
                    )
                    ax.axis("off")
                    fig.patch.set_facecolor("white")
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig); plt.close()
                except Exception as e:
                    st.image(image.resize((IMG_SIZE, IMG_SIZE)), caption=cls)

        # Result summary
        accent = CLASS_COLORS[pred_class]
        st.markdown(f"""
        <div class='result-card' style='border-left-color:{accent}; margin-top:1.2rem;'>
            <div class='result-label'>Final Prediction</div>
            <div class='result-value'>{pred_class.title()}</div>
            <div class='result-conf'>Confidence: <strong>{probs[pred_idx]*100:.1f}%</strong>
            &nbsp;·&nbsp; Severity: {CLASS_INFO[pred_class]["severity"]}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='color:#aaa; margin-top:3rem; font-size:0.95rem;'>
            Upload an MRI image above to generate Grad-CAM heatmaps.
        </div>
        """, unsafe_allow_html=True)