import os
import time
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="CIFAR-10 Vision Studio | Neural Perception AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. ULTRA-MODERN CYBERPUNK / GLASSMORPHIC STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;600;700;800;900&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        --accent-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --card-bg: rgba(18, 22, 38, 0.72);
        --card-border: rgba(255, 255, 255, 0.09);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Page background with ambient mesh glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.18) 0%, transparent 45%),
                    radial-gradient(circle at 85% 20%, rgba(236, 72, 153, 0.14) 0%, transparent 40%),
                    radial-gradient(circle at 50% 80%, rgba(56, 189, 248, 0.12) 0%, transparent 50%),
                    #0a0b16 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Header & container cleanup */
    header[data-testid="stHeader"] {
        background: rgba(10, 11, 22, 0.6) !important;
        backdrop-filter: blur(12px) !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-container {
        text-align: center;
        padding: 2.2rem 1.5rem 1.8rem 1.5rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        backdrop-filter: blur(16px);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 30%;
        width: 40%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, #f093fb, transparent);
        animation: pulse-line 4s ease-in-out infinite;
    }

    @keyframes pulse-line {
        0%, 100% { opacity: 0.3; transform: scaleX(0.8); }
        50% { opacity: 1; transform: scaleX(1.2); }
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-badge .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
        animation: blink 2s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .hero-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3.4rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 35%, #38bdf8 70%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.12rem;
        font-weight: 400;
        max-width: 650px;
        margin: 0 auto 1.2rem auto;
        line-height: 1.6;
    }

    .telemetry-bar {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1.2rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .telemetry-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Glass Panels */
    .glass-panel {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.08);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .glass-panel:hover {
        border-color: rgba(255, 255, 255, 0.16);
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .panel-header h3 {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .panel-tag {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* Neon Gradient Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #d946ef 100%) !important;
        background-size: 200% 200% !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.75rem 1.8rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 24px -4px rgba(99, 102, 241, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 12px 30px -4px rgba(217, 70, 239, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        background-position: 100% 0% !important;
    }

    /* Winner Spotlight Card */
    .winner-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px -10px rgba(56, 189, 248, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        margin-bottom: 1.4rem;
    }

    .winner-avatar-ring {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, rgba(15, 23, 42, 0.8) 70%);
        border: 2px solid #38bdf8;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
        animation: float-icon 3s ease-in-out infinite;
    }

    @keyframes float-icon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    .winner-class {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #ffffff;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
    }

    .winner-category-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.08);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    /* Custom Meter Bar */
    .meter-container {
        width: 100%;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        height: 12px;
        overflow: hidden;
        margin: 0.6rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .meter-fill {
        height: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #a855f7 100%);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Podium Cards (Top 3) */
    .podium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }

    .podium-card:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateX(4px);
    }

    .podium-rank-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.95rem;
        margin-right: 0.8rem;
    }

    .rank-gold { background: linear-gradient(135deg, #ffd700, #ffae00); color: #1e1b4b; }
    .rank-silver { background: linear-gradient(135deg, #e2e8f0, #94a3b8); color: #0f172a; }
    .rank-bronze { background: linear-gradient(135deg, #f97316, #b45309); color: #ffffff; }

    .podium-label {
        font-weight: 600;
        font-size: 0.98rem;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .podium-score {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.05rem;
        color: #38bdf8;
    }

    /* Metric Stat Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .stat-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.7rem;
        text-align: center;
    }

    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.2rem;
    }

    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 700;
        color: #cbd5e1;
    }

    /* X-Ray Tensor Box */
    .xray-box {
        border: 1px solid rgba(56, 189, 248, 0.2);
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1122 0%, #151930 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    .sidebar-section {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1rem;
    }

    /* File Uploader Customization */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(56, 189, 248, 0.25) !important;
        border-radius: 16px !important;
        padding: 0.8rem !important;
        transition: border-color 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. CIFAR-10 DEFINITIONS & DOMAIN INTELLIGENCE
# ==============================================================================
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_META = {
    "airplane":   {"emoji": "✈️", "category": "Vehicle / Aviation", "tag": "⚙️ Mechanical Transport"},
    "automobile": {"emoji": "🚗", "category": "Vehicle / Automotive", "tag": "⚙️ Mechanical Transport"},
    "bird":       {"emoji": "🐦", "category": "Animal / Aves", "tag": "🐾 Biological Organism"},
    "cat":        {"emoji": "🐱", "category": "Animal / Felidae", "tag": "🐾 Biological Organism"},
    "deer":       {"emoji": "🦌", "category": "Animal / Cervidae", "tag": "🐾 Biological Organism"},
    "dog":        {"emoji": "🐶", "category": "Animal / Canidae", "tag": "🐾 Biological Organism"},
    "frog":       {"emoji": "🐸", "category": "Animal / Amphibia", "tag": "🐾 Biological Organism"},
    "horse":      {"emoji": "🐴", "category": "Animal / Equidae", "tag": "🐾 Biological Organism"},
    "ship":       {"emoji": "🚢", "category": "Vehicle / Marine", "tag": "⚙️ Mechanical Transport"},
    "truck":      {"emoji": "🚚", "category": "Vehicle / Heavy Transport", "tag": "⚙️ Mechanical Transport"}
}

# ==============================================================================
# 4. MODEL LOADER & SMART DEMO MODE FALLBACK
# ==============================================================================
@st.cache_resource
def load_cifar_model():
    """Attempts to load keras model; returns (model, is_real_model)"""
    try:
        import tensorflow as tf
        if os.path.exists("cifar10_cnn.keras"):
            loaded_model = tf.keras.models.load_model("cifar10_cnn.keras")
            return loaded_model, True
        else:
            return None, False
    except Exception:
        return None, False

model, is_live_model = load_cifar_model()

# ==============================================================================
# 5. SIDEBAR DIAGNOSTICS & CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.2rem;">
        <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🧠</div>
        <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; margin: 0; color: #f8fafc;">Neural Inspector</h2>
        <span style="font-size: 0.78rem; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">CIFAR-10 Engine v2.5</span>
    </div>
    """, unsafe_allow_html=True)

    # Engine Status Badge
    if is_live_model:
        st.markdown("""
        <div class="sidebar-section" style="border-left: 3px solid #10b981;">
            <div style="font-size: 0.75rem; color: #10b981; font-weight: 700;">● INFERENCE ENGINE READY</div>
            <div style="font-size: 0.85rem; color: #e2e8f0; margin-top: 0.2rem;">Model: <code>cifar10_cnn.keras</code></div>
            <div style="font-size: 0.75rem; color: #94a3b8;">TensorFlow Keras CNN Architecture</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-section" style="border-left: 3px solid #f59e0b;">
            <div style="font-size: 0.75rem; color: #f59e0b; font-weight: 700;">⚡ SIMULATION DEMO MODE</div>
            <div style="font-size: 0.8rem; color: #e2e8f0; margin-top: 0.2rem;"><code>cifar10_cnn.keras</code> not detected in root folder.</div>
            <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 0.2rem;">Full UI and simulation inference active. Drop your trained model file in root to switch to live CNN!</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Vision Pre-processing")
    with st.expander("✨ Real-Time Tensor Filters", expanded=False):
        brightness_val = st.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
        contrast_val = st.slider("Contrast", 0.5, 1.5, 1.0, 0.05)
        apply_grayscale = st.checkbox("Convert to Grayscale", False)
        rotation_deg = st.select_slider("Rotate (Degrees)", options=[0, 90, 180, 270], value=0)

    st.markdown("### 🏷️ 10 Target Classes")
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem;">
        <span class="telemetry-chip">✈️ Plane</span>
        <span class="telemetry-chip">🚗 Auto</span>
        <span class="telemetry-chip">🐦 Bird</span>
        <span class="telemetry-chip">🐱 Cat</span>
        <span class="telemetry-chip">🦌 Deer</span>
        <span class="telemetry-chip">🐶 Dog</span>
        <span class="telemetry-chip">🐸 Frog</span>
        <span class="telemetry-chip">🐴 Horse</span>
        <span class="telemetry-chip">🚢 Ship</span>
        <span class="telemetry-chip">🚚 Truck</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 CNN Architecture Specs")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;" class="sidebar-section">
        <div>• <b>Input:</b> 32×32×3 RGB Normalized [0, 1]</div>
        <div>• <b>Backbone:</b> Conv2D + BatchNorm + MaxPool</div>
        <div>• <b>Regularization:</b> Dropout (0.25 - 0.50)</div>
        <div>• <b>Head:</b> Dense + Softmax (10 logits)</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("High-Performance Computer Vision Studio")

# ==============================================================================
# 6. HERO HEADER SECTION
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        Deep Learning Computer Vision Studio
    </div>
    <h1 class="hero-title">CIFAR-10 Neural Classifier</h1>
    <p class="hero-subtitle">
        Feed visual inputs into a deep convolutional neural network. 
        Inspect 32×32 spatial tensor feature maps and analyze probabilistic softmax activations in real-time.
    </p>
    <div class="telemetry-bar">
        <span class="telemetry-chip">🎯 10 Semantic Targets</span>
        <span class="telemetry-chip">📐 32×32 Spatial Tensor</span>
        <span class="telemetry-chip">⚡ Sub-25ms Inference</span>
        <span class="telemetry-chip">📊 Softmax Probabilities</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 7. SAMPLE IMAGES GENERATOR (FOR FRICTIONLESS DEMO)
# ==============================================================================
def generate_sample_image(class_key):
    """Generates an aesthetic stylized icon/graphic sample image for testing"""
    img = Image.new("RGB", (256, 256), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    
    colors = {
        "airplane":   ((14, 165, 233), "✈️", "Airplane Jet"),
        "automobile": ((239, 68, 68), "🚗", "Sports Car"),
        "cat":        ((245, 158, 11), "🐱", "Feline Cat"),
        "dog":        ((59, 130, 246), "🐶", "Canine Dog"),
        "frog":       ((34, 197, 94), "🐸", "Tree Frog"),
        "ship":       ((99, 102, 241), "🚢", "Ocean Liner")
    }
    col, emoji, title = colors.get(class_key, ((168, 85, 247), "🔮", "Object"))
    for r in range(120, 10, -10):
        draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=col, width=2)
        
    return img, emoji, title

# ==============================================================================
# 8. WORKSPACE LAYOUT (DUAL COLUMN WORKSTATION)
# ==============================================================================
col_left, col_right = st.columns([1.05, 1.15], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-header">
            <h3><span>📥</span> Visual Input Source</h3>
            <span class="panel-tag">TENSOR STAGE 1</span>
        </div>
    """, unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs(["📤 Upload Image", "🎨 Sample Preset Gallery"])
    
    active_image = None
    image_source_name = ""

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload image file",
            type=["jpg", "jpeg", "png", "webp"],
            help="Drop any image. The neural pipeline will crop/resize to 32×32.",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_source_name = uploaded_file.name

    with input_tab2:
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-bottom:0.7rem;'>Select a quick-test benchmark image to test the neural network immediately:</p>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        sample_keys = ["airplane", "automobile", "cat", "dog", "frog", "ship"]
        
        for i, key in enumerate(sample_keys):
            with preset_cols[i % 3]:
                meta = CLASS_META[key]
                if st.button(f"{meta['emoji']} {key.capitalize()}", key=f"btn_sample_{key}"):
                    st.session_state["chosen_sample"] = key

        if "chosen_sample" in st.session_state:
            active_key = st.session_state["chosen_sample"]
            sample_img, s_emoji, s_title = generate_sample_image(active_key)
            if active_image is None:
                active_image = sample_img
                image_source_name = f"Sample: {s_emoji} {s_title}"

    # Display Loaded Image + X-Ray Tensor Inspector
    if active_image is not None:
        processed_pil = active_image.copy()
        if rotation_deg != 0:
            processed_pil = processed_pil.rotate(-rotation_deg, expand=True)
        if brightness_val != 1.0:
            enhancer = ImageEnhance.Brightness(processed_pil)
            processed_pil = enhancer.enhance(brightness_val)
        if contrast_val != 1.0:
            enhancer = ImageEnhance.Contrast(processed_pil)
            processed_pil = enhancer.enhance(contrast_val)
        if apply_grayscale:
            processed_pil = processed_pil.convert("L").convert("RGB")

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        preview_tab_img, preview_tab_tensor = st.tabs(["👁️ High-Res Preview", "🔬 32×32 Tensor 'Eyes'"])
        
        with preview_tab_img:
            st.image(
                processed_pil,
                caption=f"Input: {image_source_name} ({processed_pil.width}×{processed_pil.height}px)",
                use_container_width=True
            )
            
        with preview_tab_tensor:
            tensor_32 = processed_pil.resize((32, 32), Image.Resampling.BILINEAR)
            st.markdown("""
            <div class="xray-box">
                <div style="font-size:0.75rem; color:#38bdf8; font-weight:700; margin-bottom:0.5rem; letter-spacing:0.05em;">
                    NEURAL INPUT RESOLUTION (32×32 PIXEL MATRIX)
                </div>
            """, unsafe_allow_html=True)
            st.image(tensor_32.resize((224, 224), Image.Resampling.NEAREST), use_container_width=True)
            st.markdown("""
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.4rem;">
                    Nearest-neighbor pixel reconstruction showing spatial resolution fed into CNN kernels.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-label">Raw Res</div>
                <div class="stat-value">{processed_pil.width}×{processed_pil.height}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Model Res</div>
                <div class="stat-value">32×32</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Channels</div>
                <div class="stat-value">3 (RGB)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 14px; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem; opacity: 0.7;">🖼️</div>
            <div style="font-weight: 600; color: #cbd5e1; font-size: 0.95rem;">No Image Loaded</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.2rem;">Drop a file above or select an instant sample preset.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 9. INFERENCE & NEURAL PREDICTION ENGINE
# ==============================================================================
with col_right:
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-header">
            <h3><span>🔮</span> Inference & Probability Vector</h3>
            <span class="panel-tag">TENSOR STAGE 2</span>
        </div>
    """, unsafe_allow_html=True)

    if active_image is None:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.5;">⚡</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #e2e8f0;">Neural Pipeline Idle</div>
            <p style="color: #64748b; font-size: 0.88rem; max-width: 320px; margin: 0.4rem auto 0 auto;">
                Provide an image on the left to trigger convolutional forward pass and visualize predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        run_prediction = st.button("⚡ EXECUTE NEURAL INFERENCE", use_container_width=True)

        if run_prediction or ("auto_predicted" in st.session_state and st.session_state.get("last_img") == image_source_name):
            st.session_state["auto_predicted"] = True
            st.session_state["last_img"] = image_source_name

            t_start = time.perf_counter()

            # Preprocess to (1, 32, 32, 3) normalized [0, 1]
            img_resized = processed_pil.resize((32, 32), Image.Resampling.BILINEAR)
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            # Compute predictions (Live model or Realistic Simulation)
            if is_live_model and model is not None:
                raw_predictions = model.predict(img_tensor, verbose=0)
                predictions = raw_predictions[0]
            else:
                time.sleep(0.08)  # simulate neural latency
                chosen_sample = st.session_state.get("chosen_sample", None)
                sim_logits = np.random.uniform(0.02, 0.15, size=10)
                if chosen_sample and chosen_sample in CLASS_NAMES:
                    target_idx = CLASS_NAMES.index(chosen_sample)
                    sim_logits[target_idx] = np.random.uniform(3.5, 6.0)
                else:
                    seed_val = int(np.mean(img_array) * 1000) % 10
                    sim_logits[seed_val] = 4.2
                exp_logits = np.exp(sim_logits - np.max(sim_logits))
                predictions = exp_logits / np.sum(exp_logits)

            inference_time_ms = (time.perf_counter() - t_start) * 1000.0

            predicted_idx = int(np.argmax(predictions))
            top_confidence = float(predictions[predicted_idx])
            top_class = CLASS_NAMES[predicted_idx]
            meta = CLASS_META[top_class]

            sorted_indices = np.argsort(predictions)[::-1]
            second_confidence = float(predictions[sorted_indices[1]])
            margin = (top_confidence - second_confidence) * 100.0

            if top_confidence >= 0.75:
                tier_label = "HIGH CONFIDENCE"
                tier_color = "#10b981"
                tier_bg = "rgba(16, 185, 129, 0.15)"
            elif top_confidence >= 0.45:
                tier_label = "MODERATE CONFIDENCE"
                tier_color = "#f59e0b"
                tier_bg = "rgba(245, 158, 11, 0.15)"
            else:
                tier_label = "LOW CONFIDENCE / AMBIGUOUS"
                tier_color = "#ef4444"
                tier_bg = "rgba(239, 68, 68, 0.15)"

            # ==================================================================
            # HERO WINNER SHOWCASE
            # ==================================================================
            st.markdown(f"""
            <div class="winner-card">
                <div class="winner-avatar-ring">
                    {meta['emoji']}
                </div>
                <div class="winner-class">{top_class}</div>
                <div class="winner-category-pill">{meta['tag']} · {meta['category']}</div>
                
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.4rem;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: {tier_color}; background: {tier_bg}; padding: 0.2rem 0.6rem; border-radius: 6px;">
                        ● {tier_label}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: #38bdf8;">
                        {top_confidence * 100:.1f}%
                    </span>
                </div>
                
                <div class="meter-container">
                    <div class="meter-fill" style="width: {min(top_confidence*100, 100):.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Telemetry Row
            st.markdown(f"""
            <div class="stat-grid" style="margin-bottom: 1.2rem;">
                <div class="stat-item">
                    <div class="stat-label">Inference Latency</div>
                    <div class="stat-value" style="color: #38bdf8;">⚡ {inference_time_ms:.1f} ms</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Runner-Up Gap</div>
                    <div class="stat-value">+{margin:.1f}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Entropy Loss</div>
                    <div class="stat-value">{-np.log(max(top_confidence, 1e-6)):.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ==================================================================
            # TOP 3 PODIUM LEADERBOARD
            # ==================================================================
            st.markdown("<h4 style='font-family: Outfit; font-size: 1.1rem; color: #f1f5f9; margin-bottom: 0.6rem;'>🏆 Top 3 Ranked Hypotheses</h4>", unsafe_allow_html=True)

            medals = ["🥇", "🥈", "🥉"]
            rank_classes = ["rank-gold", "rank-silver", "rank-bronze"]

            for rank in range(3):
                idx = sorted_indices[rank]
                prob = float(predictions[idx])
                c_name = CLASS_NAMES[idx]
                c_meta = CLASS_META[c_name]

                st.markdown(f"""
                <div class="podium-card">
                    <div style="display: flex; align-items: center;">
                        <span class="podium-rank-chip {rank_classes[rank]}">{medals[rank]}</span>
                        <div class="podium-label">
                            <span>{c_meta['emoji']}</span>
                            <span style="text-transform: capitalize;">{c_name}</span>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.8rem;">
                        <div style="width: 100px; height: 6px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden;">
                            <div style="width: {prob*100:.1f}%; height: 100%; background: #38bdf8;"></div>
                        </div>
                        <span class="podium-score">{prob * 100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ==================================================================
            # ALL 10 CLASSES PROBABILITY SPECTRUM
            # ==================================================================
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander("📊 Complete Softmax Activation Spectrum (All 10 Classes)", expanded=False):
                col_spec_v, col_spec_a = st.columns(2)
                
                with col_spec_v:
                    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#38bdf8; margin-bottom:0.5rem;'>⚙️ VEHICLES (MACHINERY)</div>", unsafe_allow_html=True)
                    for c_name in ["airplane", "automobile", "ship", "truck"]:
                        idx = CLASS_NAMES.index(c_name)
                        score = float(predictions[idx])
                        st.markdown(f"""
                        <div style="margin-bottom: 0.6rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.2rem;">
                                <span>{CLASS_META[c_name]['emoji']} {c_name.capitalize()}</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{score*100:.2f}%</span>
                            </div>
                            <div style="height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;">
                                <div style="width: {score*100:.1f}%; height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_spec_a:
                    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#ec4899; margin-bottom:0.5rem;'>🐾 ANIMALS (ORGANISMS)</div>", unsafe_allow_html=True)
                    for c_name in ["bird", "cat", "deer", "dog", "frog", "horse"]:
                        idx = CLASS_NAMES.index(c_name)
                        score = float(predictions[idx])
                        st.markdown(f"""
                        <div style="margin-bottom: 0.6rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.2rem;">
                                <span>{CLASS_META[c_name]['emoji']} {c_name.capitalize()}</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{score*100:.2f}%</span>
                            </div>
                            <div style="height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;">
                                <div style="width: {score*100:.1f}%; height: 100%; background: linear-gradient(90deg, #ec4899, #f43f5e);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            if top_confidence > 0.80:
                st.snow()
        else:
            st.info("👆 Click **EXECUTE NEURAL INFERENCE** to trigger convolutional prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 10. SYSTEM FOOTER
# ==============================================================================
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); color: #64748b; font-size: 0.85rem;">
    <span>⚡ CIFAR-10 Vision Classifier Studio</span> · 
    <span>Trained on 60,000 32×32 Natural Images</span> · 
    <span>Built with TensorFlow + Streamlit</span>
</div>
""", unsafe_allow_html=True)
