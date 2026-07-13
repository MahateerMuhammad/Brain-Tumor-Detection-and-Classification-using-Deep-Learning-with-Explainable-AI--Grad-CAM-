"""
Brain Tumor MRI Classification — Streamlit Dashboard
Student Capstone Project · ResNet50 Transfer Learning · Grad-CAM Explainability
"""

import io
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "best_transfer_model.keras")

BASE_MODEL_CANDIDATES = ["resnet50", "densenet121", "efficientnetb0"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #161b27;
    --bg-card:       #1c2333;
    --border:        #2a3347;
    --text-primary:  #e8ecf4;
    --text-secondary:#8b96b0;
    --text-muted:    #556070;
    --accent:        #4f8ef7;
    --accent-soft:   rgba(79,142,247,0.12);
    --accent-border: rgba(79,142,247,0.30);
    --radius:        10px;
    --radius-lg:     16px;
}

.stApp { background: var(--bg-primary); color: var(--text-primary); }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.main .block-container { padding: 2.5rem 3rem 4rem 3rem; max-width: 1200px; }

h1 { font-size:1.85rem !important; font-weight:700 !important; letter-spacing:-0.02em; color:var(--text-primary) !important; }
h2 { font-size:1.25rem !important; font-weight:600 !important; color:var(--text-primary) !important; }
p, li { color:var(--text-secondary) !important; line-height:1.65; font-size:0.92rem !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
}

.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}

.pred-class {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    text-transform: capitalize;
    line-height: 1.1;
}
.pred-confidence {
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--accent);
    margin-top: 0.3rem;
    font-variant-numeric: tabular-nums;
}
.pred-label-row {
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;
}
.status-dot {
    width:8px; height:8px; border-radius:50%; background:var(--accent);
    display:inline-block; box-shadow:0 0 0 3px var(--accent-soft);
}

.section-label {
    font-size:0.68rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:var(--text-muted);
    margin-bottom:0.75rem; display:block;
}

.disclaimer {
    background: rgba(240,168,67,0.07);
    border: 1px solid rgba(240,168,67,0.25);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: #c8934a !important;
    font-size: 0.83rem !important;
    line-height: 1.6 !important;
}

.thin-divider { border:none; border-top:1px solid var(--border); margin:1.5rem 0; }

.app-header {
    display:flex; align-items:flex-start; justify-content:space-between;
    padding-bottom:1.2rem; border-bottom:1px solid var(--border); margin-bottom:2rem;
}
.app-title { font-size:1.75rem; font-weight:700; letter-spacing:-0.03em; color:var(--text-primary); line-height:1.1; }
.app-subtitle { font-size:0.85rem; color:var(--text-muted); margin-top:0.35rem; }
.version-tag {
    font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:var(--text-muted);
    background:var(--bg-card); border:1px solid var(--border); border-radius:6px;
    padding:0.25rem 0.6rem; align-self:flex-start; margin-top:0.15rem;
}

.sidebar-section {
    font-size:0.65rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase;
    color:var(--text-muted) !important; margin:1.5rem 0 0.4rem 0; display:block;
}
.sidebar-value { font-size:0.88rem; font-weight:500; color:var(--text-primary) !important; }
.sidebar-note { font-size:0.8rem; color:var(--text-secondary) !important; line-height:1.55 !important; }
.acc-pill {
    display:inline-block; background:rgba(52,199,120,0.12); border:1px solid rgba(52,199,120,0.3);
    border-radius:20px; padding:0.2rem 0.75rem; font-size:0.85rem; font-weight:600;
    color:#34c778 !important; font-variant-numeric:tabular-nums;
}

.upload-hint { text-align:center; padding:3rem 2rem; color:var(--text-muted) !important; }
.upload-icon { font-size:2.5rem; display:block; margin-bottom:0.75rem; opacity:0.5; }
.upload-hint p { font-size:0.88rem !important; color:var(--text-muted) !important; margin:0 !important; }

[data-testid="stImage"] > img { border-radius:var(--radius) !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    # Plain Keras 3 load -- the model was saved with Keras 3, so it must be
    # loaded with Keras 3. Do NOT route this through tf_keras / legacy Keras 2
    # compat -- that layer cannot deserialize Keras-3-format .keras files and
    # will fail with a "cannot import parent module" TypeError.
    return keras.models.load_model(path, compile=False)


def detect_base_model_key(model) -> str:
    for layer in model.layers:
        if layer.name in BASE_MODEL_CANDIDATES:
            return layer.name
    return None


# ---------------------------------------------------------------------------
# Grad-CAM — Keras-3-safe eager replay
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, base_model_key=None, pred_index=None):
    """
    Manually replays the model's layers in eager mode inside a GradientTape,
    instead of building a new graph-based sub-model around a nested
    Functional submodel's intermediate output. Keras 3 does not reliably
    support tapping an intermediate layer of a nested submodel that way --
    building keras.Model(inputs=..., outputs=[nested_layer.output, ...])
    throws a KeyError at call time, reload or not. This eager replay stays
    entirely in normal Python execution, using the same trained layer
    objects (including the nested pretrained base as one callable layer),
    so it sidesteps that limitation completely.
    """
    if base_model_key is None:
        base_model_key = detect_base_model_key(model)
        if base_model_key is None:
            raise RuntimeError("Could not detect the pretrained base model layer.")

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    conv_output = None

    with tf.GradientTape() as tape:
        for layer in model.layers:
            if layer.__class__.__name__ == "InputLayer":
                continue
            x = layer(x, training=False)
            if layer.name == base_model_key:
                conv_output = x
                tape.watch(conv_output)
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_0 = conv_output[0]
    heatmap = conv_output_0 @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, predictions.numpy()[0]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)   # raw 0-255, no manual normalisation
    return np.expand_dims(arr, axis=0)


def overlay_gradcam(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.40):
    hm_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE, Image.LANCZOS)
    ) / 255.0
    jet = cm.get_cmap("jet")
    jet_rgb = jet(hm_resized)[:, :, :3]
    orig = np.array(pil_img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    blended = alpha * jet_rgb + (1.0 - alpha) * orig
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Probability chart
# ---------------------------------------------------------------------------
CHART_BG   = "#1c2333"
CHART_FG   = "#8b96b0"
ACCENT_CLR = "#4f8ef7"

def render_probability_chart(probs: np.ndarray, pred_index: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    colors = ["#2d3a52"] * len(CLASS_NAMES)
    colors[pred_index] = ACCENT_CLR

    y_pos = np.arange(len(CLASS_NAMES))
    bars = ax.barh(y_pos, probs, color=colors, height=0.55, zorder=3)

    for i, (bar, p) in enumerate(zip(bars, probs)):
        clr = "#e8ecf4" if i == pred_index else CHART_FG
        ax.text(
            min(p + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}",
            va="center", ha="left",
            fontsize=8.5, color=clr,
            fontfamily="monospace",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=9, color=CHART_FG)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(length=0)
    fig.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(model):
    base_key = detect_base_model_key(model)
    arch_name = (base_key or "custom_cnn").upper()

    st.sidebar.markdown("### Model Info")

    st.sidebar.markdown('<span class="sidebar-section">Architecture</span>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<span class="sidebar-value">{arch_name} Transfer Learning</span>', unsafe_allow_html=True)

    st.sidebar.markdown('<span class="sidebar-section">Strategy</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sidebar-value">Frozen base · fine-tuned top 30 layers</span>', unsafe_allow_html=True)

    st.sidebar.markdown('<span class="sidebar-section">Test Accuracy</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="acc-pill">~91–92 %</span>', unsafe_allow_html=True)

    st.sidebar.markdown('<span class="sidebar-section">Input shape</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sidebar-value">224 × 224 × 3 · raw pixels</span>', unsafe_allow_html=True)

    st.sidebar.markdown('<span class="sidebar-section">Output classes</span>', unsafe_allow_html=True)
    for i, cls in enumerate(CLASS_NAMES):
        st.sidebar.markdown(
            f'<span class="sidebar-value" style="display:block;margin-bottom:0.15rem;">'
            f'<span style="color:#556070;font-variant-numeric:tabular-nums;">{i}&nbsp;&nbsp;</span>'
            f'{cls.capitalize()}</span>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Known Limitations")
    st.sidebar.markdown(
        '<p class="sidebar-note">'
        '· <strong>Glioma / meningioma overlap</strong> — shared visual features cause '
        'misclassification at lower confidence levels.<br><br>'
        '· Trained on a single public benchmark; real-world scanner variability is not covered.<br><br>'
        '· Not validated on post-contrast, FLAIR, or non-T1 sequences.<br><br>'
        '· Grad-CAM is a coarse saliency approximation — not a clinically validated localisation method.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p class="sidebar-note" style="font-size:0.75rem !important;">'
        'Capstone project · 2024–25 · Not a medical device.'
        '</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">Brain Tumor MRI Classifier</div>
            <div class="app-subtitle">ResNet50 Transfer Learning · Grad-CAM Explainability · 4-class</div>
        </div>
        <div class="version-tag">v1.0 · Capstone</div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>Research / Educational Use Only.</strong>&ensp;This is a student capstone project,
        not a clinically validated diagnostic tool. Predictions made by this model
        <em>must not</em> be used to inform real medical decisions. Always consult a qualified
        radiologist or neurologist for any clinical assessment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Load model
    model_abs = os.path.abspath(MODEL_PATH)
    if not os.path.exists(model_abs):
        st.error(
            f"**Model file not found.**\n\nExpected: `{model_abs}`\n\n"
            "Place `best_transfer_model.keras` inside the `models/` directory at the project root."
        )
        st.stop()

    with st.spinner("Loading model…"):
        model = load_model(model_abs)

    render_sidebar(model)

    # Upload
    st.markdown('<span class="section-label">Input</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload MRI image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Upload a brain MRI image (JPG or PNG). Axial T1 slices work best.",
    )

    if uploaded is None:
        st.markdown("""
        <div class="upload-hint">
            <span class="upload-icon">⬡</span>
            <p>Upload a JPG or PNG brain MRI image to run inference</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Process
    pil_img   = Image.open(uploaded)
    img_array = preprocess_image(pil_img)

    with st.spinner("Running inference & computing Grad-CAM…"):
        try:
            heatmap, pred_index, probs = make_gradcam_heatmap(
                img_array, model, base_model_key=detect_base_model_key(model)
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    pred_class      = CLASS_NAMES[pred_index]
    pred_confidence = probs[pred_index]
    gradcam_pil     = overlay_gradcam(pil_img, heatmap, alpha=0.40)

    # Results
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Results</span>', unsafe_allow_html=True)

    col_pred, col_chart = st.columns([1, 1.4], gap="large")

    with col_pred:
        st.markdown(f"""
        <div class="result-card">
            <div class="pred-label-row">
                <span class="status-dot"></span>
                <span style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                             text-transform:uppercase;color:var(--text-muted);">Prediction</span>
            </div>
            <div class="pred-class">{pred_class}</div>
            <div class="pred-confidence">{pred_confidence:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        if pred_confidence < 0.60:
            st.markdown("""
            <div style="background:rgba(224,92,92,0.08);border:1px solid rgba(224,92,92,0.2);
                        border-radius:8px;padding:0.7rem 1rem;margin-top:0.5rem;">
                <span style="font-size:0.8rem;color:#c86060;">
                    Low confidence (&lt;60 %). Glioma / meningioma visual overlap may be a factor.
                    Do not rely on this result.
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif pred_confidence < 0.80:
            st.markdown("""
            <div style="background:rgba(240,168,67,0.07);border:1px solid rgba(240,168,67,0.2);
                        border-radius:8px;padding:0.7rem 1rem;margin-top:0.5rem;">
                <span style="font-size:0.8rem;color:#c8934a;">
                    Moderate confidence (60–80 %). Review the Grad-CAM overlay below.
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown(
            '<div class="result-card" style="padding-bottom:1rem;">'
            '<span class="section-label">Class probabilities</span>',
            unsafe_allow_html=True,
        )
        fig = render_probability_chart(probs, pred_index)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Image panels
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_img, col_cam = st.columns(2, gap="large")

    with col_img:
        st.markdown('<span class="section-label">Uploaded image</span>', unsafe_allow_html=True)
        st.image(pil_img.convert("RGB").resize(IMG_SIZE), use_container_width=True)

    with col_cam:
        st.markdown('<span class="section-label">Grad-CAM · saliency overlay</span>', unsafe_allow_html=True)
        st.image(gradcam_pil, use_container_width=True)

    st.markdown("""
    <p style="font-size:0.78rem;color:var(--text-muted);margin-top:0.6rem;line-height:1.55;">
        Grad-CAM highlights regions most influential to the prediction — warmer colours indicate
        higher attribution. Generated by replaying model layers in eager mode and computing
        gradients with respect to the ResNet50 final convolutional block output.
    </p>
    """, unsafe_allow_html=True)

    # Download
    buf = io.BytesIO()
    gradcam_pil.save(buf, format="PNG")
    st.download_button(
        label="Download Grad-CAM overlay",
        data=buf.getvalue(),
        file_name=f"gradcam_{pred_class}_{pred_confidence:.2f}.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()