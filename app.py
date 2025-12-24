import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image

# importing build_model from model.py
from model import build_model

# UI
def apply_custom_styles():
    IMAGE_URL = "https://tse3.mm.bing.net/th/id/OIP.wOz-U9xMDdI1GPuEHA3MBgHaEo?cb=ucfimg2&pid=ImgDet&ucfimg=1&w=474&h=296&rs=1&o=7&rm=3"
    st.markdown(f"""
    <style>
    /* Hypnotic Moving Fractal Background */
    .stApp {{
        background-image: url("{IMAGE_URL}");
        background-attachment: fixed;
        background-size: 300% 300%;
        background-position: center center;
        animation: moveBackground 45s ease-in-out infinite;
    }}

    @keyframes moveBackground {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Glassmorphism Cards */
    .result-card {{
        background: rgba(0, 0, 0, 0.75);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 15px;
        margin-bottom: 30px;
        width: 100%;
    }}

    /* Legibility */
    h1, h2, h3, p, span, label, div[data-testid="stMarkdownContainer"] > p {{
         color:
         text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(15px);
    }}
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="HistoPath AI | Tumor Detection", page_icon="🧬", layout="wide")
apply_custom_styles()


@st.cache_resource
def load_model():
    model = build_model(input_shape=(96, 96, 3))
    try:
        model.load_weights("histopath_model.h5")
    except Exception as e:
        st.error(f"⚠️ Model weights not found! Error: {e}")
        st.stop()
    return model


model = load_model()


# Grad-CAM func
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(image, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image)
    superimposed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img



# Side-bar
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    try:
        show_cam = st.toggle("Enable Grad-CAM Overlay", value=True)
    except AttributeError:
        show_cam = st.checkbox("Enable Grad-CAM Overlay", value=True)
    cam_opacity = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5)
    st.divider()
    st.header("📌 Model Specs")
    st.info("**Architecture:** ResNet50\n\n**Task:** Metastatic Tissue Detection\n\n**Input:** 96x96 PCam Patches")
    st.warning("⚠️ Research Use Only.")

st.title("🧬 Histopathology Tumor Detection")
st.markdown("### AI-Powered Tissue Analysis")

uploaded_files = st.file_uploader("Upload tissue patches (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=True)

if uploaded_files:
    st.subheader("🤖 Analyzing...")
    st.divider()
    results_list = []

    for file in uploaded_files:
        with st.container():
            # 1. Image Row: Only 2 columns for the visual data
            col_img, col_cam = st.columns(2)

            image = Image.open(file).convert("RGB")
            img_resized = image.resize((96, 96))
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            start_time = time.time()
            prediction_prob = model.predict(img_array, verbose=0)[0][0]
            inference_duration = time.time() - start_time

            is_malignant = prediction_prob >= 0.5
            label = "MALIGNANT" if is_malignant else "BENIGN"
            conf_score = prediction_prob if is_malignant else 1 - prediction_prob
            theme_color = "#ef4444" if is_malignant else "#22c55e"

            with col_img:
                st.image(image, caption="Original Patch", use_column_width=True)

            with col_cam:
                if show_cam:
                    heatmap_data = make_gradcam_heatmap(img_array, model)
                    cam_visual = overlay_gradcam(image, heatmap_data, alpha=cam_opacity)
                    st.image(cam_visual, caption="Grad-CAM Focus", use_column_width=True)
                else:
                    st.write("Visualization Disabled")

            # 2. Result Row: The card appears directly below the images
            st.markdown(f"""
            <div class="result-card" style="border-top: 6px solid {theme_color};">
                <h2 style="color:{theme_color} !important; margin:0;">{label}</h2>
                <p style="margin:10px 0 5px 0; font-size: 1.2em;">Confidence: <b>{conf_score * 100:.2f}%</b></p>
                <p style="font-size: 0.9em; opacity: 0.7;">Inference Time: {inference_duration:.3f}s | File: {file.name}</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(conf_score))

            results_list.append({"Filename": file.name, "Result": label, "Confidence": f"{conf_score * 100:.2f}%"})
            st.markdown("<br>", unsafe_allow_html=True)  # Extra spacing between batches

    with st.expander("📊 View Batch Statistics"):
        st.dataframe(results_list)


# Edu footer using tabs
st.divider()
st.subheader("📘 Understanding the Technology")

tab1, tab2 = st.tabs(["🔍 What is Grad-CAM?", "🧠 The PCam Dataset"])

with tab1:
    st.markdown("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique used to make Convolutional Neural Networks (CNNs) transparent.

    It uses the gradients of the target concept (e.g., "Malignant") flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting that concept.

    * 🔴 **Red/Hot Regions:** Areas heavily used by the model to make its prediction (e.g., high cell density, irregular nuclei).
    * 🔵 **Blue/Cold Regions:** Areas that had little impact on the decision.
    """)

with tab2:
    st.markdown("""
    The **PatchCamelyon (PCam)** dataset is a clinically-relevant benchmark dataset for histopathology.

    * It consists of **327,680 color images** (96x96px) extracted from histopathologic scans of lymph node sections.
    * Each image is annotated with a binary label indicating the presence of metastatic tissue.
    * The challenge is that the metastatic tissue might only occupy a small portion of the central region of the patch.
    """)
