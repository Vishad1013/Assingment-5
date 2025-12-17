import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Brain Tumor Classification (AI)",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification using AI")
st.caption("EfficientNetB0 • Grad-CAM • Deep Learning")
st.write("Upload a brain MRI image to classify tumor type and visualize AI attention regions.")

# =====================================================
# LOAD MODEL & CLASS LABELS
# =====================================================
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists("best_model.keras"):
        st.error("❌ Model file 'best_model.keras' not found.")
        st.stop()

    if not os.path.exists("class_labels.json"):
        st.error("❌ 'class_labels.json' not found.")
        st.stop()

    model = tf.keras.models.load_model("best_model.keras")

    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)

    labels = {v: k.replace("_", " ").title() for k, v in class_indices.items()}
    return model, labels

model, LABELS = load_model_and_labels()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img)

    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =====================================================
# GRAD-CAM
# =====================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Prediction
    preds = model.predict(img_array)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    # =====================================================
    # AI RESULT
    # =====================================================
    st.subheader("🤖 AI Result")

    if confidence >= 80:
        st.success(
            f"🧠 **Prediction:** {LABELS[pred_class]}  \n"
            f"🎯 **Confidence:** {confidence:.2f}%  \n"
            "✅ High confidence prediction."
        )
    elif confidence >= 60:
        st.warning(
            f"🧠 **Prediction:** {LABELS[pred_class]}  \n"
            f"🎯 **Confidence:** {confidence:.2f}%  \n"
            "⚠️ Moderate confidence. Use as decision-support only."
        )
    else:
        st.error(
            f"🧠 **Prediction:** {LABELS[pred_class]}  \n"
            f"🎯 **Confidence:** {confidence:.2f}%  \n"
            "❗ Low confidence. Expert medical review recommended."
        )

    # =====================================================
    # GRAD-CAM VISUALIZATION
    # =====================================================
    st.subheader("🔥 Grad-CAM Visualization")

    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = np.array(image.resize((224, 224)))

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    st.image(overlay, caption="Grad-CAM Heatmap (Model Attention)", use_container_width=True)

    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================
    st.subheader("📊 Class Probabilities")
    prob_dict = {LABELS[i]: float(preds[0][i]) for i in range(len(LABELS))}
    st.bar_chart(prob_dict)

# =====================================================
# DISCLAIMER & ETHICS
# =====================================================
st.markdown("---")

st.markdown("""
### ⚖️ Medical & Legal Disclaimer
This AI system is **not a medical device** and **must not be used for diagnosis or treatment**.  
Predictions are generated using a deep learning model trained on retrospective datasets and may not generalize to all populations.

### 🎓 Academic Disclaimer
This project is developed **solely for educational and research purposes** as part of an academic curriculum.  
Results should not replace professional clinical judgment.

### 🤖 AI Ethics Statement
- The model may contain dataset bias  
- Predictions include uncertainty  
- Human oversight is mandatory  
- Transparency via Grad-CAM is provided  

### 📚 Citation
Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
""")

st.markdown("Developed using **EfficientNetB0**, **Grad-CAM**, and **Streamlit**")
