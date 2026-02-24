"""
Pg 1: Vehicle Damage Detection
Upload image -> AI detection (damage and severity)
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(page_title="Damage Detection", page_icon="📸", layout="wide")
st.markdown("## Vehicle Damage Detection")
st.markdown("Upload a photo of the vehicle to detect and classify damage.")
st.divider()


# Load Model

@st.cache_resource
def load_damage_model():
    """Load the trained damage detection model."""
    from tensorflow.keras.models import load_model

    model_path = os.path.join("models", "damage_model.h5")

    if not os.path.exists(model_path):
        return None, None

    model = load_model(model_path)

    classes_path = os.path.join("models", "damage_classes.json")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = ["Damaged", "Not Damaged"]

    return model, class_names


model, class_names = load_damage_model()


# Img upload and prediction
uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of the vehicle damage",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Analysing Results")

        if model is None:
            st.error(
                "Model not found!⚠️Please train the damage detection model first⚠️ \n\n"
                "Run: `python notebooks/01_damage_detection.py`"
            )
        else:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx] * 100

                predicted_class = class_names[predicted_class_idx]

                # Map prediction to severity
                # Adjust this mapping based on your actual class names
                damage_detected = "damage" in predicted_class.lower() or "00" in predicted_class

                if damage_detected:
                    # Determine severity from confidence
                    if confidence > 90:
                        severity = "Severe Damage"
                        severity_color = "🔴"
                        severity_desc = "Significant damage detected. Major repairs likely needed."
                    elif confidence > 70:
                        severity = "Moderate Damage"
                        severity_color = "🟠"
                        severity_desc = "Moderate damage detected. Repairs recommended."
                    else:
                        severity = "Minor Damage"
                        severity_color = "🟡"
                        severity_desc = "Minor damage detected. Cosmetic repairs may be needed."

                    st.error(f"**DAMAGE DETECTED !!**")
                    st.markdown(f"**Severity:** {severity_color} **{severity}**")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                    st.info(severity_desc)

                    # Store results in session state for claim estimation
                    st.session_state["damage_detected"] = True
                    st.session_state["damage_severity"] = severity
                    st.session_state["damage_confidence"] = confidence

                else:
                    st.success("**NO DAMAGE DETECTED**")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                    st.info("The vehicle appears to be in good condition.")

                    st.session_state["damage_detected"] = False
                    st.session_state["damage_severity"] = "None"
                    st.session_state["damage_confidence"] = confidence

                # Show raw prediction probabilities
                st.divider()
                st.markdown("**Prediction Probabilities:**")
                for i, class_name in enumerate(class_names):
                    prob = prediction[0][i] * 100
                    st.progress(prob / 100, text=f"{class_name}: {prob:.1f}%")

    # Navigation
    st.divider()
    if st.session_state.get("damage_detected", False):
        st.success("Damage analysis complete! Proceed to claim estimation.")
        if st.button("Proceed to Claim Estimation →", type="primary", use_container_width=True):
            st.switch_page("pages/claim_estimation.py")
else:
    # Show sample instructions
    st.info(
        "Upload vehicle image to get started.\n\n"
        "**Tips for best results:**\n"
        "- Use a clear, well-lit photo\n"
        "- Capture the damaged area prominently\n"
        "- Avoid blurry or dark images"
    )