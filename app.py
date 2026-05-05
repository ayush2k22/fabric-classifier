import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# ---------------------- #
#   Image Input          #
# ---------------------- #

st.markdown("### 📸 Upload or Capture Fabric Image")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

uploaded_file = None
camera_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload fabric image...", type=["jpg", "jpeg", "png"])

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture of fabric")

img = None

if camera_image is not None:
    img = Image.open(camera_image)
    st.success("Image captured from camera")

elif uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.success("Image uploaded from device")

# ---------------------- #
#   Prediction Pipeline  #
# ---------------------- #

if img is not None:
    st.image(img, caption="Selected Fabric Image", use_container_width=True)

    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions[0]) * 100

    st.subheader(f"🩺 Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # (keep your Grad-CAM code here inside this block)

else:
    st.info("Please upload or capture a fabric image to begin classification.")
    #  Explainable AI (Grad-CAM)
    # ---------------------- #
    import tensorflow as tf
    last_conv_layer_name = None
    # Find last conv layer automatically
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Convert to OpenCV format
        img_cv = np.array(img.convert("RGB"))
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        st.markdown("### 🔍 Explainable AI: Model Attention Map (Grad-CAM)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_cv, caption="Original", use_container_width=True)
        with col2:
            st.image(superimposed_img, caption="Model Focus (Grad-CAM)", use_container_width=True)

    # Confidence plot
    st.markdown("### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, predictions[0], color=['#ff7f0e', '#1f77b4'])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig)

else:
    st.info("Please upload a fabric image to begin classification.")