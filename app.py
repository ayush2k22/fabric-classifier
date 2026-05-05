import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ---------------------- #
#  Model & Class Config  #
# ---------------------- #
MODEL_PATH = "model.h5"
CLASS_NAMES = ["Blended", "Cotton"]

@st.cache_resource
def load_cotton_model():
    return load_model(MODEL_PATH)

model = load_cotton_model()

# ---------------------- #
#   Preprocessing        #
# ---------------------- #
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ---------------------- #
#   Grad-CAM Function    #
# ---------------------- #
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ---------------------- #
#   UI Input             #
# ---------------------- #

st.title("🧶 Fabric Classifier")

st.markdown("### 📸 Upload or Capture Fabric Image")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

img = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload fabric image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.success("Image uploaded")

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.success("Image captured")

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

    # -------- Grad-CAM -------- #
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        img_cv = np.array(img.convert("RGB"))
        img_cv = cv2.resize(img_cv, (224, 224))

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        st.markdown("### 🔍 Grad-CAM Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_cv, caption="Original", use_container_width=True)

        with col2:
            st.image(superimposed_img, caption="Model Focus", use_container_width=True)

    # -------- Confidence Plot -------- #
    st.markdown("### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, predictions[0])
    ax.set_ylim([0, 1])

    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')

    st.pyplot(fig)

else:
    st.info("Please upload or capture a fabric image to begin classification.")