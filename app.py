import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import os

# 🎨 Page configuration
st.set_page_config(
    page_title="AI Age Predictor",
    page_icon="📷",
    layout="centered"
)

# 🏷️ Title and description
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'> AI Age Predictor</h1>
    <p style='text-align: center; font-size:18px; color: #555;'>
        Upload a face image and let the model predict the estimated age.
    </p>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = r"E:\\age-prediction-app\\best.h5"

# 🧠 Load model safely
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model file 'best.h5' not found!")
        st.stop()
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Ensure safe prediction model (Functional API)
    input_tensor = model.input
    output_tensor = model.output
    prediction_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    return prediction_model

# 🖼️ Preprocess image
def preprocess(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# 🔄 Load model
model = load_model()

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display uploaded image
        img_pil = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(img_pil, caption="📸 Uploaded Image", use_container_width=True)

        with st.spinner("⏳ Predicting age..."):
            # Resize image to match model input
            H, W = model.input_shape[1:3]
            img_arr = preprocess(img_pil, target_size=(H, W))

            # Run prediction
            prediction = model.predict(img_arr)
            age = float(prediction[0][0])

        # 🎉 Show result
        st.success(f"👤 Estimated Age: **{age:.1f} years**")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
