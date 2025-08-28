import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import os

st.set_page_config(page_title="Age Predictor", page_icon="📷", layout="centered")

st.title("🎯 تطبيق التنبؤ بالعمر")
st.write("ارفع صورة (وجه شخص) والموديل هيتنبأ بالعمر المتوقع.")

MODEL_PATH = r"E:\\age-prediction-app\\best.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ ملف الموديل best.h5 مش موجود!")
        st.stop()
    
    # تحميل الموديل الأصلي
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # إنشاء نسخة Prediction-safe باستخدام Functional API
    input_tensor = model.input
    output_tensor = model.output
    prediction_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    return prediction_model

def preprocess(img, target_size=(224,224)):
    img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# تحميل الموديل الجاهز للتنبؤ
model = load_model()

uploaded_file = st.file_uploader("📤 ارفع صورة", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    try:
        img_pil = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(img_pil, caption="📸 الصورة المرفوعة", use_column_width=True)
        st.write("⏳ جاري التنبؤ...")

        H, W = model.input_shape[1:3]
        img_arr = preprocess(img_pil, target_size=(H, W))

        prediction = model.predict(img_arr)
        age = float(prediction[0][0])

        st.success(f"👤 العمر المتوقع: **{age:.1f} سنة**")

    except Exception as e:
        st.error(f"حصل خطأ أثناء التنبؤ: {e}")
