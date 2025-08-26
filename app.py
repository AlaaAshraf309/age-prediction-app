import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import os

st.set_page_config(page_title="Age Predictor", page_icon="📷", layout="centered")

st.title("🎯 تطبيق التنبؤ بالعمر")
st.write("ارفع صورة (وجه شخص) والموديل هيتنبأ بالعمر المتوقع.")

MODEL_PATH = r"E:\age-prediction-app\best.h5"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ ملف الموديل best.h5 مش موجود في نفس الفولدر!")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess(img, target_size=(224,224)):
    """تحويل الصورة لشكل مناسب للموديل"""
    img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# تحميل الموديل
model = load_model()

uploaded = st.file_uploader("📤 ارفع صورة", type=["jpg","jpeg","png"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="📸 الصورة المرفوعة", use_column_width=True)

        st.write("⏳ جاري التنبؤ...")
        # ناخد حجم الإدخال من الموديل (مثلاً 224x224)
        H, W = model.input_shape[1:3]
        img_arr = preprocess(image, target_size=(H, W))

        prediction = model.predict(img_arr)
        age = float(prediction[0][0])

        st.success(f"👤 العمر المتوقع: **{age:.1f} سنة**")

    except Exception as e:
        st.error(f"حصل خطأ أثناء التنبؤ: {e}")
