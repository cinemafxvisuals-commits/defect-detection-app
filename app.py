import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("🔍 Defect Detection App")
st.write("Upload an image to detect defects using AI model")

# Cache model
@st.cache_resource
def load_model():
    return YOLO("model.pt")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            image.save(tmp.name)

            with st.spinner("⏳ Detecting..."):
                results = model.predict(source=tmp.name)

        st.success("✅ Detection Completed!")

        result_image = results[0].plot()
        st.image(result_image, caption="🎯 Detection Result", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
