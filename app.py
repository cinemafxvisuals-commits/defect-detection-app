import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Defect Detection App")

# Load model
model = YOLO("model.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        
        # Run prediction
        results = model(tmp.name)

        # Show results
        for r in results:
            st.image(r.plot(), caption="Detected Image")
