import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Defect Detection + Length Measurement")

model = YOLO("model.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model(tmp.name)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                width = x2 - x1
                height = y2 - y1

                # Approx length = max dimension
                length = max(width, height)

                st.write(f"Detected defect length (pixels): {int(length)}")

            st.image(r.plot(), caption="Detected Image")
