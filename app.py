import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("Defect Detection using YOLOv8")
st.subheader("Scratch Detection, Length and Severity Analysis")

# Load model
@st.cache_resource
def load_model():
    return YOLO("model.pt")

model = load_model()

# Pixel to mm conversion (adjust if needed)
PIXEL_TO_MM = 0.1

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)

            results = model(tmp.name, conf=0.15, iou=0.3)

        for r in results:
            boxes = r.boxes

            if boxes is not None and len(boxes) > 0:
                st.success(f"Total scratches detected: {len(boxes)}")

                count = 1
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    width = x2 - x1
                    height = y2 - y1

                    length_pixels = max(width, height)
                    length_mm = length_pixels * PIXEL_TO_MM

                    # Severity classification
                    if length_mm < 5:
                        severity = "Low"
                    elif length_mm < 15:
                        severity = "Medium"
                    else:
                        severity = "High"

                    conf = float(box.conf[0])

                    st.write(f"Scratch {count}")
                    st.write(f"Length: {length_mm:.2f} mm")
                    st.write(f"Severity: {severity}")
                    st.write(f"Confidence: {conf:.2f}")

                    count += 1

            else:
                st.warning("No scratches detected in this image")

            st.image(r.plot(), caption="Detection Result", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
