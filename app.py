import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Title
st.title("Defect Detection using YOLOv8")
st.subheader("Scratch Detection and Length Measurement")
st.write("Upload a surface image to detect scratches and measure defect length.")

# Load model
model = YOLO("model.pt")

# Upload file
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Run detection
        results = model(tmp.name, conf=0.25, iou=0.5)

        for r in results:
            boxes = r.boxes

            if boxes is not None and len(boxes) > 0:
                st.success(f"Total scratches detected: {len(boxes)}")

                count = 1
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    width = x2 - x1
                    height = y2 - y1

                    length = max(width, height)

                    st.write(f"Scratch {count} length: {int(length)} pixels")
                    count += 1

            else:
                st.warning("No scratches detected in this image")

            # Show detection result
            st.image(r.plot(), caption="Detected Image", use_column_width=True)
