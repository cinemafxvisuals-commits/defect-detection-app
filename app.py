import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Defect Detection + Length Measurement")

# Load model
model = YOLO("model.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Run model with filtering
        results = model(tmp.name, conf=0.6, iou=0.3)

        for r in results:
            boxes = r.boxes

            if boxes is not None and len(boxes) > 0:
                best_length = 0

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    width = x2 - x1
                    height = y2 - y1

                    length = max(width, height)

                    # Ignore very large boxes (almost full image)
                    if length < 0.8 * image.size[0]:
                        if length > best_length:
                            best_length = length

                if best_length > 0:
                    st.success(f"Defect length (pixels): {int(best_length)}")
                else:
                    st.warning("No valid defect detected")

            else:
                st.warning("No defects detected")

            # Show detection image
            st.image(r.plot(), caption="Detected Image", use_column_width=True)
