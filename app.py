import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f2937;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# Title
st.write("App started")
st.markdown('<div class="title">🪖 Helmet Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect Helmet and No Helmet using YOLOv11</div>', unsafe_allow_html=True)
st.write("")

# Load model
@st.cache_resource
def load_model():
    return YOLO("helmet-no-helmet.pt")   # change path if needed

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting helmets..."):
            img_array = np.array(image)
            results = model(img_array)

            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.success("Detection Completed")
        st.image(annotated_img, caption="Detection Result", use_container_width=True)

        # Show detections summary
        st.subheader("Detection Summary")
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                st.write(f"• {label} with confidence {conf:.2f}")
        else:
            st.write("No objects detected")

# Footer
st.markdown('<div class="footer">Built with Streamlit and YOLOv11</div>', unsafe_allow_html=True)


