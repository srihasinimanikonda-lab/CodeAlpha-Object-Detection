
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎀",
    layout="centered"
)

# Pink Theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #fff0f6 0%,
        #ffe4ec 25%,
        #ffd6e7 50%,
        #ffc2dc 75%,
        #ffb3d1 100%
    );
}

h1 {
    text-align: center;
    color: #d63384;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.8);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎀 AI Object Detection System")
st.write("Upload an image and detect objects using YOLOv8 ✨")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="📸 Uploaded Image",
        use_container_width=True
    )

    with st.spinner("🔍 Detecting Objects..."):

        model = YOLO("yolov8n.pt")

        results = model(np.array(image))

        detected_image = results[0].plot()

        st.image(
            detected_image,
            caption="🎯 Detected Objects",
            use_container_width=True
        )

        detected_classes = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            detected_classes.append(model.names[cls])

        if detected_classes:
            st.success(
                f"✅ Detected Objects: {', '.join(set(detected_classes))}"
            )
        else:
            st.warning("⚠️ No objects detected in the image.")

