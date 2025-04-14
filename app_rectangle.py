import streamlit as st
from PIL import Image,ImageDraw
import numpy as np
import io
import cv2
from ultralytics import YOLO
from utils import resize_image
from config import YOLO_MODEL_PATH
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_model():
    model = YOLO(YOLO_MODEL_PATH)  # or yolov8s.pt, yolov5s.pt, etc.
    return model
# Title

model = load_model()
st.title("YOLO Object Detection App")

# Session state to keep the uploaded image
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'count' not in st.session_state:
    st.session_state.count=0

if 'annotated_image' not in st.session_state:
    st.session_state.uploaded_image = None

if "draw_mode" not in st.session_state:
    st.session_state.draw_mode = False

# Function: Dummy YOLO predictor — replace this with your actual model logic
def detect_objects(image: Image.Image) -> Image.Image:
    # Replace this dummy logic with your YOLO detection
    # For demo, we'll just convert it to grayscale as a "detected" dummy output
    return image.convert("L").convert("RGB")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Buttons
col1, col2, col3 = st.columns(3)
upload_pressed = col1.button("Show Image")
detect_pressed = col2.button("Detect")
annotated_pressed = col3.button("Annotate")

if upload_pressed and uploaded_file:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Original Image", use_container_width=True)

elif detect_pressed:
    if st.session_state.uploaded_image is None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
    
    original = st.session_state.uploaded_image
    image_np=np.array(original)
    results=model(image_np)
    annotated_image=results[0].plot()
    st.session_state.annotated_image = annotated_image

    # annotated_image=Image.fromarray(annotated_image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_container_width=True)
    with col2:
        st.image(annotated_image, caption="Detected Image", use_container_width=True)

elif annotated_pressed:
    if st.session_state.uploaded_image is None:
        st.session_state.uploaded_image = Image.open(uploaded_file)

    st.session_state.draw_mode = True

if st.session_state.draw_mode == True:
    original = st.session_state.uploaded_image
    image_np=np.array(original)
    print(st.session_state.count)
    # results=model(image_np)
    # annotated_image=results[0].plot()
    annotated_image_pil=resize_image(st.session_state.annotated_image,original.size[0],original.size[1])
    if st.session_state.count==0:
        st.session_state.original_resized = resize_image(np.array(original),original.size[0],original.size[1])
    # #annotated_image_pil=Image.fromarray(annotated_image)
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Red fill with alpha
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=annotated_image_pil,
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            # Take the first drawn rectangle
            for i, rect in enumerate(objects):
                # rect = objects[0]
                left = rect["left"]
                top = rect["top"]
                width = rect["width"]
                height = rect["height"]
                # img_with_box = original_resized.copy()
                draw = ImageDraw.Draw(st.session_state.original_resized)
                draw.rectangle(
                    [(left, top), (left + width, top + height)],
                    outline="green",
                    width=3,
                )
                st.session_state.count=st.session_state.count+1
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original Image", use_container_width=True)
            with col2:
                st.image(st.session_state.original_resized, caption="Annotated Image", use_container_width=True)
            if st.button("Done", type="primary"):
                # st.session_state.draw_mode = False
                for key in [
                    "uploaded_image",
                    "annotated_image",
                    "annotated_image_pil",
                    "original_resized",
                    "count",
                    "draw_mode",
                ]:
                    st.session_state.pop(key, None)
                st.success("✅ Annotation saved. Upload a new image to start again!")
                st.rerun()

        #st.image(img_with_box, caption="Image with Corrected Rectangle")

