import streamlit as st
from PIL import Image,ImageDraw
import numpy as np
import io
import cv2
from ultralytics import YOLO
from utils import resize_image,draw_points_on_black_image
from config import YOLO_MODEL_PATH,ANNOTATED_MASK_PATH
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_model():
    model = YOLO(YOLO_MODEL_PATH)  # or yolov8s.pt, yolov5s.pt, etc.
    return model
# Title

file_name=""

model = load_model()
st.title("YOLO Object Detection App")

if 'file_name' not in st.session_state:
    st.session_state.file_name=None
# Session state to keep the uploaded image
if 'original_resized_image' not in st.session_state:
    st.session_state.original_resized_image = None

if 'count' not in st.session_state:
    st.session_state.count=0

if 'annotated_image' not in st.session_state:
    st.session_state.uploaded_image = None

if "draw_mode" not in st.session_state:
    st.session_state.draw_mode = False

if 'tempList' not in st.session_state:
    st.session_state.tempList = {}

if 'black_image' not in st.session_state:
    st.session_state.black_image=None

if "done_clicked" not in st.session_state:
    st.session_state.done_clicked = False

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
    
    st.session_state.file_name = uploaded_file.name
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Original Image", width=500)

elif detect_pressed:
    if st.session_state.uploaded_image is None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
    
    original = st.session_state.uploaded_image
    image_np = np.array(original)
    results=model(image_np)
    annotated_image=results[0].plot()
    st.session_state.annotated_image = annotated_image
    st.session_state.original_resized_image = resize_image(image_np,500,500)

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
    #annotated_image_pil=np.array(original)
    annotated_image_pil=resize_image(st.session_state.annotated_image,500,500)
    if st.session_state.count==0:
        st.session_state.original_resized = annotated_image_pil
    #     st.session_state.original_resized = resize_image(original,500,500)
    # #annotated_image_pil=Image.fromarray(annotated_image)
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Red fill with alpha
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=annotated_image_pil,
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="polygon",
        key="canvas",
    )
    # exit()
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            # Take the first drawn rectangle
            for i, obj in enumerate(objects):
                # print(obj.keys())
                path_data = obj["path"]
                # print(path_data)
                # exit()
                polygon_points = [(point[1], point[2]) for point in path_data if point[0] == 'L' or point[0] == 'M']
                draw = ImageDraw.Draw(st.session_state.original_resized)
                st.session_state.tempList["label_"+str(st.session_state.count)]=polygon_points
                
                draw.polygon(polygon_points, outline="blue", width=2)
                st.session_state.count=st.session_state.count+1
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.original_resized_image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(st.session_state.original_resized, caption="Annotated Image", use_container_width=True)
            if st.button("Show Generated Mask", type="primary"):
                st.session_state.black_image=draw_points_on_black_image(st.session_state.original_resized,st.session_state.tempList)
                black_image_pil=Image.fromarray(st.session_state.black_image)
                with col1:
                    st.image(st.session_state.original_resized, caption="Annotated Image", use_container_width=True)
            
                with col2:
                    st.image(black_image_pil, caption="Original Image", use_container_width=True)
            if st.button("Done",type="primary"):
                st.session_state.draw_mode = False
                st.session_state.done_clicked = True
            if st.session_state.done_clicked:
                cv2.imwrite(ANNOTATED_MASK_PATH+st.session_state.file_name+'.jpg',st.session_state.black_image)
                st.session_state.black_image=None
                for key in [
                    "file_name",
                    "uploaded_image",
                    "annotated_image",
                    "annotated_image_pil",
                    "original_resized",
                    "count",
                    "draw_mode",
                    "black_image",
                    "done_clicked",  # remove the flag too
                    "tempList"
                ]:
                    st.session_state.pop(key, None)
                st.success("✅ Annotation saved. Upload a new image to start again!")
                st.rerun()

        #st.image(img_with_box, caption="Image with Corrected Rectangle")

