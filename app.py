import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from config import YOLO_MODEL_PATH,ANNOTATED_MASK_PATH
from utils import (init_session_state,resize_image,draw_points_on_black_image,detect_objects)
# Load model once


@st.cache_resource
def load_model():
    return YOLO(YOLO_MODEL_PATH)


def main():

    model = load_model()
    st.title("YOLO Object Detection App")

    init_session_state()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # UI controls
    col1, col2, col3 = st.columns(3)
    upload_pressed = col1.button("Show Image")
    detect_pressed = col2.button("Detect")
    annotate_pressed = col3.button("Annotate")

    # Upload logic
    if upload_pressed and uploaded_file:
        st.session_state.file_name = uploaded_file.name
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="Original Image", width=500)

    # Detection logic
    elif detect_pressed:
        if st.session_state.uploaded_image is None and uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
        
        original_image = st.session_state.uploaded_image
        st.session_state.annotated_image = detect_objects(original_image,model)
        st.session_state.original_resized_image = resize_image(np.array(original_image), 500, 500)

        col1, col2 = st.columns(2)
        col1.image(original_image, caption="Original Image")
        col2.image(st.session_state.annotated_image, caption="Detected Image")

    # Annotate logic
    elif annotate_pressed:
        if st.session_state.uploaded_image is None and uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
        st.session_state.draw_mode = True

    # Drawing mode
    if st.session_state.draw_mode:
        base_image = resize_image(st.session_state.annotated_image, 500, 500)
        
        if st.session_state.count == 0:
            st.session_state.original_resized = base_image.copy()

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=base_image,
            update_streamlit=True,
            height=500,
            width=500,
            drawing_mode="polygon",
            key="canvas",
        )

        if canvas_result.json_data and canvas_result.json_data["objects"]:
            for obj in canvas_result.json_data["objects"]:
                path_data = obj["path"]
                polygon_points = [(pt[1], pt[2]) for pt in path_data if pt[0] in ['L', 'M']]
                ImageDraw.Draw(st.session_state.original_resized).polygon(polygon_points, outline="blue", width=2)
                st.session_state.tempList[f"label_{st.session_state.count}"] = polygon_points
                st.session_state.count += 1

            col1, col2 = st.columns(2)
            col1.image(st.session_state.original_resized_image, caption="Original Image")
            col2.image(st.session_state.original_resized, caption="Annotated Image")

            if st.button("Show Generated Mask", type="primary"):
                st.session_state.black_image = draw_points_on_black_image(
                    st.session_state.original_resized, st.session_state.tempList
                )
                black_image_pil = Image.fromarray(st.session_state.black_image)
                col1.image(st.session_state.original_resized, caption="Annotated Image")
                col2.image(black_image_pil, caption="Generated Mask")

            if st.button("Done", type="primary"):
                st.session_state.draw_mode = False
                st.session_state.done_clicked = True

            if st.session_state.done_clicked:
                mask_path = os.path.join(ANNOTATED_MASK_PATH,st.session_state.file_name)
                cv2.imwrite(mask_path, st.session_state.black_image)

                for key in [
                    "file_name", "uploaded_image", "annotated_image",
                    "annotated_image_pil", "original_resized", "count",
                    "draw_mode", "black_image", "done_clicked", "tempList"
                ]:
                    st.session_state.pop(key, None)

                st.success("✅ Annotation saved. Upload a new image to start again!")
                st.experimental_rerun()

if __name__ == "__main__":
    main()
