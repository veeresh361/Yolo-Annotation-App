import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from config import YOLO_MODEL_PATH,ANNOTATED_MASK_PATH
from utils import (init_session_state,resize_image,draw_points_on_black_image,detect_objects)
init_session_state()
image_path="/home/coe2/Veeresh/test_images/"

images=os.listdir(image_path)

if st.button("start", type="primary"):
  st.session_state.file_name =image_path+images[st.session_state.counter]
  st.session_state.uploaded_image = Image.open(st.session_state.file_name)
  st.image(st.session_state.uploaded_image, caption="Original Image", width=500)


if st.button("Done", type="primary"):
  st.session_state.draw_mode = False
  st.session_state.done_clicked = True

if st.session_state.done_clicked:
    # mask_path = os.path.join(ANNOTATED_MASK_PATH,st.session_state.file_name)
    # cv2.imwrite(mask_path, st.session_state.black_image)

    for key in [
        "file_name", "uploaded_image", "annotated_image",
        "annotated_image_pil", "original_resized", "count",
        "draw_mode", "black_image", "done_clicked", "tempList"
    ]:
        st.session_state.pop(key, None)
    st.session_state.counter+=1
    st.success("✅ Annotation saved. Upload a new image to start again!")
    st.experimental_rerun()
