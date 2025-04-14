import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load image
image = Image.open("D:\\yolov5\\ultralytics\\bus.jpg")
orig_width, orig_height = image.size

# Resize for canvas while preserving aspect ratio
MAX_CANVAS_WIDTH = 500
scale = MAX_CANVAS_WIDTH / orig_width
canvas_width = int(orig_width * scale)
canvas_height = int(orig_height * scale)
resized_image = image.resize((canvas_width, canvas_height))

# Display canvas for drawing multiple rectangles
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=resized_image,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="rect",   # still 'rect' mode
    key="canvas",
)

# Handle all rectangles drawn by user
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    if objects:
        st.markdown("### ✅ Rectangles Drawn:")
        for i, obj in enumerate(objects):
            left = obj["left"] / scale
            top = obj["top"] / scale
            width = obj["width"] / scale
            height = obj["height"] / scale

            x1, y1 = round(left), round(top)
            x2, y2 = round(left + width), round(top + height)

            st.write(f"**Box {i+1}:** ({x1}, {y1}) → ({x2}, {y2})")
