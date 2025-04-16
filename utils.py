from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import streamlit as st


def init_session_state():
    defaults = {
        "file_name": None,
        "uploaded_image": None,
        "original_resized_image": None,
        "annotated_image": None,
        "draw_mode": False,
        "start_button_pressed": None,
        "tempList": {},
        "black_image": None,
        "done_clicked": False,
        "counter":0,
        'count':0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# def load_image(file):
#     return Image.open(file).convert("RGB")



# def resize_image(image_np,orig_width,orig_height):
#     image=Image.fromarray(image_np)
#     MAX_CANVAS_WIDTH = 500
#     scale = MAX_CANVAS_WIDTH / orig_width
#     new_width = int(orig_width * scale)
#     new_height = int(orig_height * scale)
#     resized_image = image.resize((new_width, new_height))
#     return resized_image

# def draw_points_on_black_image(pil_image, point_dict):
#     """
#     Create a black image (NumPy) of the same shape as the given PIL image,
#     and set specific (x, y) coordinates to white (255) based on the input dictionary.

#     Args:
#         pil_image (PIL.Image): Input image to get shape.
#         point_dict (dict): Dictionary with values as list of (x, y) tuples.

#     Returns:
#         np.ndarray: Black image with specified points set to 255 (white).
#     """
#     width, height = pil_image.size
#     black_img = np.zeros((height, width), dtype=np.uint8)  # Grayscale black image
#     for key, points in point_dict.items():
#         # Convert list of tuples -> numpy array shape (n_points, 1, 2)
#         contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

#         # Fill polygon with white
#         cv2.fillPoly(black_img, [contour], color=(255, 255, 255))  # White fill

#         # Draw red contour on top
#         cv2.polylines(black_img, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
#     return black_img



def resize_image(image, width, height):
    """
    Resize a given image to the specified width and height.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        width (int): Target width in pixels.
        height (int): Target height in pixels.

    Returns:
        PIL.Image.Image: Resized image as a PIL Image object.
    """
    return Image.fromarray(image).resize((width, height))

def detect_objects(image,model):
    """
    Perform object detection on the given image using the YOLO model.

    Args:
        image (PIL.Image.Image): Input image as a PIL Image.

    Returns:
        np.ndarray: Annotated image with detection results plotted on it.
    """
    image_np = np.array(image)
    results = model(image_np)
    return results[0].plot()

def draw_points_on_black_image(image, annotations):
    """
    Draw labeled polygons on a black image based on user annotations.

    Args:
        image (PIL.Image.Image): Reference image to determine size.
        annotations (dict): Dictionary containing polygon points as values.

    Returns:
        np.ndarray: Black image with white polygons drawn over it.
    """
    black_image = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(black_image)
    for points in annotations.values():
        draw.polygon(points, outline="white", fill="white")
    return np.array(black_image)

