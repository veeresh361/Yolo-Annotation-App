from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def load_image(file):
    return Image.open(file).convert("RGB")



def resize_image(image_np,orig_width,orig_height):
    image=Image.fromarray(image_np)
    MAX_CANVAS_WIDTH = 500
    scale = MAX_CANVAS_WIDTH / orig_width
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def draw_points_on_black_image(pil_image, point_dict):
    """
    Create a black image (NumPy) of the same shape as the given PIL image,
    and set specific (x, y) coordinates to white (255) based on the input dictionary.

    Args:
        pil_image (PIL.Image): Input image to get shape.
        point_dict (dict): Dictionary with values as list of (x, y) tuples.

    Returns:
        np.ndarray: Black image with specified points set to 255 (white).
    """
    width, height = pil_image.size
    black_img = np.zeros((height, width), dtype=np.uint8)  # Grayscale black image
    for key, points in point_dict.items():
        # Convert list of tuples -> numpy array shape (n_points, 1, 2)
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Fill polygon with white
        cv2.fillPoly(black_img, [contour], color=(255, 255, 255))  # White fill

        # Draw red contour on top
        cv2.polylines(black_img, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
    return black_img
