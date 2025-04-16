import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH=os.path.join(BASE_DIR,'test_images')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
ANNOTATED_MASK_PATH = os.path.join(BASE_DIR, "annotated_mask")
