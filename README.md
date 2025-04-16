# 🧠 YOLOv11 + Streamlit App (Dockerized)

This project runs a custom computer vision app powered by **YOLOv11** and **Streamlit**, packaged inside a fully isolated Docker environment. It supports interactive drawing, object detection, and visual experimentation — perfect for rapid prototyping and demos.

---
## 🚀 About YOLOv11

**YOLOv11** is part of the Ultralytics YOLO family — designed for **real-time object detection**. It builds upon YOLOv8 and earlier with:

- ✅ Enhanced transformer-based backbone
- ⚡ Faster inference and higher accuracy
- 🔁 Out-of-the-box support via `ultralytics` Python package
- 🎯 Great for applications like live video detection, object tracking, and image annotation

[Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

## 🐳 Dockerized Environment

Your app runs in a **Python 3.11 slim** container with all the needed dependencies pre-installed.

### ✅ Installed Components:
- `streamlit==1.40.0`
- `ultralytics>=8.0.0`
- `opencv-python`
- `pandas`, `numpy`
- `streamlit-drawable-canvas`
- `Pillow`

---

---

## 🛠️ How to Build and Run the App

### 1️⃣ Build the Docker Image

Make sure you're in your project folder (with the Dockerfile), then run:

```bash
docker build -t yolo-streamlit-app .

docker run -it --rm -v "%cd%:/app" -w /app -p 8501:8501 my-streamlit-app

streamlit run app_1.1.py --server.address=0.0.0.0.```
