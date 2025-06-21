# Real-Time Object Detection Tool

This repository contains a real-time object detection application using deep learning and computer vision techniques. The tool leverages OpenCV and a pre-trained model to identify objects from a live webcam feed or video input.

## 🔍 Features

- Real-time object detection using OpenCV.
- Supports pre-trained YOLOv3 model.
- Configurable for custom classes.
- Displays object labels and confidence scores.
- Option to save output video with detections.

## 🛠️ Technologies Used

- Python 3
- OpenCV
- NumPy
- YOLOv3 (You Only Look Once version 3)
- COCO Dataset (Common Objects in Context)

## 🧠 Model Information

- **Model Architecture:** YOLOv3 (Darknet-53 as backbone)
- **Framework:** OpenCV DNN module
- **Trained On:** COCO dataset (80 object classes)
- **Model Files:**
  - `yolov3.cfg` – Model configuration file
  - `yolov3.weights` – Pre-trained weights file
  - `coco.names` – Class label definitions
