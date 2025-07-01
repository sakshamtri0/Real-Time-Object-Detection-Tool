# 🧠 Real-Time Object Detection Tool

A powerful and lightweight real-time object detection application built using **OpenCV**, **YOLOv3**, and **PyQt5** for a user-friendly GUI experience. This tool allows you to perform live object detection using your webcam or a video file.

---

## 🚀 Features

- 🎥 Real-time object detection using YOLOv3
- 🖥️ Simple and responsive PyQt5 GUI
- 📂 Supports webcam and video file input
- 💾 Displays detection confidence and class labels
- ⚡ Lightweight and efficient

---

## 📸 Demo

![Demo GIF](https://github.com/yourusername/Real-Time-Object-Detection-Tool/assets/demo.gif)

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv3
- PyQt5
- NumPy

---

## 📁 Project Structure
```
Real-Time-Object-Detection-Tool/
│
├── yolov3.cfg
├── yolov3.weights
├── coco.names
├── main.py
├── gui.ui
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Real-Time-Object-Detection-Tool.git
cd Real-Time-Object-Detection-Tool
```

### 2. Create a Virtual Environment (Optional)

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Download YOLOv3 Weights

Download yolov3.weights from the official YOLO website and place it in the project folder.

---

## ▶️ Running the App
```
python main.py
```

---

## 📌 Notes

- Make sure you have a good webcam or use a sample video for testing.
- You can modify the confidence threshold in the code for more accurate results.
