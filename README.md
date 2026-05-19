<!-- This file serves as the main documentation for the Radxa-A7Z-NPU-Test repository, detailing the basic, CPU, and hardware-accelerated NPU streaming pipelines. Used in the actual project. -->
# Radxa A7Z NPU Object Detection Setup

This repository contains tools, wrappers, and compilation helper scripts to achieve real-time, hardware-accelerated object detection on the **Radxa Cubie A7Z** board (Allwinner A733) using its onboard Vivante NPU. 

It provides four key components:
1. A baseline camera stream.
2. A CPU-based YOLOv5 object detection stream with a real-time system stats dashboard.
3. A native, on-board C++ compiler script for the Radxa AI SDK.
4. A hardware-accelerated NPU object detection stream utilizing the compiled C++ NPU application.

---

## 🛠️ Repository Components

### 1. `camera_stream.py`
A lightweight Python MJPEG streamer using OpenCV and `http.server`. It grabs raw frames from the USB endoscope camera (`/dev/video0`) and serves them over the network.
* **Port:** `8080`
* **Run:** `sudo python3 camera_stream.py`

### 2. `object_detection_stream.py` (CPU Version)
An advanced object detection server running YOLOv5 (`yolov5s-sim.onnx`) directly on the CPU using OpenCV's DNN module.
* **Port:** `8081`
* **Features:**
  * Real-time AJAX-powered **System Dashboard** card (showing CPU load, RAM usage, storage space, and board temperature).
  * **Recent Detections** card showing the last 3 unique objects identified in a rolling queue.
  * Sensitivity tweaked to `0.2` (20%) confidence to account for blurry endoscope imagery.
* **Run:** `sudo python3 object_detection_stream.py`

### 3. `build_npu_yolov5.sh` (Board Compiler)
To bypass the complex cross-compilation toolchain on a separate Linux host, this shell script installs OpenCV dev dependencies on the board and natively compiles the SDK's C++ NPU demo (`main.c` + `yolov5_post_process.cpp`).
* **Compiles:** Creates the native executable `yolov5_npu` in the SDK folder.
* **Run:** `sudo bash build_npu_yolov5.sh`

### 4. `npu_object_detection_stream.py` (NPU Accelerated)
The flagship application. It captures the camera feed, dumps the frames to a RAM disk (`/tmp`), and invokes the compiled C++ `yolov5_npu` binary. It parses the C++ `stderr` predictions stream, draws bounding boxes, and serves the accelerated MJPEG stream.
* **Port:** `8082`
* **Features:**
  * **Hardware-Accelerated Inference** using the A7Z Vivante NPU.
  * The same modern, sleek AJAX **System Dashboard** and **Recent Detections** card.
* **Run:** `sudo python3 npu_object_detection_stream.py`

---

## 🚀 Getting Started

### Prerequisites on the Board
Make sure OpenCV and Python 3 are installed on the board:
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-numpy
```

### Installation & Run Steps

1. **Clone the AI SDK** on the Radxa board to `~/ai-sdk/`.
2. **Natively compile the NPU app**:
   ```bash
   sudo bash build_npu_yolov5.sh
   ```
3. **Launch the Hardware-Accelerated Stream**:
   ```bash
   sudo python3 npu_object_detection_stream.py
   ```
4. **Access the Stream**:
   Open a web browser and navigate to:
   `http://192.168.29.58:8082/`

---

## 📈 System Stats Dashboard

The web interface is styled with a sleek, dark-mode design optimized for embedded devices. It is split into two panels:
- **Left Panel:** Live Video Feed with overlayed YOLOv5 detection bounding boxes.
- **Right Panel:**
  * **System Dashboard:** Live temperature monitoring (helping you keep track of thermal performance when running heavy detection models), CPU load averages, memory, and root partition storage.
  * **Recent Detections:** A time-stamped history of the last 3 uniquely identified objects.
