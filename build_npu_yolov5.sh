#!/bin/bash

# Purpose: This script natively compiles the Radxa NPU YOLOv5 example directly on the board,
# bypassing the need for a complex Tina Linux cross-compilation environment on a host PC.
# It requires libopencv-dev to be installed on the board.

echo "Installing OpenCV headers and build dependencies..."
sudo apt-get update
sudo apt-get install -y libopencv-dev g++

echo "Moving to the YOLOv5 example directory..."
cd /home/iammhk/ai-sdk/examples/yolov5/

echo "Compiling the NPU C++ application natively..."
g++ -g -O2 main.c yolov5_post_process.cpp yolov5_pre_process.cpp \
    ../libawnn_viplite/awnn_lib.c \
    ../libawnn_viplite/awnn_quantize.c \
    ../libawutils/image_utils.c \
    -o yolov5_npu \
    -I/usr/include/opencv4 \
    -I../libawnn_viplite \
    -I../libawutils \
    -I../../ \
    -I../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/inc \
    -L../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/ \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -ljpeg \
    -lNBGlinker -lVIPhal -lm \
    -DNPU_SW_VERSION=2

if [ -f "yolov5_npu" ]; then
    echo "Compilation successful! Executable created: ~/ai-sdk/examples/yolov5/yolov5_npu"
    echo "To test it manually on a single image:"
    echo "export LD_LIBRARY_PATH=~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/"
    echo "./yolov5_npu ./model/v3/yolov5.nb ./input_data/dog.jpg"
else
    echo "Compilation failed."
fi
