"""
Purpose: Object detection streaming app using YOLOv5 via OpenCV DNN.
This script streams the camera feed with bounding boxes over the local network.

Note: Since the Radxa Vivante AI SDK currently only supports C/C++ (via libNBGlinker.so) 
and lacks Python bindings, this script uses the provided YOLOv5 ONNX model 
via OpenCV's DNN module as an immediate solution. To fully utilize the NPU hardware acceleration, 
the C++ `yolov5` example in the SDK must be cross-compiled on a Linux host and integrated.
"""

import cv2
import numpy as np
import subprocess
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

recent_detections = []

def get_sysinfo():
    info = {}
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000.0
            info['temperature'] = f"{temp:.1f} °C"
    except:
        info['temperature'] = "N/A"
    try:
        free_out = subprocess.check_output(["free", "-m"], text=True).splitlines()[1].split()
        total_ram = int(free_out[1])
        used_ram = int(free_out[2])
        info['ram'] = f"{used_ram}MB / {total_ram}MB ({(used_ram/total_ram*100):.1f}%)"
    except:
        info['ram'] = "N/A"
    try:
        df_out = subprocess.check_output(["df", "-h", "/"], text=True).splitlines()[1].split()
        info['storage'] = f"{df_out[2]} / {df_out[1]} ({df_out[4]})"
    except:
        info['storage'] = "N/A"
    try:
        with open("/proc/loadavg", "r") as f:
            load = f.read().split()[:3]
            info['cpu'] = f"Load: {load[0]}, {load[1]}, {load[2]}"
    except:
        info['cpu'] = "N/A"
    info['npu'] = "Inactive (Using OpenCV CPU)"
    info['recent_detections'] = recent_detections
    return info

PORT = 8081
CAMERA_INDEX = 0
MODEL_PATH = '/home/iammhk/ai-sdk/models/yolov5s-sim/yolov5s-sim.onnx'

# Load the YOLOv5 ONNX model using OpenCV
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Basic COCO class names for YOLOv5
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            cap = cv2.VideoCapture(CAMERA_INDEX)
            
            if not cap.isOpened():
                print(f"Error: Could not open camera {CAMERA_INDEX}")
                return

            try:
                while True:
                    rc, frame = cap.read()
                    if not rc:
                        continue
                        
                    # YOLOv5 processing
                    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                    net.setInput(blob)
                    outputs = net.forward()
                    
                    # Post-processing
                    preds = outputs[0]  # shape (25200, 85)
                    rows = preds.shape[0]
                    image_height, image_width = frame.shape[:2]
                    
                    x_factor = image_width / 640
                    y_factor = image_height / 640
                    
                    conf_threshold = 0.2
                    score_threshold = 0.2
                    nms_threshold = 0.4
                    
                    class_ids = []
                    confidences = []
                    boxes = []
                    
                    for r in range(rows):
                        row = preds[r]
                        confidence = row[4]
                        if confidence >= conf_threshold:
                            classes_scores = row[5:]
                            class_id = np.argmax(classes_scores)
                            if (classes_scores[class_id] > score_threshold):
                                confidences.append(float(confidence))
                                class_ids.append(int(class_id))
                                
                                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                                left = int((x - 0.5 * w) * x_factor)
                                top = int((y - 0.5 * h) * y_factor)
                                width = int(w * x_factor)
                                height = int(h * y_factor)
                                box = np.array([left, top, width, height])
                                boxes.append(box)
                    
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                    
                    # Draw boxes
                    if len(indices) > 0:
                        for i in np.array(indices).flatten():
                            box = boxes[i]
                            left, top, width, height = box[0], box[1], box[2], box[3]
                            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                            
                            label_name = classes[class_ids[i]]
                            conf_val = confidences[i]
                            label = f"{label_name}:{conf_val:.2f}"
                            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            global recent_detections
                            current_time = time.strftime('%H:%M:%S')
                            recent_detections = [d for d in recent_detections if d['label'] != label_name]
                            recent_detections.insert(0, {'label': label_name, 'time': current_time, 'conf': f"{int(conf_val*100)}%"})
                            recent_detections = recent_detections[:3]
                    
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                        
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(f"Stream stopped: {e}")
            finally:
                cap.release()
                
        elif self.path == '/sysinfo':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(get_sysinfo()).encode())
            
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html><head><title>Object Detection Stream</title>
            <style>
                body { background:#111; color:#eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; margin: 0; padding: 20px; }
                .container { display: flex; flex-direction: row; gap: 20px; max-width: 1200px; width: 100%; justify-content: center; flex-wrap: wrap; }
                .video-card { background: #222; padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); }
                .info-card { background: #222; padding: 25px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); min-width: 300px; display: flex; flex-direction: column; gap: 15px; height: fit-content; }
                img { max-width: 100%; height: auto; border-radius: 8px; }
                h1 { margin-bottom: 30px; font-weight: 300; }
                h2 { margin-top: 0; font-size: 1.2rem; border-bottom: 1px solid #444; padding-bottom: 10px; }
                .stat-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding: 8px 0; }
                .stat-label { color: #aaa; font-size: 0.9rem; }
                .stat-value { font-weight: bold; font-size: 1.1rem; color: #4CAF50; }
            </style>
            <script>
                function updateSysInfo() {
                    fetch('/sysinfo')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('temp').innerText = data.temperature;
                            document.getElementById('cpu').innerText = data.cpu;
                            document.getElementById('ram').innerText = data.ram;
                            document.getElementById('storage').innerText = data.storage;
                            document.getElementById('npu').innerText = data.npu;
                            
                            let detHtml = '';
                            if (data.recent_detections && data.recent_detections.length > 0) {
                                data.recent_detections.forEach(d => {
                                    detHtml += `<div class="stat-row"><span class="stat-label">${d.time}</span><span class="stat-value">${d.label.toUpperCase()} (${d.conf})</span></div>`;
                                });
                            } else {
                                detHtml = '<div style="color:#aaa; text-align:center; padding: 10px; font-style:italic;">Waiting for objects...</div>';
                            }
                            document.getElementById('detections-list').innerHTML = detHtml;
                        });
                }
                setInterval(updateSysInfo, 2000);
                window.onload = updateSysInfo;
            </script>
            </head>
            <body>
                <h1>Object Detection Stream (YOLOv5)</h1>
                <div class="container">
                    <div class="video-card">
                        <img src="cam.mjpg" />
                    </div>
                    <div style="display:flex; flex-direction:column; gap:20px;">
                        <div class="info-card">
                            <h2>System Dashboard</h2>
                            <div class="stat-row"><span class="stat-label">Temperature:</span><span class="stat-value" id="temp">...</span></div>
                            <div class="stat-row"><span class="stat-label">CPU Load:</span><span class="stat-value" id="cpu">...</span></div>
                            <div class="stat-row"><span class="stat-label">RAM Usage:</span><span class="stat-value" id="ram">...</span></div>
                            <div class="stat-row"><span class="stat-label">Storage (Root):</span><span class="stat-value" id="storage">...</span></div>
                            <div class="stat-row"><span class="stat-label">NPU Status:</span><span class="stat-value" id="npu" style="color: #ff9800;">...</span></div>
                        </div>
                        <div class="info-card">
                            <h2>Recent Detections</h2>
                            <div id="detections-list">
                                <div style="color:#aaa; text-align:center; padding: 10px; font-style:italic;">Waiting for objects...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </body></html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    try:
        server = ThreadedHTTPServer(('', PORT), CamHandler)
        print(f"Starting object detection server on port {PORT}...")
        print(f"View the stream at http://192.168.29.58:{PORT}/")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.socket.close()

if __name__ == '__main__':
    main()
