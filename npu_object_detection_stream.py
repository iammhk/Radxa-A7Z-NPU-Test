"""
Purpose: Object detection streaming app using Radxa NPU (Vivante).
This script captures the camera feed, invokes the natively compiled C++ yolov5_npu binary
to run inference on the hardware NPU, parses the output, and streams it.
"""

import cv2
import subprocess
import re
import os
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
    info['npu'] = "Active (Hardware Accelerated)"
    info['recent_detections'] = recent_detections
    return info

PORT = 8082
CAMERA_INDEX = 0

# Adjust paths to match your board
NPU_BIN = "/home/iammhk/ai-sdk/examples/yolov5/yolov5_npu"
MODEL_NB = "/home/iammhk/ai-sdk/examples/yolov5/model/v3/yolov5.nb"
LD_LIBRARY_PATH = "/home/iammhk/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/"

TEMP_IMAGE_PATH = "/tmp/temp_npu_frame.jpg"

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
                # Add library path to environment for subprocess
                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

                while True:
                    rc, frame = cap.read()
                    if not rc:
                        continue
                        
                    # Save frame to RAM disk (/tmp) for the C++ NPU binary to process
                    cv2.imwrite(TEMP_IMAGE_PATH, frame)
                    
                    # Run the NPU binary
                    try:
                        result = subprocess.run([NPU_BIN, MODEL_NB, TEMP_IMAGE_PATH],
                                                capture_output=True, text=True, env=env)
                        output = result.stderr  # The C++ binary prints detections to stderr!
                    except Exception as e:
                        print(f"Failed to run NPU binary: {e}")
                        continue
                        
                    # Parse the C++ standard output for bounding boxes
                    # Example format: "16:  83%, [ 113,  249,  254,  594], dog"
                    lines = output.split('\n')
                    for line in lines:
                        match = re.search(r'(\d+):\s+(\d+)%,\s+\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s+(.*)', line)
                        if match:
                            cls_id = int(match.group(1))
                            conf = int(match.group(2))
                            left = int(match.group(3))
                            top = int(match.group(4))
                            right = int(match.group(5))
                            bottom = int(match.group(6))
                            label = match.group(7)
                            
                            # Draw the bounding box
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label}: {conf}%", (left, top - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        
                            global recent_detections
                            current_time = time.strftime('%H:%M:%S')
                            recent_detections = [d for d in recent_detections if d['label'] != label]
                            recent_detections.insert(0, {'label': label, 'time': current_time, 'conf': f"{conf}%"})
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
            <html><head><title>Radxa NPU Stream</title>
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
                <h1>Hardware Accelerated NPU Stream</h1>
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
                            <div class="stat-row"><span class="stat-label">NPU Status:</span><span class="stat-value" id="npu" style="color: #00bcd4;">...</span></div>
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
        print(f"Starting hardware NPU server on port {PORT}...")
        print(f"View the stream at http://192.168.29.58:{PORT}/")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.socket.close()

if __name__ == '__main__':
    main()
