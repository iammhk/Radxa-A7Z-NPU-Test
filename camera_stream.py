"""
Purpose: This script creates a lightweight, local HTTP server to stream MJPEG video from a connected USB camera using OpenCV and the built-in http.server.
Usage: It is a temporary script to test the endoscope camera on the Radxa board and view the feed from any device on the local network.
"""

import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Settings
PORT = 8080
CAMERA_INDEX = 0

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
                    rc, img = cap.read()
                    if not rc:
                        continue
                    
                    ret, jpeg = cv2.imencode('.jpg', img)
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
                
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head><title>Camera Stream</title></head>')
            self.wfile.write(b'<body style="text-align:center; background:#222; color:#fff; font-family:sans-serif;">')
            self.wfile.write(b'<h1>Radxa Endoscope Stream</h1>')
            self.wfile.write(b'<img src="cam.mjpg" style="max-width: 100%; height: auto; border: 2px solid #555; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);"/>')
            self.wfile.write(b'</body></html>')
        else:
            self.send_response(404)
            self.end_headers()

def main():
    try:
        server = ThreadedHTTPServer(('', PORT), CamHandler)
        print(f"Starting server on port {PORT}...")
        print(f"View the stream at http://192.168.29.58:{PORT}/")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.socket.close()

if __name__ == '__main__':
    main()
