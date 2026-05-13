import http.server
import socketserver
import json
import os
from urllib.parse import unquote

PORT = 8000
INPUT_DIR = "input"
OUTPUT_DIR = "output"

class DataHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def do_GET(self):
        if self.path == '/api/data':
            # Returns the list of all available documents
            docs = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"documents": docs}).encode())
            
        elif self.path.startswith('/api/file/'):
            filename = unquote(self.path.replace('/api/file/', ''))
            filepath = os.path.join(INPUT_DIR, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")

        elif self.path.startswith('/api/results/'):
            # New endpoint: Fetch results for a specific document
            # Request: /api/results/tender_name.txt -> Looks for output/tender_name.json
            filename = unquote(self.path.replace('/api/results/', ''))
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(OUTPUT_DIR, json_filename)
            
            # Fallback to result.json if specific one doesn't exist yet
            if not os.path.exists(json_path):
                json_path = os.path.join(OUTPUT_DIR, "result.json")

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                except Exception as e:
                    self.send_error(500, f"Error reading results: {str(e)}")
            else:
                # If no results at all, return empty object
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({}).encode())
        else:
            super().do_GET()

print(f"Data Server at port {PORT}")
with socketserver.TCPServer(("", PORT), DataHandler) as httpd:
    httpd.serve_forever()
