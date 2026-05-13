import http.server
import socketserver
import os
import json

PORT = 8000

class DataHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # List available documents and their results
            data = {
                "documents": [],
                "results": {}
            }
            
            if os.path.exists('input'):
                data["documents"] = [f for f in os.listdir('input') if f.endswith('.txt')]
            
            if os.path.exists('output/result.json'):
                try:
                    with open('output/result.json', 'r', encoding='utf-8') as f:
                        data["results"] = json.load(f)
                except Exception as e:
                    print(f"Error loading result.json: {e}")
                    data["results"] = {"error": "Invalid result.json. Please re-run the workflow."}
            
            self.wfile.write(json.dumps(data).encode())

        elif self.path.startswith('/api/file/'):
            filename = self.path.replace('/api/file/', '')
            filepath = os.path.join('input', filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            return super().do_GET()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("", PORT), DataHandler) as httpd:
    print(f"Data Server at port {PORT}")
    httpd.serve_forever()
