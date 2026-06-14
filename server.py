"""
server.py
─────────
Unified Web & ML Server for Crop Disease Classifier.
Serves the UI on port 8000 and handles /predict uploads.
Model loading happens in a background thread so the server starts instantly.
"""
import os, sys, json, tempfile, traceback, re, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Model state - protected by lock
_lock    = threading.Lock()
_state   = {"status": "loading", "model": None, "transform": None,
             "gradcam": None, "classes": None, "device": None}

def _load_model_thread():
    try:
        print("[ML] Starting background model load (importing PyTorch)...", flush=True)
        import torch, yaml
        from model   import build_model
        from dataset import get_val_transforms
        from predict import GradCAM

        print("[ML] PyTorch loaded. Reading config...", flush=True)
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classes = cfg["classes"]
        model   = build_model(cfg).to(device)

        ckpt = "outputs/checkpoints/best_model.pth"
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"])
            print(f"[ML] Checkpoint loaded from {ckpt}.", flush=True)
        else:
            print("[ML] No checkpoint found. Using untrained weights for demonstration.", flush=True)

        model.eval()

        with _lock:
            _state["model"]     = model
            _state["transform"] = get_val_transforms(cfg["data"]["image_size"])
            _state["gradcam"]   = GradCAM(model)
            _state["classes"]   = classes
            _state["device"]    = device
            _state["status"]    = "ready"

        print("[ML] Model ready! Predictions are now live.", flush=True)

    except Exception:
        with _lock:
            _state["status"] = "error"
        print("[ML] ❌ Error loading model:", flush=True)
        traceback.print_exc()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def parse_multipart(headers, body: bytes):
    """Simple parser for multipart/form-data."""
    ct = headers.get("Content-Type", "")
    m = re.search(r'boundary=([^\s;]+)', ct)
    if not m: return {}
    bnd = m.group(1).encode()
    result = {}
    for part in body.split(b'--' + bnd)[1:]:
        if part.strip() in (b'', b'--'): continue
        sep = b'\r\n\r\n' if b'\r\n\r\n' in part else b'\n\n'
        if sep not in part: continue
        hdr_raw, data = part.split(sep, 1)
        data = data.rstrip(b'\r\n--')
        nm = re.search(r'name="([^"]+)"', hdr_raw.decode(errors='replace'))
        if nm: result[nm.group(1)] = data
    return result

CONTENT_TYPES = {
    ".html": "text/html",
    ".css":  "text/css",
    ".js":   "application/javascript",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Only log errors/warnings to keep console clean, or log everything if debugging
        print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        
        # API Routes
        if path == "/status":
            with _lock: status = _state["status"]
            return self._ok({"status": status})
            
        # Static file routing
        if path == "/" or path == "":
            path = "/index.html"

        if path.startswith("/outputs/"):
            file_path = os.path.join(PROJECT_ROOT, path.lstrip("/"))
        else:
            file_path = os.path.join(STATIC_DIR, path.lstrip("/"))

        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            ct = CONTENT_TYPES.get(ext, "application/octet-stream")
            with open(file_path, "rb") as f: data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(data))
            self._cors(); self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
            
        try:
            with _lock:
                status = _state["status"]
                model  = _state["model"]
                tf     = _state["transform"]
                gc     = _state["gradcam"]
                cls    = _state["classes"]
                dev    = _state["device"]

            if status == "loading":
                return self._err(503, "Model is still loading in the background. Please wait a moment.")
            if status == "error":
                return self._err(500, "Model failed to load on the server. Check logs.")

            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            fields = parse_multipart(self.headers, body)

            img_bytes = fields.get("file")
            if not img_bytes:
                return self._err(400, "No 'file' field found in form data.")

            from predict import predict_single

            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd, "wb") as f: f.write(img_bytes)

            out_dir = Path(PROJECT_ROOT) / "outputs" / "gradcam"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = predict_single(
                    image_path=tmp, model=model, transform=tf,
                    classes=cls, device=dev, gradcam=gc, output_dir=out_dir
                )
            finally:
                if os.path.exists(tmp): os.remove(tmp)

            # Convert gradcam absolute path to URL path
            result["gradcam_url"] = "/outputs/gradcam/" + Path(result["gradcam"]).name
            self._ok(result)

        except Exception as e:
            traceback.print_exc()
            self._err(500, f"Internal Server Error: {str(e)}")

    def _ok(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self._cors(); self.end_headers()
        self.wfile.write(data)

    def _err(self, code, msg):
        data = json.dumps({"error": str(msg)}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self._cors(); self.end_headers()
        self.wfile.write(data)

if __name__ == "__main__":
    # Start model loading in a background thread
    t = threading.Thread(target=_load_model_thread, daemon=True)
    t.start()

    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Unified Server on http://0.0.0.0:{port}/")
    print(f"   (Model is loading in the background. UI is available immediately.)")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
