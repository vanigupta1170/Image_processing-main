# server.py  (PRISMIFY backend - lightweight ONNX version, no PyTorch needed)
import io
import os

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from skimage.color import rgb2gray, lab2rgb

from db import init_db
from auth import auth_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(auth_bp)
init_db()  # Create DB if not exists

# Load the ONNX model once (converted from checkpoint-trained.pth.tar)
session = ort.InferenceSession("pretrained/colornet.onnx", providers=["CPUExecutionProvider"])


def resize_channel(channel, width, height):
    """Bilinear-resize one float channel to (width, height)."""
    img = Image.fromarray(channel.astype(np.float32), mode="F")
    return np.asarray(img.resize((width, height), Image.BILINEAR))


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "PRISMIFY API is running"})


@app.route("/colorize", methods=["POST"])
def colorize_image():
    if "image" not in request.files:
        return jsonify({"message": "No image uploaded"}), 400

    image = Image.open(request.files["image"].stream).convert("RGB")

    # Keep large uploads reasonable for a free server
    max_side = 1024
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    w, h = image.size

    # Full-resolution lightness channel (keeps all the detail)
    L_full = rgb2gray(np.asarray(image))

    # The model predicts colour at 224x224
    small = np.asarray(image.resize((224, 224), Image.LANCZOS))
    L_small = rgb2gray(small).astype(np.float32)[None, None, :, :]
    ab = session.run(None, {"L": L_small})[0][0]  # shape (2, 224, 224)

    # Scale the predicted colour back up to the original size
    a = resize_channel(ab[0], w, h)
    b = resize_channel(ab[1], w, h)

    lab = np.zeros((h, w, 3))
    lab[:, :, 0] = L_full * 100
    lab[:, :, 1] = a * 255 - 128
    lab[:, :, 2] = b * 255 - 128
    rgb = (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
