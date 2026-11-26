# server.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
from skimage.color import rgb2gray, lab2rgb
from model import ColorNet
from db import init_db
from auth import auth_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(auth_bp)
init_db()  # Create DB if not exists

# Load model once
model = ColorNet()
checkpoint = torch.load("pretrained/checkpoint-trained.pth.tar", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "PRISMIFY API is running"})

@app.route("/colorize", methods=["POST"])
def colorize_image():
    if "image" not in request.files:
        return jsonify({"message": "No image uploaded"}), 400

    image = Image.open(request.files["image"].stream).convert("RGB")
    orig_w, orig_h = image.size

    # Keep large uploads reasonable for a free CPU server
    max_side = 1024
    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        image = image.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        orig_w, orig_h = image.size

    # Full-resolution lightness channel (keeps all the detail)
    L_full = rgb2gray(np.asarray(image))

    # The model predicts colour at 224x224
    small = np.asarray(image.resize((224, 224), Image.LANCZOS))
    L_small = torch.from_numpy(rgb2gray(small)).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output_ab = model(L_small)

    # Scale the predicted colour back up to the original size
    output_ab = torch.nn.functional.interpolate(
        output_ab, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    output_ab = output_ab.squeeze(0).cpu().numpy().transpose((1, 2, 0))

    lab_output = np.zeros((orig_h, orig_w, 3))
    lab_output[:, :, 0] = L_full * 100
    lab_output[:, :, 1:] = output_ab * 255 - 128
    rgb_result = (np.clip(lab2rgb(lab_output), 0, 1) * 255).astype(np.uint8)

    img_bytes = io.BytesIO()
    Image.fromarray(rgb_result).save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype="image/png")

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))