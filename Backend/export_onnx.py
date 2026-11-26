# export_onnx.py - run once to convert the PyTorch checkpoint into a small ONNX model
# Usage:  pip install torch torchvision onnx  &&  python export_onnx.py
import torch
from model import ColorNet

model = ColorNet()
checkpoint = torch.load("pretrained/checkpoint-trained.pth.tar", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

dummy = torch.rand(1, 1, 224, 224)
torch.onnx.export(model, dummy, "pretrained/colornet.onnx",
                  input_names=["L"], output_names=["ab"], opset_version=17, dynamo=False)
print("Saved pretrained/colornet.onnx")
