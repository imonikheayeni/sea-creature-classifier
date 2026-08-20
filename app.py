"""Interactive Gradio demo: drag in a photo, get the predicted sea creature.

Great for embedding in a portfolio site or hosting free on Hugging Face Spaces.

Run locally:
    python app.py

Then open the printed local URL. Set share=True below for a temporary public link.
"""

from __future__ import annotations

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import build_model
from src.data import build_transforms
from src.train import pick_device

CKPT_PATH = "outputs/best_model.pt"

device = pick_device()
ckpt = torch.load(CKPT_PATH, map_location=device)
CLASSES = ckpt["classes"]
IMG_SIZE = ckpt.get("img_size", 224)

model = build_model(len(CLASSES), arch=ckpt["arch"], pretrained=False).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
transform = build_transforms(IMG_SIZE, train=False)


@torch.no_grad()
def classify(image: Image.Image):
    if image is None:
        return {}
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu()
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a sea-creature photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Sea Creature Classifier",
    description=(
        "A ResNet fine-tuned to recognise sea creatures. "
        "Upload a photo to see the top predicted classes."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
