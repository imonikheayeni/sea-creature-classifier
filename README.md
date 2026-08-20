# Sea Creature Image Classifier

A deep-learning computer-vision project that recognises sea creatures from photos.
It fine-tunes an ImageNet-pretrained ResNet to classify underwater images into
species groups such as dolphin, jellyfish, shark, sea ray, and whale, and ships
with an interactive demo you can drop into a portfolio site.

> Built as a personal project to practise transfer learning, image
> augmentation, handling class imbalance, and model evaluation in PyTorch.

## Results

Fill these in after your first training run (see [Reproducing the results](#reproducing-the-results)):

| Metric | Value |
| --- | --- |
| Test accuracy | _e.g. 0.92_ |
| Macro F1 | _e.g. 0.90_ |
| Classes | _e.g. 9_ |
| Backbone | ResNet18 (fine-tuned) |

![Training curves](outputs/training_curves.png)
![Confusion matrix](outputs/confusion_matrix.png)

## Dataset

Sea Animals Image Dataset by vencerlanz09 on Kaggle:
https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste

It holds ~13k images across 23 sea-creature classes in ImageFolder layout (one
subfolder per class). This project can train on all 23 or a subset. Please
review and respect the dataset's license on the Kaggle page, and keep the
attribution above.

The raw images are not committed to this repo. Download them with the helper
below.

## Project structure

```
sea-creatures-classifier/
├── app.py                  # Gradio demo (drag in an image, get a prediction)
├── requirements.txt
├── README.md
└── src/
    ├── download_data.py    # pull the dataset from Kaggle
    ├── data.py             # transforms + stratified train/val/test split
    ├── model.py            # ResNet backbone with a new classification head
    ├── train.py            # training loop, class-weighted loss, checkpointing
    ├── evaluate.py         # test-set report + confusion matrix
    └── predict.py          # single-image inference
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Create a free Kaggle API token (Kaggle > Settings > Create New Token) so
`kagglehub` can authenticate, then:

```bash
# All 23 classes
python -m src.download_data --dest data/sea_animals

# Or the smaller 9-class version
python -m src.download_data --dest data/sea_animals --course-subset
```

You can also download the zip manually from the Kaggle page and unzip it so that
`data/sea_animals/<ClassName>/*.jpg` exists.

## Reproducing the results

```bash
# Train (fine-tunes the whole backbone by default)
python -m src.train --data-dir data/sea_animals --arch resnet18 --epochs 15

# Evaluate the best checkpoint on the held-out test split
python -m src.evaluate --data-dir data/sea_animals --ckpt outputs/best_model.pt

# Predict a single image
python -m src.predict --ckpt outputs/best_model.pt --image some_photo.jpg
```

Useful flags: `--freeze-backbone` (train only the head, faster), `--arch
resnet34`/`resnet50`, `--batch-size`, `--lr`, `--epochs`.

No GPU? Add `--freeze-backbone` and use `resnet18`; it trains on CPU in a
reasonable time, just slower.

## Interactive demo

```bash
python app.py
```

Opens a local Gradio page where you upload an image and see the top predicted
classes. To share it, host it free on Hugging Face Spaces or set
`demo.launch(share=True)` in `app.py` for a temporary public link, then embed or
link it from your portfolio.

## What this project demonstrates

- Transfer learning: fine-tuning a pretrained CNN instead of training from scratch
- Data augmentation and correct per-split transforms (no leakage into val/test)
- Stratified splitting and class-weighted loss to handle an imbalanced dataset
- Proper evaluation: held-out test set, per-class precision/recall/F1, confusion matrix
- Packaging a model behind a simple interactive demo

## Possible extensions

- Grad-CAM heatmaps to visualise what the model looks at
- Test-time augmentation for a small accuracy bump
- Export to ONNX / TorchScript for faster deployment
- A "confidence threshold" that returns "unsure" on out-of-distribution photos

## License and attribution

Code in this repository is my own. The dataset belongs to its original authors
on Kaggle (link above); consult that page for its terms before redistributing
any images.
