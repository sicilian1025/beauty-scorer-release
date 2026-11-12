# beauty-scorer

Facial beauty scoring with calibrated scores and percentile output.

This package detects faces in an image, predicts a 1–5 beauty score for each
face, and converts the score into a percentile of a reference population. It
ships with a pretrained model, a face detector, and the calibration data needed
to run out of the box.

> **Restrictions:** the pretrained weights were trained on SCUT-FBP5500, which is
> licensed for **non-commercial research only**. See [`NOTICE.md`](NOTICE.md)
> before using this project commercially.

## Features

- Face detection with duplicate suppression for large faces.
- Calibrated 1–5 beauty score.
- Percentile output relative to a reference distribution.
- Multi-face support, or single-face mode.
- Image, folder, and real-time video input.
- Command-line interface and Python API.
- CPU and CUDA support.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenCV, NumPy, Pillow

For headless servers, replace `opencv-python` with `opencv-python-headless`.

## Installation

```bash
git clone <your-repository-url>
cd beauty-scorer
python -m pip install -e .
```

Or install the dependencies only:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Score one image:

```bash
beauty-score score --input photo.jpg
```

Score a folder and save annotated images:

```bash
beauty-score score --input photos/ --output-dir results/
```

Return JSON:

```bash
beauty-score score --input photo.jpg --json
```

Keep only the most confident face:

```bash
beauty-score score --input photo.jpg --single-face
```

Real-time camera:

```bash
beauty-score video --source 0
```

Video file:

```bash
beauty-score video --source sample.mp4 --save-dir captures/
```

Show model and device information:

```bash
beauty-score info
```

## Python API

```python
from beauty_scorer import BeautyScorer

scorer = BeautyScorer()
results = scorer.score_file("photo.jpg", all_faces=True)

for item in results:
    print(item.score, item.percentile, item.box)
```

Score an in-memory BGR image:

```python
import cv2
from beauty_scorer import BeautyScorer

image = cv2.imread("photo.jpg")
scorer = BeautyScorer()
results = scorer.score_image(image)
annotated = scorer.annotate(image, results)
cv2.imwrite("photo_scored.jpg", annotated)
```

## Configuration

The defaults work without a config file. To override settings, pass a JSON file:

```bash
beauty-score score --input photo.jpg --config configs/default.json
```

```json
{
  "detector": {
    "score_threshold": 0.55,
    "max_side": 1024
  },
  "model": {
    "input_size": 224,
    "device": "auto"
  },
  "scoring": {
    "population": "asian",
    "margin": 0.25
  }
}
```

Supported populations: `asian`, `all`, `caucasian`.

## Output

Each face produces:

| Field | Meaning |
|---|---|
| `score` | Calibrated beauty score in `[1, 5]` |
| `percentile` | Percentage of the reference population at or below the score |
| `raw_score` | Uncalibrated model output |
| `box` | Face box in original image coordinates |
| `inference_ms` | Model forward time |

## Project structure

```
beauty-scorer/
  configs/default.json
  src/beauty_scorer/
    __init__.py
    cli.py
    config.py
    detector.py
    errors.py
    image_utils.py
    model.py
    percentile.py
    scoring.py
    video.py
    assets/
      model.pt
      face_detection_yunet_2023mar.onnx
      reference.json
  tests/
  .github/workflows/ci.yml
```

## Testing

```bash
python -m pip install -e ".[dev]"
pytest
```

The unit tests cover duplicate suppression and percentile conversion. They do
not require the model weights.

## Model

- Backbone: ResNet18 with global average pooling.
- Input: 224×224 RGB face crop.
- Output: one calibrated score.
- Training data: SCUT-FBP5500 (non-commercial research use only).
- Parameters: about 11.3M.
- Checkpoint size: about 45 MB.

GitHub allows files under 100 MB, so the checkpoint can be committed directly.
If you prefer, track `*.pt` with Git LFS.

## License

Code: MIT. Weights and calibration data: non-commercial research use only.
See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).

## Disclaimer

Beauty is subjective. This project produces a statistical estimate, not an
objective judgment. Do not use it for decisions about individuals, and comply
with privacy laws when processing face images.
