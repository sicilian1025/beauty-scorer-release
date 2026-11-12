# Model card

## Model

- Architecture: ResNet18 + global average pooling + two-layer regression head.
- Input: 224×224 RGB face crop.
- Output: a calibrated score in `[1, 5]`.
- Parameters: about 11.3M.
- Checkpoint: `model.pt` (about 45 MB).
- Calibration: a linear mapping from raw output to `[1, 5]`, stored inside the
  checkpoint as `a` and `b`.

## Training data

- Dataset: SCUT-FBP5500.
- 5,500 frontal face images.
- Each image was rated on a 1–5 scale by 60 volunteers; the mean rating is the
  target.
- Official split: 3,300 training images and 2,200 test images.
- The released model was trained on 2,970 images and validated on 330 images
  taken from the official training split.

## Reference distribution

`reference.json` contains sorted human rating values from the training split,
separately for `all`, `asian`, and `caucasian`. The default configuration uses
the `asian` subset for percentile conversion.

## Intended use

Research and evaluation of facial beauty prediction. The output is a
statistical estimate and must not be used for decisions about individuals.

## Limitations

- Attractiveness is subjective and culturally dependent.
- The model may reflect biases in the training data, including age, gender,
  ethnicity, lighting, and image quality.
- Percentiles are relative to the SCUT-FBP5500 population, not to any
  real-world population.
- Predictions shrink toward the mean, so extreme scores are under-estimated.

## License

The weights were produced from SCUT-FBP5500, which is licensed for
non-commercial research use only. See the repository `NOTICE.md`.
