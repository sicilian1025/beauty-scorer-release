# Notices and restrictions

## Code license

The source code in this repository is released under the MIT License. See `LICENSE`.

## Model weights and reference distribution

The pretrained weights in `src/beauty_scorer/assets/model.pt` and the reference
distribution in `src/beauty_scorer/assets/reference.json` were produced using the
**SCUT-FBP5500** dataset.

SCUT-FBP5500 is released by its authors for **non-commercial research use only**.
As a result:

- the pretrained weights and the reference distribution are provided for
  **non-commercial research and evaluation** only;
- they must not be used in commercial products or commercial services;
- for commercial use, retrain the model on a dataset whose license permits
  commercial use.

Source: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

## Face detector

`src/beauty_scorer/assets/face_detection_yunet_2023mar.onnx` is the YuNet face
detector from OpenCV Zoo. Refer to the OpenCV Zoo repository for its license
terms:

https://github.com/opencv/opencv_zoo

## Intended use and limitations

- Attractiveness is subjective. The output is a statistical estimate derived
  from a limited human-rated dataset, not an objective fact.
- Scores depend on lighting, pose, image quality, age, gender, ethnicity, and
  other factors. The model may reflect biases present in the training data.
- The reference percentile is relative to the SCUT-FBP5500 population used for
  calibration. It is not a ranking of the Chinese population or any other
  real-world population.
- Do not use this software to make decisions about individuals, including
  employment, education, housing, insurance, policing, or any high-impact
  decision.
- Face images are personal data. Obtain informed consent and comply with
  applicable privacy and data-protection laws.
