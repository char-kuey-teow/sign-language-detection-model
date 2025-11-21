# Sign Language Detection (TensorFlow Lite + MediaPipe)

## Overview
- **Project:** A lightweight sign language detection system that uses MediaPipe for hand/keypoint capture and a TensorFlow/TensorFlow Lite model for gesture classification.
- **Goal:** Collect hand landmarks, train a classifier, and run real-time inference with a `.tflite` model for deployment on edge and mobile devices.

## Repository Files
- `create_withmediapipe.py`: Capture hand landmarks and create labeled examples using MediaPipe.
- `train_withtensor.py`: Train a TensorFlow model from the captured dataset and export a TensorFlow Lite model.
- `sign_language_model.tflite`: Trained TFLite model (ready for edge/mobile inference).
- `label_mapping.json`: Mapping from class indices to human-readable labels.
- `check.py`: Run real-time inference (webcam) using the TFLite model.

## Requirements
- Python 3.8+ (Windows tested)
- Key Python packages:
  - `mediapipe`
  - `opencv-python`
  - `tensorflow` (for training) and/or `tensorflow-lite` for converter/runtime as needed
  - `numpy`

Install dependencies (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install mediapipe opencv-python tensorflow numpy
```

Note: If you plan to only run inference with the `.tflite` file, you can install `tensorflow` or use the platform-specific TFLite runtime.

## Usage

- Capture training/examples (uses webcam and MediaPipe):

```powershell
python .\create_withmediapipe.py
```

- Train the model (uses captured dataset):

```powershell
python .\train_withtensor.py
```

- Run real-time inference with the exported TFLite model:

```powershell
python .\check.py
```

If `check.py` accepts arguments for the model path or label file, pass them like:

```powershell
python .\check.py --model .\sign_language_model.tflite --labels .\label_mapping.json
```

## Deployment Options
- **Android / iOS (Mobile):** Use the `.tflite` model with the TensorFlow Lite Interpreter (Android: `org.tensorflow.lite.Interpreter`; iOS: `TFLiteSwift`). Add the `.tflite` file to your app assets and run inference on-device.
- **Raspberry Pi / Edge devices:** Install the TFLite runtime or full TensorFlow and run `check.py` or a small custom wrapper for your camera. For improved speed on supported hardware, compile/use Edge TPU (requires conversion and different toolchain).
- **Web (Browser):** Convert the TFLite model to TensorFlow.js format using `tensorflowjs_converter`, or re-export a model trained in TensorFlow.js for inference in-browser.

Conversion tips:
- Convert Keras/TF model to TFLite with quantization for smaller size and faster inference (if not already quantized).
- Use `tensorflowjs_converter` to serve model in web apps; for mobile, bundle the `.tflite` directly.

## Notes & Next Steps
- Inspect `label_mapping.json` to confirm class labels and ordering used during training.
- If you want, I can:
  - Add a `requirements.txt` or `environment.yml` for reproducible installs.
  - Create a short Android example showing how to load the `.tflite` model.
  - Convert the `.tflite` model to TFJS format and include a demo web page.

## License & Contact
- Add your preferred license file (e.g., `LICENSE`) if you plan to share this publicly.
- For questions or to extend the README with platform-specific examples, open an issue or ask here.
