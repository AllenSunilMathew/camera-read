# camera-read (Real-time Camera OCR + Text-to-Speech)

A small Python app that opens your webcam, lets you press **SPACE** to scan text from the current frame using **EasyOCR**, overlays detected bounding boxes on the video, and reads the detected text aloud via **pyttsx3**.

## Features
- Real-time webcam feed (OpenCV)
- OCR on-demand (press **SPACE**) using **EasyOCR**
- Displays detected text + bounding boxes for a few seconds
- Text-to-speech of detected text
- Uses CUDA automatically if available (PyTorch/EasyOCR)

## Requirements
- Python 3.9+ (works with other versions too)
- A webcam

## Install dependencies
Run:

```bash
pip install opencv-python easyocr pyttsx3 numpy torch
```

### Notes on GPU
The script auto-detects CUDA via PyTorch:
- If CUDA is available, EasyOCR will use GPU.
- If not, it will fall back to CPU.

Make sure your PyTorch install matches your CUDA setup if you want GPU acceleration.

## Run
```bash
python cam.py
```

## Controls
- **SPACE**: Capture current frame and run OCR
- **Q** / **ESC**: Exit the app

## How it works (high level)
- Open webcam with OpenCV
- When SPACE is pressed, start an OCR thread so the video UI stays responsive
- Convert frame to grayscale and call `easyocr.Reader.readtext`
- Filter low-confidence results (`prob > 0.35`)
- Draw bounding boxes and labels
- Speak detected text using `pyttsx3`

## Troubleshooting
- **No webcam found**: try changing webcam index in `cv2.VideoCapture(0)` to `1` (the code already attempts 1 if 0 fails).
- **TTS issues on Windows**: ensure audio devices are available and try again.
- **OCR accuracy**: improve lighting, move closer, and press SPACE when the text is steady.

## File overview
- `cam.py` – main application
- `README.md` – this documentation

