# Vehicle Speed Detection using YOLOv8

This project implements vehicle detection and speed estimation using YOLOv8 and computer vision techniques.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the detection on a video file:
```bash
python main.py --model path/to/your/model.pt --source path/to/video.mp4 --output output.mp4
```

Arguments:
- `--model`: Path to your fine-tuned YOLOv8 model weights (required)
- `--source`: Path to input video file (required)
- `--conf-thres`: Confidence threshold for detections (default: 0.5)
- `--output`: Path to save the output video (default: output.mp4)

## Project Structure

- `main.py`: Entry point script containing the vehicle detection and speed estimation logic
- `requirements.txt`: List of Python dependencies
- Additional modules will be added for:
  - Vehicle tracking
  - Speed estimation
  - Camera calibration
  - Data visualization

## TODO

- [ ] Implement vehicle tracking across frames
- [ ] Add camera calibration module
- [ ] Implement speed estimation algorithm
- [ ] Add data logging and visualization
- [ ] Implement real-time processing capabilities 