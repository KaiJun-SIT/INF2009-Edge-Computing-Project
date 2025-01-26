# INF2009-Edge-Computing-Project

## Design & Justification (20%)

Design & Justification (Week 8: Friday, 28th Feb 2025 @ 23:59 hrs.)

• Introduction, Problem Statement and Objectives

• Design (System Architecture only)


## Project Demonstration (30%)

Video (10 mins) & Poster (Week 13: Tuesday, 1st April 2025 @ 09:00 hrs.)

• Dropbox at xSITe

• Final Presentation (Week 13: Thursday, 3rd April @ 11:00 hrs. i.e. during lab session)

• 20 mins per Team (demo)

• QnA


# Stereo Camera Depth Estimation

This project implements real-time depth estimation using a stereo camera setup with OpenCV. The system tracks objects using HSV color filtering and calculates their distance using triangulation.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- imutils
- Two USB webcams with identical specifications
- Secure parallel mount of said webcams

## Setup

1. Connect both USB cameras to your computer
2. Verify camera IDs (typically 0 and 2 for USB cameras)
3. Ensure cameras are mounted parallel at a fixed distance

## Camera Calibration

### Step 1: Capture Calibration Images
1. Load `pattern.jpg` (9x6 chessboard pattern) on an iPad or display
2. Run calibration image capture:
```bash
python capture_calib_img.py
```
3. Follow the prompts:
   - Press Enter to start
   - Press 'y' if camera arrangement is correct, 'n' to swap IDs
4. Capture calibration images:
   - Move the pattern around covering different angles and positions
   - Ensure the entire pattern is visible in both cameras
   - Capture 28 distinct poses
5. Press 'ESC' when finished

### Step 2: Generate Calibration Parameters
1. Run the calibration generator:
```bash
python generate_calib_params.py
```
2. This creates `calib_generate_params_py.xml` in the data directory

## Running the System

1. Start the main program:
```bash
python main.py
```

2. The system will display:
   - Left and right camera feeds
   - HSV masks for object tracking
   - Real-time depth measurements in centimeters

3. Press 'q' to exit

## System Parameters

- `B`: Baseline (distance between cameras) in cm
- `f`: Focal length in mm
- `alpha`: Field of view in degrees
- HSV filter values can be adjusted in `HSV_filter.py`

## Project Structure

- `main.py`: Core program for stereo vision and depth estimation
- `HSV_filter.py`: Color-based object detection
- `shape_recognition.py`: Circle detection and tracking
- `triangulation.py`: Depth calculation algorithms
- `calibration.py`: Camera calibration utilities

## Troubleshooting

- If tracking is lost, verify HSV filter settings
- Ensure consistent lighting conditions
- Check camera IDs if initialization fails
- Recalibrate if depth measurements are inaccurate

## Notes

- System accuracy depends on proper camera calibration
- Performance may vary with lighting conditions
- Depth calculations assume cameras are perfectly parallel