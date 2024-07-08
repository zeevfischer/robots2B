# Robot Vision Assignment 2B - ArUco Marker Detection and Pose Estimation

## Authors
- Zeev Fischer: 318960242
- Eden Mor: 316160332
- Danielle Musai: 206684755

## Project Overview
This project focuses on real-time ArUco marker detection and pose estimation using a camera feed. It processes live video input to detect ArUco markers, calculate their 3D pose, and provide movement commands to align the camera with a reference image.

## Features
- Real-time ArUco marker detection
- 3D pose estimation (distance, yaw, pitch, roll)
- Movement command generation for camera alignment
- Live video processing and display
- Data logging to CSV files
- Frame-by-frame analysis and visualization

## Requirements
- Python 3.x
- OpenCV (cv2) with ArUco module
- NumPy
- Pandas

## Setup
1. Ensure all required libraries are installed:
pip install opencv-contrib-python numpy pandas
2. Set up the following directory structure:
project_root/
├── data/
│   └── output_data_video.csv
├── output/
│   ├── img/
│   ├── frames/
│   └── vid/
└── final.py

## Usage
1. Run the script:python final.py
2. The program will access the camera (default is camera index 1, change if necessary).
3. It will process the live feed, detecting ArUco markers and displaying the results in real-time.
4. Press 'q' to quit the program.

## Key Functions
- `calculate_3d_info`: Calculates distance, yaw, pitch, and roll from pose vectors.
- `process_live_frame`: Processes each frame from the live feed, detecting markers and calculating poses.
- `calculate_movement_commands`: Generates movement commands based on current and reference poses.
- `draw_directions`: Visualizes movement commands on the frame.
- `display_live_stream`: Main function for capturing and processing the live video feed.

## Output
- Live video display with detected ArUco markers and pose information
- CSV file (`output_data_live.csv`) with frame-by-frame detection data
- Saved frames with detected markers in the `output/frames/` directory

## Notes
- The code uses camera parameters for a 720p resolution and 82.6° Field of View.
- Adjust the camera index in `cv2.VideoCapture(1)` if necessary.
- The reference information is read from `data/output_data_video.csv`.

## Troubleshooting
- If facing issues with OpenCV, try reinstalling with:pip install opencv-contrib-python
- Ensure all directories exist before running the script.
- Check camera permissions if the video feed doesn't start.
