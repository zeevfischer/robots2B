# Robot Vision Assignment 2B - ArUco Marker Detection and Pose Estimation

## Authors
- Zeev Fischer: 318960242
- Eden Mor: 316160332
- Danielle Musai: 206684755

## Project Overview
There are 3 files in this reposetory:
1. compare_image_to_stream: This is what we were asked to do for the presentation. This is what needs to be tested.   
2. compare_video_to_stream: This takes a video instead of an image and gives you directions to follow the video. This is what we thought we needed to do.   
3. The last file is a past version of compare_video_to_stream that I saved.
   
This project focuses on real-time ArUco marker detection and pose estimation using a camera feed. It processes live video/image input to detect ArUco markers, calculate their 3D pose, and provide movement commands to align the camera with a reference image.

## Features
- Real-time ArUco marker detection
- 3D pose estimation (distance, yaw, pitch, roll)
- Movement command generation for camera alignment
- Live video processing and display
- Data logging to CSV files
- Frame-by-frame analysis and visualization

## Requirements
- Process each frame in real-time (under 30 ms).
- Use Tello's camera parameters: 720p resolution, 82.6 FoV
- Python 3.x
- OpenCV (cv2) with ArUco module
- NumPy
- Pandas

## How To Run
**NOTE: For this code, there are some path folders that you need to check and update to match your computer path.!!!**   
1. Basic understanding of running code is required.
2. Ensure all necessary libraries are installed. If you encounter issues with cv2, use pip install opencv-contrib-python.
3. Use PyCharm or any other Python IDE to open the workspace.
4. Download the repository and place final.py and the data and output directories into your Python workspace.
5. Ensure all files are in the same workspace, or update the paths in the code to match your directory structure

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
- Adjust the camera index in `cv2.VideoCapture(1)` if necessary because we connected an external camera for better handeling.
- The reference information is read from `data/output_data_video.csv`.


