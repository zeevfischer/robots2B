import cv2
import numpy as np
import pandas as pd
import os
import shutil

# Define camera parameters
camera_resolution = (1280, 720)  # 720p
fov = 75  # Field of View in degrees
focal_length = (camera_resolution[0] / 2) / np.tan(np.deg2rad(fov / 2))
camera_matrix = np.array([[focal_length, 0, camera_resolution[0] / 2],
                          [0, focal_length, camera_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))

# Load the Aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Function to calculate distance, yaw, pitch, and roll
def calculate_3d_info(rvec, tvec):
    distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return distance, np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# Function to process image and extract Aruco marker info
def process_img(image_path, output_csv, output_image):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # List to store the output data for CSV
    output_data = []

    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)

            # Draw the marker and axis
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Extract 2D corner points
            corner_points = corners[i][0]
            corner_points_list = corner_points.tolist()

            # Draw rectangle around the QR code
            pts = np.array(corner_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            # Add QR ID text
            cv2.putText(image, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Calculate 3D pose information
            distance, yaw, pitch, roll = calculate_3d_info(rvec, tvec)

            # Append data to the output list
            output_data.append([ids[i][0], corner_points_list, distance, yaw, pitch, roll])

        # Save the image with detected QR codes
        cv2.imwrite(output_image, image)

    # Write the output data to CSV
    columns = ['QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)', 'Roll (degrees)']
    df = pd.DataFrame(output_data, columns=columns)
    df.to_csv(output_csv, index=False)

    return output_data, ids[0]

# Function to set up directories
def setup_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            # Clear the directory
            shutil.rmtree(directory)
        os.makedirs(directory)

# Function to calculate yaw, pitch, and distance movement commands
def calculate_movement_commands(reference_info, current_info):
    ref_distance = reference_info[0][2]
    ref_yaw = reference_info[0][3]
    ref_pitch = reference_info[0][4]

    # Extract values from current_info
    cur_distance = current_info[0]
    cur_yaw = current_info[1]
    cur_pitch = current_info[2]

    # Calculate differences
    distance_diff = ref_distance - cur_distance
    yaw_diff = ref_yaw - cur_yaw
    pitch_diff = ref_pitch - cur_pitch

    # Print values
    print(f"Reference Info - Distance: {ref_distance:.2f}, Yaw: {ref_yaw:.2f}, Pitch: {ref_pitch:.2f}")
    print(f"Current Info   - Distance: {cur_distance:.2f}, Yaw: {cur_yaw:.2f}, Pitch: {cur_pitch:.2f}")
    print(f"Differences    - Distance: {distance_diff:.2f}, Yaw: {yaw_diff:.2f}, Pitch: {pitch_diff:.2f}")

    yaw_command = None
    distance_command = None
    pitch_command = None

    if abs(yaw_diff) > 3:
        if yaw_diff > 0:
            yaw_command = "right"
        else:
            yaw_command = "left"

    if abs(distance_diff) > 0.05:
        if distance_diff < 0:
            distance_command = "forward"
        else:
            distance_command = "backward"

    if abs(pitch_diff) > 3:
        if pitch_diff > 0:
            pitch_command = "down"
        else:
            pitch_command = "up"

    return yaw_command, distance_command, pitch_command

# Function to draw directions on the frame
def draw_directions(frame, yaw_command, distance_command, pitch_command):
    height, width, _ = frame.shape
    text_position = (50, height - 50)  # Starting position for the text

    direction_text = ""
    if yaw_command:
        direction_text += f"Yaw: {yaw_command}  "
    if distance_command:
        direction_text += f"Distance: {distance_command}  "
    if pitch_command:
        direction_text += f"Pitch: {pitch_command}  "

    cv2.putText(frame, direction_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

# Function to process live frame
def process_live_frame(frame, frame_id, csv_file, output_frames_dir, reference_info):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    output_data = []

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]
            # Draw the marker and axis
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Extract 2D corner points
            corner_points = corners[i][0]
            corner_points_list = corner_points.tolist()

            # Draw rectangle around the QR code
            pts = np.array(corner_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            # Add QR ID text
            cv2.putText(frame, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Calculate 3D pose information
            distance, yaw, pitch, roll = calculate_3d_info(rvec, tvec)

            # Append data to the output list
            output_data.append([frame_id, ids[i][0], corner_points_list, distance, yaw, pitch, roll])

            # Calculate movement commands to align the camera with the reference image
            current_info = (distance, yaw, pitch, roll)
            yaw_command, distance_command, pitch_command = calculate_movement_commands(reference_info, current_info)
            draw_directions(frame, yaw_command, distance_command, pitch_command)

        # Save the frame image with detected QR codes
        frame_filename = os.path.join(output_frames_dir, f'frame_{frame_id}.jpg')
        cv2.imwrite(frame_filename, frame)

    # Append data to CSV file
    if output_data:
        df = pd.DataFrame(output_data, columns=['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)', 'Roll (degrees)'])
        df.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)

# Function to display live stream with Aruco detection from camera
def display_live_stream(csv_file, output_frames_dir, reference_info):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream from USB camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

    frame_id = 0

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Process the frame and save data
            process_live_frame(frame, frame_id, csv_file, output_frames_dir, reference_info)

            # Display the frame
            cv2.imshow("USB Camera Live Video with Aruco Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1
    finally:
        # Release the camera
        cap.release()

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()

reference_image_path = 'data\\2B\\3.jpg'
output_image = 'output\\img\\output_image_with_markers.jpg'
output_csv = 'output\\output_data.csv'
output_csv_live = 'output\\output_data_live.csv'
output_directories = ['output', 'output\\img', 'output\\frames']

# Set up directories
setup_directories(output_directories)

# Process the reference image
reference_info, reference_id = process_img(reference_image_path, output_csv, output_image)

# Display live stream with Aruco detection and process frames
display_live_stream(output_csv_live, 'output\\frames', reference_info)
