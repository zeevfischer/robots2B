import cv2
import numpy as np
import pandas as pd
import os
import shutil



# Define camera parameters
camera_resolution = (1280, 720)  # 720p
fov = 82.6  # Field of View in degrees
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
    commands = []
    for i in range(len(current_info)):
        for j in range(len(reference_info)):
            #for every aruco we see we analyze if we have commands for it
            what_i_see = int(current_info[i][0])
            what_we_referance = int(reference_info[j][1])
            if what_i_see == what_we_referance:

                ref_distance = reference_info[j][3]
                ref_yaw = reference_info[j][4]
                ref_pitch = reference_info[j][5]

                # Extract values from current_info
                cur_distance = current_info[i][1]
                cur_yaw = current_info[i][2]
                cur_pitch = current_info[i][3]

                # Calculate differences
                distance_diff = ref_distance - cur_distance
                yaw_diff = ref_yaw - cur_yaw
                pitch_diff = ref_pitch - cur_pitch
                # if abs(distance_diff) > 100:
                #     distance_diff = 0.4
                # if abs(yaw_diff) > 100:
                #     yaw_diff = 2
                # if abs(pitch_diff) > 100:
                #     yaw_diff = 9

                # Print values
                print(f"Reference Info - aruco: {current_info[i][0]:.2f} Distance: {ref_distance:.2f}, Yaw: {ref_yaw:.2f}, Pitch: {ref_pitch:.2f}")
                print(f"Current Info   - aruco: {current_info[i][0]:.2f} Distance: {cur_distance:.2f}, Yaw: {cur_yaw:.2f}, Pitch: {cur_pitch:.2f}")
                print(f"Differences    - aruco: {current_info[i][0]:.2f} Distance: {distance_diff:.2f}, Yaw: {yaw_diff:.2f}, Pitch: {pitch_diff:.2f}")

                yaw_command = None
                distance_command = None
                pitch_command = None

                if abs(yaw_diff) > 10:
                    if yaw_diff > 0:
                        yaw_command = "right"
                    else:
                        yaw_command = "left"

                if abs(distance_diff) > 0.1:
                    if distance_diff < 0:
                        distance_command = "forward"
                    else:
                        distance_command = "backward"

                if abs(pitch_diff) > 25:
                    if pitch_diff < 0:
                        pitch_command = "down"
                    else:
                        pitch_command = "up"
                command = [current_info[i][0] ,yaw_command, distance_command, pitch_command]
                commands.append(command)

    return commands

# Function to draw directions on the frame
# def draw_directions(frame, yaw_command, distance_command, pitch_command,aruco,frame_id_video):
def draw_directions(frame, commands):
    height, width, _ = frame.shape

    for i in range(len(commands)):

        direction_text = ""
        if commands[i][0]:
            direction_text += f"aruco: {commands[i][0]}  "
        if commands[i][1]:
            direction_text += f"Yaw: {commands[i][1]}  "
        if commands[i][2]:
            direction_text += f"Distance: {commands[i][2]}  "
        if commands[i][3]:
            direction_text += f"Pitch: {commands[i][3]}  "

        cv2.putText(frame, direction_text,(50, height - 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if direction_text == "" or direction_text == f"aruco: {commands[i][0]}  ":
            return True
        else:
            return False

# Function to process live frame
def process_live_frame(frame, frame_id, csv_file, output_frames_dir, reference_info):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    current_info = []
    output_data = []

    next = None

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
            current_info.append([ids[i][0],distance, yaw, pitch, roll])

        # Save the frame image with detected QR codes
        frame_filename = os.path.join(output_frames_dir, f'frame_{frame_id}.jpg')
        cv2.imwrite(frame_filename, frame)


        #ids is what the frame detects !!!!

        comands = calculate_movement_commands(reference_info, current_info)
        next = draw_directions(frame, comands)

        # Calculate movement commands to align the camera with the reference image
        temp1 = [] # this is referanced
        for j in range(len(reference_info)):
            temp1.append(reference_info[j][1])
        # Add frame number and Aruco ID to the top left corner
        info_text = f"Frame ID: {reference_info[0][0]} Aruco ID: {temp1}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # Append data to CSV file
    if output_data:
        df = pd.DataFrame(output_data, columns=['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)', 'Roll (degrees)'])
        df.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)

    return next

# Function to display live stream with Aruco detection from camera
def display_live_stream(csv_file, output_frames_dir, output_csv_video):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream from USB camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

    frame_id = 0
    reference_info = []
    reference_info_id = None

    video_data = pd.read_csv(output_csv_video)
    end_of_csv = False
    add = True

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            if not end_of_csv:
                if reference_info == []:
                    for _, row in video_data.iterrows():
                        if reference_info_id is None:
                            reference_info_id = row['Frame ID']

                        if row['Frame ID'] == reference_info_id and add == True:
                            reference_info.append([row['Frame ID'], row['QR ID'], row['QR 2D (Corner Points)'], row['Distance'], row['Yaw (degrees)'],row['Pitch (degrees)'], row['Roll (degrees)']])
                        else:
                            if reference_info_id < row['Frame ID']:
                                reference_info_id = row['Frame ID']
                                add = False
                                break
                    else:
                        end_of_csv = True  # Set flag to true if end of CSV is reached
            if end_of_csv:
                print("end of file")
                break

            # Process the frame and save data
            next = process_live_frame(frame, frame_id, csv_file, output_frames_dir, reference_info)
            if next:
                reference_info = []
                add = True

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


def process_video(video_path, output_csv, output_video, output_frames_dir):
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate and size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)  # Calculate the delay for the correct frame rate

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # List to store the output data for CSV
    output_data = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)

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

                # Calculate 3D pose information using the improved method
                distance, yaw, pitch, roll = calculate_3d_info(rvec, tvec)

                # Append data to the output list
                output_data.append([frame_id, ids[i][0], corner_points_list, distance, yaw, pitch, roll])

            # Save the frame image with detected QR codes
            frame_filename = os.path.join(output_frames_dir, f'frame_{frame_id}.jpg')
            cv2.imwrite(frame_filename, frame)

        # Write the frame into the file
        out.write(frame)
        frame_id += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write the output data to CSV
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)',
               'Roll (degrees)']
    df = pd.DataFrame(output_data, columns=columns)
    df.to_csv(output_csv, index=False)

reference_image_path = 'data\\2B\\3.jpg'
output_image = 'output\\img\\output_image_with_markers.jpg'
output_csv = 'output\\output_data.csv'
output_csv_live = 'output\\output_data_live.csv'
output_directories = ['output', 'output\\img', 'output\\frames']
output_frames_dir = 'output\\img'
output_video = 'output\\vid\\output_video.avi'
output_csv_video = 'data\\output_data_video.csv'
video_path = 'data\\vid5.mp4'

# Set up directories
# setup_directories(output_directories)

# process_video(video_path, output_csv_video, output_video, output_frames_dir)

display_live_stream(output_csv_live, 'output\\frames',output_csv_video)