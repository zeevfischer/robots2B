import cv2
import numpy as np
import pandas as pd

# Define camera parameters (same as before)
camera_resolution = (1280, 720)  # 720p
fov = 75 #82.6  # Field of View in degrees
focal_length = (camera_resolution[0] / 2) / np.tan(np.deg2rad(fov / 2))
camera_matrix = np.array([[focal_length, 0, camera_resolution[0] / 2],
                          [0, focal_length, camera_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))

# Load the Aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Function to calculate distance, yaw, pitch, and roll (same as before)
def calculate_3d_info(rvec, tvec):
    distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return distance, np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# Function to process image and extract Aruco marker info (same as before)
# def process_img(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at {image_path} could not be loaded")
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#     if ids is not None:
#         rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
#         return calculate_3d_info(rvec[0], tvec[0]), ids[0][0]
#     else:
#         raise ValueError("No Aruco markers found in the image")
def process_img(image_path, output_csv, output_image):
    output = None
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

            # Calculate 3D pose information using the improved method
            output = calculate_3d_info(rvec, tvec)
            distance, yaw, pitch, roll = calculate_3d_info(rvec, tvec)

            # Append data to the output list
            output_data.append([ids[i][0], corner_points_list, distance, yaw, pitch, roll])

        # Save the image with detected QR codes
        cv2.imwrite(output_image, image)

    # Write the output data to CSV
    columns = ['QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)', 'Roll (degrees)']
    df = pd.DataFrame(output_data, columns=columns)
    df.to_csv(output_csv, index=False)

    return output, ids[0]

# Function to process the live video frame and get the Aruco marker info
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            # Draw the marker and axis
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        return calculate_3d_info(rvec[0], tvec[0]), ids[0][0]
    else:
        return None, None

# Function to calculate yaw, pitch, and distance movement commands
def calculate_movement_commands(reference_info, current_info):
    error_margin = 1
    yaw_diff = reference_info[1] - current_info[1]
    distance_diff = reference_info[0] - current_info[0]
    pitch_diff = reference_info[2] - current_info[2]

    yaw_command = None
    distance_command = None
    pitch_command = None

    if abs(yaw_diff) > 1:
        if yaw_diff > 0:
            yaw_command = "right"
        else:
            yaw_command = "left"

    if abs(distance_diff) > 0.02:
        if distance_diff > 0:
            distance_command = "forward"
        else:
            distance_command = "backward"

    if abs(pitch_diff) > 2:
        if pitch_diff > 0:
            pitch_command = "down"
        else:
            pitch_command = "up"

    return yaw_command, distance_command, pitch_command

# Function to draw arrows on the frame
def draw_arrows(frame, yaw_command, distance_command, pitch_command):
    height, width, _ = frame.shape
    arrow_size = 50

    if yaw_command == "left":
        cv2.arrowedLine(frame, (width // 2, height // 2), (width // 2 - arrow_size, height // 2), (0, 0, 255), 5)
    elif yaw_command == "right":
        cv2.arrowedLine(frame, (width // 2, height // 2), (width // 2 + arrow_size, height // 2), (0, 0, 255), 5)

    if distance_command == "forward":
        cv2.arrowedLine(frame, (width // 2, height - arrow_size), (width // 2, height - 2 * arrow_size), (0, 0, 255), 5)
    elif distance_command == "backward":
        cv2.arrowedLine(frame, (width // 2, arrow_size), (width // 2, 2 * arrow_size), (0, 0, 255), 5)

    if pitch_command == "up":
        cv2.arrowedLine(frame, (width // 2, height // 2), (width // 2, height // 2 - arrow_size), (0, 0, 255), 5)
    elif pitch_command == "down":
        cv2.arrowedLine(frame, (width // 2, height // 2), (width // 2, height // 2 + arrow_size), (0, 0, 255), 5)

def main():
    # Load reference image and get the Aruco marker info
    reference_image_path = 'data\\2B\\3.jpg'
    output_image = 'output\\img\\output_image_with_markers.jpg'
    output_csv = 'output\\output_data.csv'

    # process_img(image_path, output_csv, output_image)
    reference_info, reference_id = process_img(reference_image_path,output_csv,output_image)

    # Initialize the USB camera
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream from USB camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Process the frame and get the Aruco marker info
            current_info, current_id = process_frame(frame)

            if current_id is not None and current_id == reference_id:
                # Calculate movement commands to align the camera with the reference image
                yaw_command, distance_command, pitch_command = calculate_movement_commands(reference_info, current_info)
                draw_arrows(frame, yaw_command, distance_command, pitch_command)

            # Display the frame
            cv2.imshow("USB Camera Live Video", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the camera
        cap.release()

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
