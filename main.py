import cv2
import mediapipe as mp
import time
import random

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define stick figure keypoints
stick_figure = {
    "left_arm": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "right_arm": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
    "left_leg": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
    "right_leg": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "shoulders": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
    "waist": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
    "left": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
    "right": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP],
    # 'lip': [mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT],
}

# Define black background
background_color = (0, 0, 0)  # Black

# Define white color for stick figure
drawing_color = (255, 255, 0)  # White

# Visibility threshold for landmarks
visibility_threshold = 0.7

# Blinking state flag
is_eye_blinking = False

# Blinking duration and interval
blink_duration = random.uniform(0.09, 0.2)
blink_interval = random.uniform(3, 10)

# Blinking timer
blink_timer = time.time() + blink_interval

def draw_stick_figure(frame_bgr, joint_points):
    if len(joint_points) >= 2:
        # Draw lines connecting the keypoints in white color
        for i in range(len(joint_points) - 1):
            cv2.line(frame_bgr, joint_points[i], joint_points[i+1], drawing_color, 2)

def draw_head(frame_bgr, results, joint_points):
    if results.pose_landmarks and results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > visibility_threshold:
        nose_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_bgr.shape[1])
        nose_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_bgr.shape[0])

        if joint_points and len(joint_points) > 0:
            radius = int((joint_points[0][1] - nose_y) * 0.7)
            if radius >= 0:
                cv2.circle(frame_bgr, (nose_x, nose_y), radius, drawing_color, 2)

        # Draw eyes or hide them during blinking
        if not is_eye_blinking:
            left_eye_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

            if left_eye_landmark.visibility > visibility_threshold:
                left_eye_x = int(left_eye_landmark.x * frame_bgr.shape[1])
                left_eye_y = int(left_eye_landmark.y * frame_bgr.shape[0])
                eye_radius = int((joint_points[0][1] - nose_y) * 0.1)
                cv2.line(frame_bgr, (left_eye_x, left_eye_y - eye_radius), (left_eye_x, left_eye_y + eye_radius), drawing_color, 2)

            if right_eye_landmark.visibility > visibility_threshold:
                right_eye_x = int(right_eye_landmark.x * frame_bgr.shape[1])
                right_eye_y = int(right_eye_landmark.y * frame_bgr.shape[0])
                eye_radius = int((joint_points[0][1] - nose_y) * 0.1)
                cv2.line(frame_bgr, (right_eye_x, right_eye_y - eye_radius), (right_eye_x, right_eye_y + eye_radius), drawing_color, 2)

         # Draw circular smile
        smile_radius = int((joint_points[0][1] - nose_y) * 0.3)
        smile_center_x = nose_x
        smile_center_y = nose_y + int((joint_points[0][1] - nose_y) * 0.1)

        start_angle = 30
        end_angle = 155

        # Draw circular arc for smile
        cv2.ellipse(frame_bgr, (smile_center_x, smile_center_y), (smile_radius, smile_radius), 0, start_angle, end_angle, drawing_color, 2)


def update_blink_state():
    global is_eye_blinking, blink_timer

    if time.time() >= blink_timer:
        is_eye_blinking = not is_eye_blinking
        if is_eye_blinking:
            blink_timer = time.time() + blink_duration
        else:
            blink_timer = time.time() + blink_interval

# Set up webcam video capture
cap = cv2.VideoCapture(0)


# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


process_frame = True 
# Main program loop
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        # Read webcam frame
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        if process_frame:
            # Convert the image to RGB for processing by MediaPipe Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Clear the frame with the black background
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr[:] = background_color

            # Initialize joint points as an empty list
            joint_points = []

            # Draw the stick figure on the frame
            if results.pose_landmarks:
                for part in stick_figure.values():
                    joint_points = []
                    for landmark in part:
                        if results.pose_landmarks.landmark[landmark].visibility > visibility_threshold:
                            joint_points.append(mp_drawing._normalized_to_pixel_coordinates(
                                results.pose_landmarks.landmark[landmark].x,
                                results.pose_landmarks.landmark[landmark].y,
                                frame_bgr.shape[1], frame_bgr.shape[0]))

                    draw_stick_figure(frame_bgr, joint_points)

            draw_head(frame_bgr, results, joint_points)

            # Update blinking state
            update_blink_state()
            
            # Set the desired window width and height
            window_width = 800
            window_height = 600
            

            # Calculate the aspect ratio of the frame
            frame_aspect_ratio = frame_width / frame_height

            # Calculate the corresponding window width and height to maintain the aspect ratio
            if window_width / window_height > frame_aspect_ratio:
                window_width = int(window_height * frame_aspect_ratio)
            else:
                window_height = int(window_width / frame_aspect_ratio)

            # Create the named window with the calculated width and height
            cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Webcam', window_width, window_height)

            # Inside the while loop:
            cv2.imshow('Webcam', frame_bgr)
                
            output.write(frame_bgr)
        process_frame = not process_frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all windows
cap.release()
output.release()
cv2.destroyAllWindows()
