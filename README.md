# Human-pose-estimation
How to detect Human pose with full body
import cv2
import mediapipe as mp

# Initialize Mediapipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load dance video file
video_path = r"D:\Downloads\221026_02_Dancer_4k_042_preview.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize pose estimator
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read frame from the video
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get the pose landmarks
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # Show the image with pose landmarks
        cv2.imshow('Pose Estimation', image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
