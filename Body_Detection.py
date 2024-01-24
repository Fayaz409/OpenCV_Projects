# Import OpenCV for image processing
import cv2 as cv
# Import time for measuring frame rates
import time
# Import MediaPipe for pose estimation
import mediapipe as mp

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters for pose estimation
        self.mode = mode  # Set mode for static or dynamic poses
        self.upBody = upBody  # Set to True for upper body only estimation
        self.smooth = smooth  # Enable smoothing for smoother pose tracking
        self.detectionCon = detectionCon  # Set minimum detection confidence
        self.trackCon = trackCon  # Set minimum tracking confidence

        # Access MediaPipe drawing utilities
        self.mpDraw = mp.solutions.drawing_utils

        # Access MediaPipe pose solution
        self.mpPose = mp.solutions.pose

        # Create a pose object with specified parameters
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findBody(self, img, draw=True):
        # Convert image to RGB format (MediaPipe expects RGB)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Process the image to detect pose
        self.results = self.pose.process(imgRGB)

        # Draw pose landmarks on the image if draw is True
        if self.results.pose_landmarks:
            if draw:
                # Use MediaPipe drawing utilities to visualize pose landmarks
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img  # Return the image with or without landmarks

    def Position(self, img, draw=True):
        # Get list of pose landmarks
        lmList = []

        if self.results.pose_landmarks:
            myBody = self.results.pose_landmarks

            # Iterate through each pose landmark
            for idx, lm in enumerate(myBody.landmark):
                # Get height, width, and channels of the image
                h, w, c = img.shape

                # Calculate coordinates of the landmark (x, y) in pixels
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Add landmark index and coordinates to the list
                lmList.append([idx, cx, cy])

                # Draw circles on the landmarks if draw is True
                if draw:
                    cv.circle(img, (cx, cy), 4, (0, 0, 255), cv.FILLED)

        return lmList  # Return the list of landmark positions


# Create a PoseDetector object
detector = PoseDetector()

# Open a video capture from a file
cap = cv.VideoCapture('Dance.mp4')

while True:
    # Read a frame from the video
    success, img = cap.read()

    # Detect pose and draw landmarks
    img = detector.findBody(img)

    # Get pose landmark positions
    lmList = detector.Position(img)

    # Print all landmark positions
    print(lmList)

    # Print the coordinates of the 6th landmark (right wrist) if pose is detected
    if len(lmList) != 0:
        print(lmList[5])

    # Display the image with pose landmarks
    cv.imshow('Body Detector', img)

    # Exit if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break

    



