# Import OpenCV for image processing
import cv2 as cv

# Import MediaPipe for face detection
import mediapipe as mp

# Import time for measuring frame rates
import time


class FaceDetection():
    def __init__(self):
        # Access MediaPipe face detection solution
        self.mpFaceDetection = mp.solutions.face_detection

        # Create a face detection object
        self.faceDetection = self.mpFaceDetection.FaceDetection()

        # Access MediaPipe drawing utilities
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        # Convert image to RGB format (MediaPipe expects RGB)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Process the image to detect faces
        results = self.faceDetection.process(imgRGB)

        # Draw bounding boxes and scores on detected faces
        if results.detections:
            for id, detection in enumerate(results.detections):
                # Get the bounding box coordinates
                b_Box = detection.location_data.relative_bounding_box

                # Get the detection confidence score
                score = detection.score

                # Calculate absolute bounding box coordinates based on image dimensions
                h, w, c = img.shape
                bBox = int(b_Box.xmin * w), int(b_Box.ymin * h), int(b_Box.width * w), int(b_Box.height * h)

                # Draw the bounding box and score if draw is True
                if draw:
                    cv.rectangle(img, bBox, (255, 0, 255), 3)  # Draw a purple rectangle
                    cv.putText(img, f'Score: {int(score[0] * 100)}%', (bBox[0], bBox[1] - 20),  # Display score above box
                                cv.FONT_HERSHEY_DUPLEX, .8, (200, 100, 0), 2)  # Orange text

        return img  # Return the image with or without bounding boxes

# Open video capture from webcam
cap = cv.VideoCapture(0)

# Create a FaceDetection object
detector = FaceDetection()

# Initialize previous time for FPS calculation
pTime = 0

while True:
    # Read a frame from the video
    success, img = cap.read()

    # Detect faces and draw bounding boxes
    img = detector.findFace(img)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'fps {int(fps)}', (20, 50), cv.FONT_HERSHEY_PLAIN, 1.8, (100, 100, 255), 2)  # Blue text

    # Display the image with face detections
    cv.imshow('Fayaz', img)

    # Exit if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break














