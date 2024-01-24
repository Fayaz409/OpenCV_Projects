import cv2 as cv  # Import OpenCV for image processing
import time  # Import time for measuring frame rates
import mediapipe as mp  # Import MediaPipe for hand tracking

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters for hand detection
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert image to RGB format (MediaPipe expects RGB)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process the image to detect hands
        self.results = self.hands.process(imgRGB)
        # Draw hand landmarks on the image if draw is True
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def position(self, img, handNo=0, draw=True):
        # Get list of hand landmarks
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(myHand.landmark):
                # Get landmark coordinates (x, y)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])
                # Draw circles on the landmarks if draw is True
                if draw:
                    cv.circle(img, (cx, cy), 15, (200, 200, 0), cv.FILLED)
        return lmList
    
# Create a HandDetector object
detector = HandDetector()

# Start video capture from webcam
cap = cv.VideoCapture(0)

while True:
    # Read a frame from the video
    success, img = cap.read()

    # Detect hands and draw landmarks
    img = detector.findHands(img)

    # Get hand landmark positions
    lmList = detector.position(img, draw=False)

    # Print the coordinates of the 4th landmark (index fingertip) if hand is detected
    if len(lmList) != 0:
        print(lmList[4])

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (20, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 3)

    # Display the image with hand landmarks
    cv.imshow('Fayaz', img)

    # Exit if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break


















