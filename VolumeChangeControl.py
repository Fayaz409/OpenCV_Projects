import cv2 as cv
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import hand_detector as hd  
import numpy as np
import math

# Set webcam dimensions
hCam, wCam = 1080, 700

# Initialize webcam
cap = cv.VideoCapture(0)

# Initialize HandDetector
detector = hd.HandDetector()

# Get audio devices and volume control interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
volRan = volume.GetVolumeRange()
minVol = volRan[0]
maxVol = volRan[1]

# Initializations
pTime = 0
vol = 0
volBar = 400

while True:
    # Read frame from webcam
    success, img = cap.read()

    # Detect hands in the frame
    img = detector.findHands(img)

    # Get hand landmarks
    lmList = detector.position(img)

    if len(lmList) != 0:
        # Extract thumb and index finger landmarks
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw circles at thumb and index finger positions
        cv.circle(img, (x1, y1), 7, (255, 0, 0), cv.FILLED)
        cv.circle(img, (x2, y2), 7, (255, 0, 0), cv.FILLED)

        # Draw line connecting thumb and index finger
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw circle at the center of the line
        cv.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (255, 0, 0), cv.FILLED)

        # Calculate the length of the line between thumb and index finger
        length = int(math.hypot(x2 - x1, y2 - y1))

        # Map the length to the volume range
        vol = np.interp(length, [30, 300], [minVol, maxVol])

        # Map the length to the volume bar range
        volBar = np.interp(length, [30, 400], [400, 150])

        # Set the system volume
        volume.SetMasterVolumeLevel(int(vol), None)

        # If the length is less than 50, draw a green circle at the center
        if length < 50:
            cv.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (0, 255, 0), cv.FILLED)

    # Calculate and display frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (20, 75), cv.FONT_HERSHEY_COMPLEX, 1.1, (255, 0, 255), 2)

    # Draw the volume bar
    cv.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)

    # Color the volume bar based on the volume level
    if int(volBar) < 250:
        cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv.FILLED)
    else:
        cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)

    # Show the webcam feed
    cv.imshow('Web Cam', img)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break

        