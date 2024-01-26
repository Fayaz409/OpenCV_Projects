# Importing necessary libraries
import cv2 as cv
import mediapipe as mp
import time

# Creating a FaceMesh class to encapsulate the functionality
class FaceMesh():
    # Constructor with default values for parameters
    def __init__(self, max_num_faces=1, thickness=1, circle_radius=1):
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.max_num_faces = max_num_faces
        # Initializing Mediapipe FaceMesh and Drawing utilities
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.max_num_faces)
        self.mpDrawing = mp.solutions.drawing_utils
        # Drawing specifications for landmarks
        self.drawSpecs = self.mpDrawing.DrawingSpec(thickness=self.thickness, circle_radius=self.circle_radius)
    
    # Method to draw face mesh on the input image
    def DrawMesh(self, img, draw=True):
        # Convert the image to RGB format as FaceMesh processes RGB images
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process the image with FaceMesh
        self.results = self.FaceMesh.process(imgRGB)
        
        # Check if there are multiple face landmarks detected
        if self.results.multi_face_landmarks:
            # Loop through each set of face landmarks
            for f_mesh_ps in self.results.multi_face_landmarks:
                # Draw the landmarks on the image if draw is True
                if draw:
                    self.mpDrawing.draw_landmarks(img, f_mesh_ps, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpecs, self.drawSpecs)
        return img
    
    # Method to extract landmark positions
    def landMarks(self, img, draw=True):
        lmList = []
        # Check if there are multiple face landmarks detected
        if self.results.multi_face_landmarks:
            # Loop through each set of face landmarks
            for f_mesh_ps in self.results.multi_face_landmarks:
                # Loop through each landmark in the set
                for id, lm in enumerate(f_mesh_ps.landmark):
                    h, w, c = img.shape
                    # Extract x, y coordinates of the landmark and add to lmList
                    x, y = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, x, y])
                    # Draw a circle at the landmark position if draw is True
                    if draw:
                        cv.circle(img, (x, y), 1, (255, 0, 255), 1)
        return lmList

# Open a video capture object
cap = cv.VideoCapture(0)
# Create an instance of FaceMesh class
detector = FaceMesh(max_num_faces=2)
# Variable to store previous frame time
pTime = 0

# Main loop to capture and process video frames
while True:
    # Read a frame from the video capture
    success, img = cap.read()
    # Process the frame to draw the face mesh
    img = detector.DrawMesh(img)
    # Extract landmarks and optionally print them
    lmList = detector.landMarks(img)
    # Display the image with the drawn face mesh
    cv.imshow('Face Mesh', img)
    
    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break

    

