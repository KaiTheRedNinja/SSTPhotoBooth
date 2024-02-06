from imutils import face_utils
import dlib
import cv2
from time import time

from pondfilter import pondFilter
from sunglassesfilter import sunglassesFilter
from graywallfilter import graywallFilter
from medalsfilter import medalsFilter

# Story:
# [x] pondstar
# [x] gray wall
# [x] sunglasses
# [x] north korean general (6 leadership positions)

# load the model
print("Loading the model...")
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
print("Finished loading the model.")

# set up the camera
print(" [NOTICE] Setting up camera")
captureIndex = 0
while True: # account for certain channels being unavailable (ie when using continuity camera)
    try:
        cap = cv2.VideoCapture(captureIndex)
        break
    except:
        print("Could not set up on channel " + str(captureIndex))
        captureIndex += 1
print(" [NOTICE] Finished setting up camera on channel " + str(captureIndex))

print("""
=========<INSTRUCTIONS>=========
Press , (comma) to go to the previous filter
Press . (full stop) to go to the next filter
Press esc to exit
========</INSTRUCTIONS>=========
""")

state = 0
image = None
shapes = [None]

def changeState(newvalue):
    global state
    state = newvalue
    if state == 0:
        print("[ X . . . ] Kai's Sunglasses")
    elif state == 1:
        print("[ . X . . ] Congrats, you're a Pondstar")
    elif state == 2:
        print("[ . . X . ] The ICONIC Gray Wall")
    elif state == 3:
        print("[ . . . X ] When your batch can theoretically get 6 badges")

changeState(0)

while True:
    # Getting out image by webcam
    _, image = cap.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    shapes = []
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shapes.append(shape)

    # # state machine, at some point
    result = image
    if state == 0: # sunglasses
        result = sunglassesFilter(image, shapes)
    elif state == 1: # pond
        result = pondFilter(image, shapes)
    elif state == 2: # gray wall
        result = graywallFilter(image, shapes)
    elif state == 3: # north korean general
        result = medalsFilter(image, shapes)

    # show image
    cv2.imshow("SST 2025 Photo Booth", result)

    # process shortcuts
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc
        break
    elif k == 44: # <
        changeState(max(0, state-1))
    elif k == 46: # >
        changeState(min(3, state+1))

cv2.destroyAllWindows()
cap.release()
