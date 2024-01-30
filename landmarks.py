from imutils import face_utils
import dlib
import cv2
import threading

from pondfilter import pondFilter
from sunglassesfilter import sunglassesFilter

# Story:
# [ ] grayscale pinapple man
# [x] pondstar
# [ ] gray wall
# [x] sunglasses
# [ ] no o levels except chinese (maybe the angel-demon thing)
# [ ] north korean general (6 leadership positions)

# load the model
print("Loading the model...")
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
print("Finished loading the model.")

# set up the camera
print("Setting up camera")
captureIndex = 0
while True: # account for certain channels being unavailable (ie when using continuity camera)
    try:
        cap = cv2.VideoCapture(captureIndex)
        break
    except:
        print("Could not set up on channel " + str(captureIndex))
        captureIndex += 1
print("Finished setting up camera on channel " + str(captureIndex))

state = 0
image = None
shapes = [None]

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

    # state machine, at some point
    result = image
    if state == 0: # sunglasses
        result = sunglassesFilter(image, shapes[0])
    elif state == 1: # pond
        result = pondFilter(image, shapes[0])

    # show image
    cv2.imshow("Output", result)

    # process shortcuts
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc
        break
    elif k == 44: # <
        state = max(0, state-1)
    elif k == 46: # >
        state = min(1, state+1)

cv2.destroyAllWindows()
cap.release()
