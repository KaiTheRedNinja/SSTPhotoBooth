import cv2
import numpy as np
from utils import resize_with_translation, get_face_polygon, overlay_images

pondimgsrc = cv2.imread("media/pond.jpeg")

def pondFilter(image, shapes):
    global pondimgsrc

    (height, width, _) = image.shape
    if image.size != pondimgsrc.size:
        pondimgsrc = cv2.resize(pondimgsrc, (width, height))
    pondimg = pondimgsrc.copy()

    # For each detected face, find the landmark.
    for shape in shapes:
        # projected top of the face
        face_polygon = get_face_polygon(shape)

        # extract face
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [face_polygon], -1, (255,255,255), -1, cv2.LINE_AA)
        face = cv2.bitwise_and(image, image, mask=mask)

        # resize it half size, but keep it where it is
        mask = resize_with_translation(mask, (width/2, height/2), 0.5)
        face = resize_with_translation(face, (width/2, height/2), 0.5)

        # apply the face and mask
        reverse_mask = cv2.bitwise_not(mask)
        pondimg = cv2.bitwise_and(pondimg, pondimg, mask=reverse_mask)
        pondimg = cv2.bitwise_or(pondimg, face)
    return pondimg
