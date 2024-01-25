import cv2
# import numpy as np
import math
# from utils import drawFace, resize, transform_image, overlay_images
from utils import apply, drawFace
# from imutils import face_utils

sunglassessrc = cv2.imread("media/sunglasses.png", cv2.IMREAD_UNCHANGED)

def sunglassesFilter(image, shapes):
    global sunglassessrc
    
    (height, width, _) = image.shape
    
    for shape in shapes:
        image = apply(
            image=image, 
            overlaysrc=sunglassessrc, 
            shape=shape,
            center=28,
            translation=(0, 0, 0)
        )
    
    return image