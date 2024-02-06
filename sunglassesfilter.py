import cv2
from utils import apply

sunglassessrc = cv2.imread("media/sunglasses.png", cv2.IMREAD_UNCHANGED)

def sunglassesFilter(image, shapes):
    global sunglassessrc
    
    for shape in shapes:
        image = apply(
            image=image, 
            overlaysrc=sunglassessrc, 
            shape=shape,
            center=28, # a point on the nose
            translation=(0, 0, 0)
        )
    
    return image