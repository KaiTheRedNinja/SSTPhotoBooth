import cv2
from utils import apply

medalssrc = cv2.imread("media/medals.png", cv2.IMREAD_UNCHANGED)

def medalsFilter(image, shapes):
    global medalssrc
    
    (height, width, _) = image.shape
    
    for shape in shapes:
        image = apply(
            image=image,
            overlaysrc=medalssrc,
            shape=shape,
            center=28,
            scale=4,
            translation=(0, 1.5, 0)
        )
    
    return image