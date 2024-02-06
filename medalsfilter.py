import cv2
from utils import apply

medalssrc = cv2.imread("media/medals.png", cv2.IMREAD_UNCHANGED)

def medalsFilter(image, shapes):
    global medalssrc
    
    (height, width, _) = image.shape
    
    for shape in shapes:
        try:
            image = apply(
                image=image,
                overlaysrc=medalssrc,
                shape=shape,
                center=28,
                scale=4,
                translation=(0, 1.5, 0)
            )
        except:
            (refx, refy) = shape[0]
            cv2.putText(image, "Move further", org=(refx, refy+20), fontFace=0, fontScale=2, thickness=5, color=(255, 255, 255))
    
    return image