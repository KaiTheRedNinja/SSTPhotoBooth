import cv2
import numpy as np
import math
from imutils import face_utils

# for debugging purposes
FACIAL_LANDMARKS_COLORS = {
    "mouth":(255, 0, 0),
    "inner_mouth":(0, 100, 0),
    "right_eyebrow":(0, 100, 0),
    "left_eyebrow":(0, 100, 0),
    "right_eye":(0, 0, 255),
    "left_eye":(0, 255, 0),
    "nose":(0, 255, 255),
    "jaw":(255, 255, 0)
}

"""
Draws the points of the face on the image
"""
def drawFace(image, shape, includeProjectedTop=False):
    # For each detected face, find the landmark.
    for (name, points) in face_utils.FACIAL_LANDMARKS_68_IDXS.items():
        for index in range(points[0], points[1]):
            (x, y) = shape[index]
            cv2.circle(
                img=image,
                center=(x, y),
                radius=6,
                color=(FACIAL_LANDMARKS_COLORS[name]),
                thickness=-1
            )
            image = cv2.putText(img=image, text=str(index), org=(x+10, y+10), fontFace=0, fontScale=0.3, color=FACIAL_LANDMARKS_COLORS[name])

    # if we include the projected top, draw that too
    if not includeProjectedTop: return

    projected = project_face_top(shape)
    for (x, y) in projected:
        cv2.circle(
            img=image,
            center=(x, y),
            radius=6,
            color=(100,100,100),
            thickness=-1
        )

"""
Estimates the points for the top of the head from a 68 point shape.

Projects the points of the jaw around the center of the face
"""
def project_face_top(shape):
    # get the range of points considered as the jaw
    jawrange = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
    # add those to the face polygon first
    projected = []

    # get the center of the face
    (ref1x, ref1y) = shape[jawrange[0]]
    (ref2x, ref2y) = shape[jawrange[1]-1]
    (centerx, centery) = (int((ref1x+ref2x)/2), int((ref1y+ref2y)/2))

    # rotate each point on the jaw 180 degrees, pivoting on the center of the face
    # add those rotated points to the face polygon list
    # these provide a rough but decently accurate estimate for the top half of the face
    for index in range(jawrange[0], jawrange[1]):
        (x, y) = shape[index]
        pos = (centerx*2-x, centery*2-y)
        projected.append(pos)

    return projected

"""
Gets the points for a polygon, outlining the face of the user
"""
def get_face_polygon(shape):
    # get the range of points considered as the jaw
    jawrange = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
    # add those to the face polygon first
    face_polygon = [shape[x] for x in range(jawrange[0]+1, jawrange[1]-1)]
    # add the projected points
    face_polygon += project_face_top(shape)
    # turn into numpy array
    face_polygon = np.array(face_polygon)
    return face_polygon

"""
Resizes an image to a given scale
"""
def resize(image, scale):
    # Get the height and width of the original image
    height, width = image.shape[0], image.shape[1]

    # Calculate the new dimensions after resizing
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

"""
Resizes an image, while keeping the center at a given position
"""
def resize_with_translation(image, center, scale):
    # Get the height and width of the original image
    height, width = image.shape[0], image.shape[1]

    # Resize the image
    resized_image = resize(image, scale)

    # Calculate the shift required to keep the center at the same position
    shift_x = int(center[0] - (center[0] * scale))
    shift_y = int(center[1] - (center[1] * scale))

    # Create a translation matrix for the shift
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply the translation to the resized image
    moved_image = cv2.warpAffine(resized_image, translation_matrix, (width, height))

    return moved_image

"""
Overlays a 4-channel image on top of another 3 or 4 channel image
"""
def overlay_images(original, overlay):
    # Extract the alpha channel from the 4-channel image
    alpha_channel = overlay[:, :, 3]

    # Convert the alpha channel to a 3-channel mask
    alpha_mask = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)

    # Invert the alpha mask (1 for transparent, 0 for opaque)
    inverted_alpha_mask = 1 - alpha_mask / 255.0

    # Multiply the original image by the inverted alpha mask
    background = original * inverted_alpha_mask

    # Multiply the overlay image by the alpha mask
    foreground = overlay[:, :, :3] * (alpha_mask / 255.0)

    # Add the two images to get the final result
    result = cv2.add(background, foreground)

    return result.astype(np.uint8)

"""
Transforms an image by moving it to a given point, rotating it a given angle, and resizing the image to a target size
"""
def transform_image(image, target_point, angle, target_size):
    # Get the original image size
    original_height, original_width = image.shape[:2]

    # Calculate the top-left corner coordinates of the new image
    new_width, new_height = target_size
    top_left_x = (new_width - original_width) // 2
    top_left_y = (new_height - original_height) // 2

    # Create a blank canvas with transparency (4 channels)
    canvas = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # Copy the original image to the center of the new canvas (including the alpha channel)
    canvas[top_left_y:top_left_y + original_height, top_left_x:top_left_x + original_width, :] = image

    # Rotate the canvas by the specified angle
    rotation_matrix = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), -angle, 1)
    rotated_canvas = cv2.warpAffine(canvas, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Calculate the translation needed to move the center to the target point
    translate_x = target_point[0] - (target_size[0]/2)
    translate_y = target_point[1] - (target_size[1]/2)

    # Translate the canvas to move the center to the target point
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    translated_rotated_canvas = cv2.warpAffine(rotated_canvas, translation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return translated_rotated_canvas

"""
Applies an overlay on top of an image, with reference with positions on the face.

image: The base image. Will be modified.
overlaysrc: The source image for the overlay. Will not be modified.
shape: The shape of the face
ref1, ref2: Used as reference for how to scale, angle, and position the overlay. Defaults to 0 and 16.
center: A point on `shape`. Defaults to None. When specified, this point will be used as the center.
scale: A manual adjust for the scale. Defaults to 1.
translation: A manual adjust in the format (x, y, angle) relative to the face.
             x and y are NOT in the coordinate system of the image, it is with reference to the face.
             1 = 1 face width.
"""
def apply(image, overlaysrc, shape, ref1=0, ref2=16, center=None, scale=1, translation=(0,0,0)):
    def pythagorean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def midpoint(point1, point2):
        return (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))

    def angle_of_points(point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.degrees(math.atan2(dy, dx))

    def rotate_point(point, angle_degrees):
        x, y = point
        angle_radians = math.radians(angle_degrees)
        x_rotated = x * math.cos(angle_radians) - y * math.sin(angle_radians)
        y_rotated = x * math.sin(angle_radians) + y * math.cos(angle_radians)
        return x_rotated, y_rotated

    (height, width, _) = image.shape

    # obtain width, center, and angle info
    refWidth = pythagorean_distance(shape[ref1], shape[ref2])
    if center==None:
        center = midpoint(shape[ref1], shape[ref2])
    else:
        center = shape[center]
    angle = angle_of_points(shape[ref1], shape[ref2])

    # translate the image
    overlay = resize(overlaysrc, refWidth/overlaysrc.shape[1] * scale)
    overlay = transform_image(overlay, center, angle+translation[2], (width, height))

    # translate the manual translations to the image coordinate system
    jawrange = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
    (ref1x, ref1y) = shape[jawrange[0]]
    (ref2x, ref2y) = shape[jawrange[1]-1]
    (centerx, centery) = (int((ref1x+ref2x)/2), int((ref1y+ref2y)/2))
    face_width = pythagorean_distance((ref1x, ref1y), (centerx, centery))*2
    finalTranslation = rotate_point((translation[0]*face_width, translation[1]*face_width), translation[2])

    # apply manual translations
    translation_matrix = np.float32([[1, 0, finalTranslation[0]], [0, 1, finalTranslation[1]]])
    # Apply the translation to the image
    translated_overlay = cv2.warpAffine(overlay, translation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # put the overlay on top of the image
    return overlay_images(image, translated_overlay)
