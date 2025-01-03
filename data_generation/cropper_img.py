from dlib import get_frontal_face_detector
import cv2

class Cropping:
    def __init__(self, offset_w=20, offset_h=20):
        """
        Initialize the cropping utility.
        :param offset_w: Percentage width offset for cropping.
        :param offset_h: Percentage height offset for cropping.
        """
        self.offset_w = offset_w
        self.offset_h = offset_h
        self.detector = get_frontal_face_detector()

    def crop_face(self, image):
        if image is None:
            raise ValueError("Invalid image input. Please provide a valid image.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = self.detector(gray, 0)
        if not faces:
            # print("No faces found in the image.")
            return None

        # Process only the first detected face
        face = max(faces, key=lambda face: face.width() * face.height())

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        # Crop the face from the original image
        cropped_face = image[y:y+h, x:x+w]

        # print(type(cropped_face))
        return cropped_face


