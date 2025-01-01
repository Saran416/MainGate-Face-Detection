import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os 

class Cropping:
    def __init__(self, offset_w=20, offset_h=20):
        """
        Initialize the cropping utility.
        :param offset_w: Percentage width offset for cropping.
        :param offset_h: Percentage height offset for cropping.
        """
        self.offset_w = offset_w
        self.offset_h = offset_h
        self.detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    def crop_face(self, image):
        """
        Detects face in the image and crops them with offsets.

        :param image: The input image (as a numpy array).
        :return: List of cropped face images.
        """
        if image is None:
            raise ValueError("Invalid image input. Please provide a valid image.")
        
        # Detect faces in the image
        img, bboxs = self.detector.findFaces(image, draw=False)

        if not bboxs:
            print("No faces detected in the image.")
            return 
        for i, bbox in enumerate(bboxs):
            # Get bounding box coordinates and size
            x, y, w, h = bbox['bbox']

            offsetW = (self.offset_w / 100) * w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            offsetH = (self.offset_h / 100) * h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 3.5)

            # Crop the region of interest
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face


