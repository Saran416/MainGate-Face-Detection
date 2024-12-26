import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os

class Cropper:
    def __init__(self, offset_w=25, offset_h=25):
        """
        Initialize the cropping utility.
        :param image_path: Path to the input image.
        :param output_directory: Directory to save cropped faces.
        :param offset_w: Percentage width offset for cropping.
        :param offset_h: Percentage height offset for cropping.
        """
        self.offset_w = offset_w
        self.offset_h = offset_h
        self.detector = FaceDetector(minDetectionCon=0.5, modelSelection=0) # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)

    def crop(self, image):
        """
        Detects faces in the image and crops them with offsets.
        """
        # detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
        image, bboxs = self.detector.findFaces(image, draw=False)
        if not bboxs:
            return None
        bbox = bboxs[0]
        x, y, w, h = bbox['bbox']
        offset_w = 30
        offset_h = 30
        offsetW = (offset_w / 100) * w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        offsetH = (offset_h / 100) * h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 3.5)
        image = image[y:y + h, x:x + w]
        return image

