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
            return False
        for i, bbox in enumerate(bboxs):
            # Get bounding box coordinates and size
            x, y, w, h = bbox['bbox']

            offsetW = (self.offset_w / 100) * w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            offsetH = (self.offset_h / 100) * h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 3.5)

            # Check if the bounding box is within the image dimensions
            img_height, img_width = img.shape[:2]
            if x < 0 or y < 0:
                return False  # Top-left corner of the bounding box is outside the image
            if x + w > img_width or y + h > img_height:
                return False
            # Crop the region of interest
            
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face, [x, y, w, h]


# Example Usage
if __name__ == "__main__":
    # Load an image
    input_image_path = "/Users/harshsingh/Desktop/projects/face/test.jpeg"
    input_image = cv2.imread(input_image_path)

    # Initialize the cropping class
    face_cropper = Cropping()
    # Crop faces
    cropped_face = face_cropper.crop_face(input_image)
    print("sa")
    cv2.imwrite("/Users/harshsingh/Desktop/projects/face/cropped_img.jpeg",cropped_face)
    print("Saved")
    cv2.destroyAllWindows()
