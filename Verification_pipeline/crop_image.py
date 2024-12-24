import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os

class Cropping:
    def __init__(self, image_path, output_directory,name, offset_w=25, offset_h=25):
        """
        Initialize the cropping utility.
        :param image_path: Path to the input image.
        :param output_directory: Directory to save cropped faces.
        :param offset_w: Percentage width offset for cropping.
        :param offset_h: Percentage height offset for cropping.
        """
        self.image_path = image_path
        self.output_directory = output_directory
        self.name = name
        self.offset_w = offset_w
        self.offset_h = offset_h
        self.detector = FaceDetector(minDetectionCon=0.5, modelSelection=0) # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)

        self._ensure_output_directory()

    def _ensure_output_directory(self):
        """
        Ensures the output directory exists.
        """
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def crop_faces(self):
        """
        Detects faces in the image and crops them with offsets.
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Image not found at path: {self.image_path}")
        
        img, bboxs = self.detector.findFaces(img, draw=False)

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


            image_name = self.name+"_cropped.jpg"
            img_path = os.path.join(self.output_directory, image_name)
            cv2.imwrite(img_path, cropped_face)
            print(f"Cropped face saved at: {img_path}")
        return img_path

if __name__ == "__main__":
    # Parameters
    image_path = "/Users/harshsingh/Desktop/projects/face/indian celebrities dataset/data/aadhi/220px_Aadhi_Pinisetty_at_Maragatha_Naanayam_audio_launch.jpg"
    output_directory = "/Users/harshsingh/Desktop/projects/face/Data_pipeline"
    name = "random"
    offset_percentage_w = 20  # 30% width offset
    offset_percentage_h =  20# 30% height offset

    # Initialize the cropping utility
    cropper = Cropping(image_path, output_directory, name , offset_percentage_w, offset_percentage_h)
    
    # Perform face cropping
    cropper.crop_faces()



# image_path = "/Users/harshsingh/Desktop/face/Database/Harsh/Harsh_withoutcrop.jpg"  # Replace with the actual path to the image
# output_directory = "./cropped_faces/"  
# offsetPercentageW = 30
# offsetPercentageH = 30


# # Initialize the FaceDetector object
# # minDetectionCon: Minimum detection confidence threshold
# # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
# detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# # Read the image
# img = cv2.imread(image_path)

# # Detect faces in the image
# # img: Updated image
# # bboxs: List of bounding boxes around detected faces
# img, bboxs = detector.findFaces(img, draw=False)

# # Check if any face is detected
# if bboxs:
#     for i, bbox in enumerate(bboxs):

#         # Get bounding box coordinates and size
#         x, y, w, h = bbox['bbox']


#         # Adjust the bounding box with offsets
#         offsetW = (offsetPercentageW / 100) * w
#         x = int(x - offsetW)
#         w = int(w + offsetW * 2)
#         offsetH = (offsetPercentageH / 100) * h
#         y = int(y - offsetH * 3)
#         h = int(h + offsetH * 3.5)

#         # Crop the region of interest
#         cropped_face = img[y:y + h, x:x + w]
#         cropped_face_path = f"{output_directory}face_{i + 1}.jpg"
#         cv2.imwrite(cropped_face_path, cropped_face)
#         print(f"Cropped face saved at: {cropped_face_path}")
