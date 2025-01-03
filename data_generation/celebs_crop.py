### Don't mind this, this was just a dummy code for me to crop the images of the indian celebrities dataset. 

import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os
from cropper_img import Cropping

# Define the directories
base_dir = "./data"
go_to_dir = "./cropped_data"

# Ensure the output directory exists
os.makedirs(go_to_dir, exist_ok=True)
cropper = Cropping(0,0)

# Loop through each directory in base_dir
for name in sorted(os.listdir(base_dir)):
    name_dir = os.path.join(base_dir, name)
    
    # Check if it's a directory
    if os.path.isdir(name_dir):
        # Create corresponding directory in cropped_data
        new_dir = os.path.join(go_to_dir, name)
        os.makedirs(new_dir, exist_ok=True)
        
        # Loop through each file in the directory
        for image in os.listdir(name_dir):
            img_path = os.path.join(name_dir, image)
            
            # Check if the file is a valid image
            if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Call the cropping function
                img = cv2.imread(img_path)
                cropped_img = cropper.crop_face(img)
                if cropped_img is not None:
                    cv2.imwrite(os.path.join(new_dir, image), cropped_img)
                else:
                    print("No face found in the image", str(os.path.join(new_dir, image)))
            else:
                print(f"Skipped non-image file: {img_path}")

print("Done")
