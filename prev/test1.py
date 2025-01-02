from name_fetcher import Fetcher
from cropper_img import Cropping
import cv2

fetcher = Fetcher(db_name="optimized_image_db")
cropper = Cropping()

img = cv2.imread('./data/a k hangal/15795100.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cropper.crop_face(img)
cv2.imshow("Cropped", img)
print(fetcher.fetch_name(img))