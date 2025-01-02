import math
import time
import os
import cv2
import cvzone
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# Confidence threshold
confidence = 0.85

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model
model = YOLO("./spoof_detection/l_version_1_300.pt")

# Class names for prediction
classNames = ["real", "fake"]

# Initialize variables
prev_frame_time = 0  # Initial value to save the first frame
buffer = 5  # Buffer time in seconds
first = True


# Ensure 'images' directory exists
os.makedirs("images", exist_ok=True)

try:
    while True:
        # Read frame from webcam
        success, img = cap.read()
        img.flip(1)  # Flip the image horizontally
        if not success:
            print("Failed to read from webcam.")
            break

        # Predict with YOLO model
        results = model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                if conf > confidence:
                    if classNames[cls] == 'real':
                        color = (0, 255, 0)
                        # print("Real face detected.", time.time())
                        # Save the image if the buffer time has elapsed
                        if (time.time() - prev_frame_time > buffer) or first:
                            print("Saving image...")
                            cropped_img = img[y1:y2, x1:x2]  # Crop the detected region
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"images/{timestamp}.jpg", cropped_img)
                            prev_frame_time = time.time()
                            first = False
                            
                    else:
                        color = (0, 0, 255)

                    # Draw bounding box and text
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(
                        img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                        (max(0, x1), max(35, y1)), scale=2, thickness=4,
                        colorR=color, colorB=color
                    )

        # Display the frame
        cv2.imshow("Image", img)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
