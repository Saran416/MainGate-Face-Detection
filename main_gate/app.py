import cv2
import time
from spoof_checker import Checker
from name_fetcher import Fetcher

import warnings
warnings.filterwarnings("ignore")

checker = Checker()
fetcher = Fetcher(load_vectors=False)
# fetcher.save_vectors('./img_vectors.pkl') 


# fetcher.save_to_db()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

prev = 0
next = 0
fps = 0
num = 0
prev_time = 0
name = None
buffer = 3
color = (0, 0, 255)

prev_name = None

while True:
    # Capture frame from the webcam
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    image = cv2.flip(image, 1)
    image, foundface, face = checker.idle(image)


    current_time = time.time()
    if current_time - prev_time >= buffer:
        
        if foundface:
            name = fetcher.fetch_name(face)
        else:
            name = "No face detected"
        
        if not name:
            name = "Invalid Person"
          
        
        if name != prev_name or color == (0, 0, 255):
            is_real = checker.check_spoof()

        if is_real:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
                
        prev_time = current_time
        prev_name = name 

    # Calculate the position for the text
    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

    # Calculate the position for the text
    cv2.putText(image, name, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # Show the output image
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Average FPS:", fps / num)
        break

    # Calculate FPS
    prev = next
    next = time.time()
    fps += 1 / (next - prev)
    num += 1

# Release resources
cv2.destroyAllWindows()
cap.release()
