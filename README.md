# Face Verification System

## Setup
### Create a Conda Environment
Run the following command to create a new Conda environment named `tf2`:
```bash
conda create --name tf2 python==3.9.21
```

### Install Requirements
Use the provided `requirements.txt` file to install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** If you encounter any missing modules while running the project, install them using pip and update the `requirements.txt` file accordingly.

---

## Project Structure

### Data Pipeline Folder
This folder contains the following files:

- **`test.py`**
  - Captures images from Streamlit using the webcam.
  - **Known Issue:** Occasionally captures and stores completely black images due to an infinite loop issue.

- **`crop_image.py`**
  - Crops images to a specified size, adjustable using height and width offsets.

- **`webcam_capture.py`**
  - Integrates the functionality of image capture via Streamlit and cropping to restrict the image to the face.
  - Saves the cropped image to the `Database` folder under a subfolder named after the user-provided name in Streamlit.

### `Recognition_pipeline`
1. Pipeline to run Recognition task. The database will be build using a dictionary, will later migrate to db.
2. Dictionary will take 2.5 seconds to build for a 45 images.

###  `Verification_pipeline`
1. For verification. Similar to Recognition pipeline.

---

## Steps to Run

### Data Collection
1. Navigate to the `Data_pipeline` folder.
2. Run the following command to start the Streamlit app for data collection:
   ```bash
   streamlit run webcam_capture.py
   ```
3. Use the app to capture and save images.

### Face Verification
1. Configure the recognition.py properly, by giving suitable input image, offsets, dictionary, etc.
2. use:
  ```bash 
  python recognition.py
  ```

---

## Future Improvements
2. Implement a more robust database solution.
3. Train and integrate a face verification model.
4. Explore data augmentation techniques for improved model performance.