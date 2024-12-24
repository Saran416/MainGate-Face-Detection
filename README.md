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

### `face.ipynb`
- Jupyter Notebook for:
  1. Running face verification.
  2. Creating the database (currently implemented using a dictionary).

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
1. Open the `face.ipynb` notebook.
2. Follow the instructions within the notebook to:
   - Verify faces.
   - Manage the database.

---

## Future Improvements
1. Develop a final integration pipeline in `.py` format.
2. Implement a more robust database solution.
3. Train and integrate a face verification model.
4. Explore data augmentation techniques for improved model performance.