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

The final project is in the Final Directory.

It has

1. **app.py**

- This is the main file to run the streamlit app.
- It captures the image using the webcam.
- crops the face using Face Detection.
- saves the image in a MongoDB database.

```bash
streamlit run app.py
```

2. **recognition.ipynb**

3. **crop_image.py**

- This file contains the function to crop the face from the image.

## Future Improvements

2. Implement a more robust database solution.
3. Train and integrate a face verification model.
4. Explore data augmentation techniques for improved model performance.
