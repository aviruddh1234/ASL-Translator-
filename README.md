# ✋ Hand Sign Language Detection System (26 Alphabets)

## Project Overview

This project implements a real-time hand sign language detection system capable of recognizing all 26 alphabets. Leveraging computer vision and machine learning techniques, the system detects hand gestures from a webcam feed and classifies them into their corresponding alphabetical signs. This solution offers a foundational tool for bridging communication gaps, particularly for American Sign Language (ASL) learners or as a component in more complex human-computer interaction systems.

The core idea involves two main stages:
1.  **Hand Detection & Preprocessing:** Accurately locating the hand within a video frame and preparing the image for classification by normalizing its size and aspect ratio.
2.  **Sign Classification:** Using a trained machine learning model to classify the preprocessed hand image into one of the 26 alphabet signs.

## Features

* **Real-time Detection:** Processes live video input from a webcam.
* **26-Alphabet Support:** Robustly recognizes all 26 letters of the alphabet (A-Z) in hand sign form.
* **Data Collection Script:** Includes a script to efficiently capture and prepare custom datasets for each sign.
* **Aspect Ratio Normalization:** Ensures consistent input for the classification model by intelligently adding padding to hand images.
* **Modular Design:** Separate scripts for data collection and real-time testing, promoting ease of development and debugging.
* **Leverages Pre-trained Models (via Teachable Machine):** Streamlines the model training process using accessible platforms.

## Technologies Used

* **Python 3.x**
* **OpenCV (`cv2`)**: For real-time video capture, image processing, and drawing.
* **`cvzone`**: A wrapper library built on OpenCV and MediaPipe, simplifying common computer vision tasks like hand detection.
* **`mediapipe`**: For robust and accurate hand landmark detection.
* **`numpy`**: For numerical operations, especially array manipulation.
* **`tensorflow` / `keras`**: For loading and running the trained machine learning model.
* **Google's Teachable Machine**: Used as a user-friendly platform for training the image classification model.

## Getting Started

Follow these steps to set up and run the Hand Sign Detection system on your local machine.

### Prerequisites

Before you begin, ensure you have:
* Python 3.x installed.
* A webcam connected to your computer.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Hand-Sign-Language-Detection.git](https://github.com/your-username/Hand-Sign-Language-Detection.git)
    cd Hand-Sign-Language-Detection
    ```
    (Remember to replace `your-username` with your actual GitHub username.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install opencv-python cvzone mediapipe numpy tensorflow
    ```

### Dataset Generation (Data Collection)

You will need to generate a dataset of hand signs for all 26 alphabets.

1.  **Create directories for each sign:**
    Inside your project folder, create a directory named `Data`. Within `Data`, create 26 subdirectories, one for each alphabet (e.g., `A`, `B`, `C`, ..., `Z`).

    ```
    Hand-Sign-Language-Detection/
    ├── Data/
    │   ├── A/
    │   ├── B/
    │   ├── C/
    │   └── ... (up to Z)
    ├── HandSignDetector_DataCollection.py
    ├── HandSignDetector_Test.py
    └── README.md
    ```

2.  **Run the data collection script:**
    ```bash
    python HandSignDetector_DataCollection.py
    ```
    This script will open your webcam. Follow the on-screen instructions:
    * Position your hand to make the sign for 'A' within the bounding box.
    * Press the 's' key repeatedly to capture multiple images (aim for 200-300 images per sign).
    * Switch to the next alphabet directory (e.g., 'B') in your code/terminal when you are ready for the next sign. **You will need to manually adjust the `folder` variable in the script to point to the correct alphabet folder (e.g., `folder = "Data/B"`) before collecting images for the next sign.**
    * Repeat for all 26 alphabets.
    * Press 'q' to quit the script.

### Model Training (Using Google's Teachable Machine)

Once you have collected your dataset for all 26 alphabets:

1.  Go to [Google's Teachable Machine website](https://teachablemachine.withgoogle.com/train/image).
2.  Create a new "Image Project" -> "Standard image model".
3.  **Add classes:** For each of the 26 alphabets (A-Z), create a new class.
4.  **Upload images:** For each class, upload all the images you collected for that specific alphabet from your `Data/` directory.
5.  **Train Model:** Click the "Train Model" button. This may take some time depending on the size of your dataset.
6.  **Export Model:**
    * Once training is complete, click "Export Model".
    * Select the "TensorFlow" tab and then "Keras".
    * Click "Download my model".
    * This will download a `.zip` file containing `keras_model.h5` (your trained model) and `labels.txt` (the list of your classes).
7.  **Place Model Files:** Extract the downloaded files (`keras_model.h5` and `labels.txt`) and place them in your project's root directory.

### Running the Hand Sign Detection System

After collecting data and training your model:

1.  **Ensure model files are in place:** Make sure `keras_model.h5` and `labels.txt` are in the same directory as `HandSignDetector_Test.py`.
2.  **Run the test script:**
    ```bash
    python HandSignDetector_Test.py
    ```
    This will open your webcam feed. The system will now detect your hand signs and display the predicted alphabet in real-time.

## Project Structure

Hand-Sign-Language-Detection/
├── Data/                     # Stores collected images for each alphabet
│   ├── A/
│   ├── B/
│   └── ... (up to Z)
├── keras_model.h5          # Your trained Keras model file
├── labels.txt              # Labels corresponding to your model (A, B, C, ...)
├── HandSignDetector_DataCollection.py # Script for collecting hand sign images
├── HandSignDetector_Test.py         # Script for real-time hand sign detection
└── README.md                        # This file
