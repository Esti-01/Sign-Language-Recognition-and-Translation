# Sign Language Recognition and Translation

This project aims to assist mute people in communicating with others by recognizing and translating American Sign Language (ASL) gestures using computer vision techniques. The code has been developed using Python 3.11.9.

# Project OneDrive Link:

## Project Structure

The project consists of the following files:

- `collect_images.py`: Collects image data for each ASL letter and stores them in the `data` directory.
- `create_dataset.py`: Creates a dataset using the collected images and saves it as a pickle file.
- `train_dataset.py`: Trains a machine learning model using the created dataset and saves the trained model as a pickle file.
- `test_dataset.py`: Tests the trained model on live video feed from the camera.
- `user_interface.py`: Provides a user interface for real-time sign language recognition and translation.
- `test_suite.py`: Runs a test suite to ensure the functionality of different components of the project.
- `requirements.txt`: Lists all the required Python libraries and their versions.

## Virtual Environment Setup

To ensure a consistent development environment across different devices, it is recommended to use a virtual environment. Follow the instructions below to set up a virtual environment for your device:

### Windows

1. Open a command prompt or PowerShell.
2. Navigate to the project directory.
3. Run the following command to create a virtual environment:
   python -m venv venv
4. Activate the virtual environment:
   venv\Scripts\activate
5. To deactivate the virtual environment:
   deactivate

### macOS and Linux

1. Open a terminal.
2. Navigate to the project directory.
3. Run the following command to create a virtual environment:
   python3.11 -m venv venv
4. Activate the virtual environment:
   source venv/bin/activate
5. To deactivate the virtual environment:
   deactivate
   
## Dependencies

The project relies on the following dependencies:

- OpenCV (cv2) for image processing and camera input.
- MediaPipe for hand landmark detection.
- Scikit-learn for machine learning model training and evaluation.
- NumPy for numerical operations.
- Matplotlib and Seaborn for data visualization.

Make sure to install the required dependencies before running the project.

## Installing Dependencies

After setting up the virtual environment, you need to install the required Python libraries. The project includes a `requirements.txt` file that lists all the necessary dependencies.

To install the dependencies, run the following command:
pip install -r requirements.txt

This command will install all the libraries specified in the `requirements.txt` file within the virtual environment.

## Usage

1. Run `collect_images.py` to collect image data for each ASL letter. Follow the prompts to capture images for each letter.

2. Run `create_dataset.py` to create a dataset using the collected images. The dataset will be saved as a pickle file.

3. Run `train_dataset.py` to train a machine learning model using the created dataset. The trained model will be saved as a pickle file.

4. Run `test_dataset.py` to test the trained model on live video feed from the camera. The script will open the default camera and display the recognized ASL letter.

5. Run `user_interface.py` to launch the user interface for real-time sign language recognition and translation. The script will open the camera and display the recognized ASL letter and the translated sentence.

6. Run `test_suite.py` to execute the test suite and ensure the functionality of different components of the project.

## Model Training

The project uses a Random Forest classifier for recognizing ASL gestures. The model is trained on the dataset created using the collected images. The training process involves extracting hand landmarks using MediaPipe and training the classifier on the extracted features.

## User Interface

The user interface provides real-time sign language recognition and translation. It captures video feed from the camera, detects hand landmarks using MediaPipe, and predicts the corresponding ASL letter using the trained model. The recognized letters are then concatenated to form a sentence, which is displayed on the screen.

The user interface also includes functionality to handle special gestures such as "space" for adding a space between words and "del" for deleting the last recognized letter.

## Testing

The project includes a test suite (`test_suite.py`) that runs various tests to ensure the functionality of different components. The tests cover data collection, dataset creation, model training, model testing, and the user interface.

To run the test suite, execute `test_suite.py`. The test results will be displayed in the console.

## Contributions

Contributions to the project are welcome. If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
