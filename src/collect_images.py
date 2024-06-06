import os
import cv2
import subprocess
import string

# Directory setup
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parameters
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del']
dataset_size = 100

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Open an image in the current directory
image_name = "images/ASL Alphabet.jpeg"  # Change to your desired image name
image_path = os.path.join(".", image_name)

if os.path.exists(image_path):
    # Open the image in the default image viewer
    try:
        if os.name == 'posix':  # For Mac and Linux
            subprocess.Popen(["open", image_path])
        elif os.name == 'nt':  # For Windows
            subprocess.Popen(["start", image_path], shell=True)
    except Exception as e:
        print(f"Failed to open the reference image: {str(e)}")
else:
    print("Reference image not found.")

# Data collection loop
for label in labels:
    # Ensure label directory exists
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f'Collecting data for label: {label}')
    print('Press "Q" to start capturing images. Press "Delete" to exit.')

    # Wait for 'Q' to start or 'Delete' to exit
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from the camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start, "Delete" to exit.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == 127:  # "Delete" key to exit on MacBook
            cap.release()
            cv2.destroyAllWindows()
            print("Exiting the program.")
            exit()  # Exits the entire script

    # Start capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from the camera.")
            break

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == 127:  # "Delete" key to exit on MacBook
            break

        # Save the captured image
        image_path = os.path.join(label_dir, f'{label}_{counter}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")