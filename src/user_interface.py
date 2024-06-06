import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import time
from gtts import gTTS
from playsound3 import playsound

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading the model: {str(e)}")
    exit(1)

cap = cv2.VideoCapture(1)  # Use 0 for the built-in camera on MacBook
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
    'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
    'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'space': 'space', 'del': 'del'
}
labels_dict_inv = {v: k for k, v in labels_dict.items()}

predicted_sentence = ""
last_prediction_time = time.time()

delete_timer = None
delete_delay = 1.5  # Adjust this value to change the delete delay (in seconds)
prediction_delay = 3  # Adjust this value to change the prediction delay (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from the camera")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirrored effect
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            min_x = min(x_)
            min_y = min(y_)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Pad the input data to match the expected number of features
            data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
            data_aux = data_aux.reshape(1, -1)  # Reshape the data to match the model's input shape

            try:
                prediction = model.predict(data_aux)
                predicted_character = labels_dict_inv[prediction[0]]
            except (ValueError, KeyError) as e:
                print(f"Error during prediction: {str(e)}")
                continue

            current_time = time.time()

            if predicted_character == 'space':
                if not predicted_sentence.endswith(' '):
                    predicted_sentence += ' '
                last_prediction_time = current_time
            elif predicted_character == 'del':
                if delete_timer is None:
                    delete_timer = current_time
                elif current_time - delete_timer >= delete_delay:
                    predicted_sentence = predicted_sentence[:-1]
                    delete_timer = current_time
            elif current_time - last_prediction_time >= prediction_delay:
                predicted_sentence += predicted_character
                last_prediction_time = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Create a semi-transparent overlay for the predicted sentence
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Display the predicted sentence on the overlay
    cv2.putText(frame, "Predicted Sentence: " + predicted_sentence, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the predicted sentence to a file
try:
    with open("predicted_sentence.txt", "w") as file:
        file.write(predicted_sentence)
except IOError as e:
    print(f"Error saving predicted sentence: {str(e)}")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Convert the predicted sentence to speech
try:
    with open("predicted_sentence.txt", "r") as file:
        predicted_sentence = file.read().strip()
        if predicted_sentence:
            tts = gTTS(text=predicted_sentence, lang='en')
            tts.save("predicted_sentence.mp3")
            playsound("predicted_sentence.mp3")
            os.remove("predicted_sentence.mp3")
        else:
            print("No predicted sentence found.")
except IOError as e:
    print(f"Error reading predicted sentence: {str(e)}")