import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for label in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, label)):
        data_aux = []
        x_ = []
        y_ = []
        
        img = cv2.imread(os.path.join(DATA_DIR, label, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
        
        data.append(data_aux)
        labels.append(label)

# Visualize data distribution
label_counts = np.unique(labels, return_counts=True)
classes = label_counts[0]
counts = label_counts[1]

plt.figure(figsize=(8, 6))
plt.bar(classes, counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Data Distribution')
plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()