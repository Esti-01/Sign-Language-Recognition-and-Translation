import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from pickle file with error handling
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    raise Exception("Data file not found. Make sure './data.pickle' exists.")

# Check that the data has consistent length
data = np.array(data_dict['data'], dtype=object)
labels = np.array(data_dict['labels'], dtype=object)
if len(data) != len(labels):
    raise Exception("Data and labels have different lengths.")

# Pad the data arrays to ensure consistent lengths
max_length = max(len(x) for x in data)
data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data])

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels,
)

# Train the model with Random Forest
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_features='sqrt',  # Number of features to consider at each split
    random_state=42,  # Ensure reproducibility
)
model.fit(x_train, y_train)

# Prediction and evaluation
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Classification report
print(classification_report(y_test, y_predict))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)