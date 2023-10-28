import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize the Mediapipe Hands object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)


# Function to load data from files and subFolders
def load_data_from_folders(data_folder):
    gestures = []
    labels = []

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        data = []
                        for line in file:
                            x, y, z = map(float, line.strip().split())
                            data.extend([x, y, z])
                        gestures.append(data)
                        labels.append(folder_name)

    return np.array(gestures), np.array(labels)


data_folder = "hand_shape_data"
gestures, labels = load_data_from_folders(data_folder)

X_train, X_test, y_train, y_test = train_test_split(
    gestures, labels, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data = []
            for landmark in hand_landmarks.landmark:
                data.extend([landmark.x, landmark.y, landmark.z])
            data = np.array(data).reshape(1, -1)

            predicted_gesture = model.predict(data)[0]

            cv2.putText(
                frame,
                f"Gesture: {predicted_gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
