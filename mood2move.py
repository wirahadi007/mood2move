import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('models/keras_model.h5')
mood_labels = ['Happy', 'Sad', 'Neutral']

def predict_mood(image):
    image = cv2.resize(image, (224, 224))  # Adjust image size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    mood_index = np.argmax(predictions)
    return mood_labels[mood_index]

def recommend_activity(mood):
    activities = {
        "Happy": "Try a high-energy workout like running or dancing!",
        "Sad": "Yoga or a calming walk can lift your mood.",
        "Neutral": "Go for a light jog or a refreshing walk!"
    }
    return activities.get(mood, "Take some time to relax!")

# Start webcam feed
cap = cv2.VideoCapture(2)  # Change to appropriate index if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mood = predict_mood(frame_rgb)
        activity = recommend_activity(mood)

        cv2.putText(frame, f'Mood: {mood}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, activity, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Webcam Image", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII for the ESC key
            break

cap.release()
cv2.destroyAllWindows()
