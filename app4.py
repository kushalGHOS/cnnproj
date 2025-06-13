import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from gtts import gTTS
from playsound import playsound
import tempfile
import os


model = load_model('isl_gesture_recognition_cnn.h5')


train_dir = 'archive (2)/asl_alphabet_train/asl_alphabet_train'


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
)


class_indices = train_generator.class_indices
index_to_letter = {v: k for k, v in class_indices.items()}



def speak_letter(letter):
    tts = gTTS(text=letter, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    playsound(audio_path)
    os.remove(audio_path) 

cap = cv2.VideoCapture(0)
prev_class = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3))

    predictions = model.predict(reshaped_frame)
    predicted_class = np.argmax(predictions)
    predicted_letter = index_to_letter.get(predicted_class, "Unknown")

    if predicted_letter != prev_class:
        print(f"Predicted Letter: {predicted_letter}")
        speak_letter(predicted_letter)
        prev_class = predicted_letter

    cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('ISL Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
