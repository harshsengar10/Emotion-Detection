import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Argument parser: choose whether to train or display emotions

parser = argparse.ArgumentParser(description="Facial Emotion Recognition System")
parser.add_argument("--mode", choices=["train", "display"], required=True,
                    help="Select mode: 'train' for training the model, 'display' for real-time emotion detection.")
args = parser.parse_args()
mode = args.mode


# Plot accuracy and loss curves

def plot_training_curves(history):
    """Plot training and validation accuracy/loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


# Image Augmentation + Normalization

train_dir = 'data/train'
val_dir = 'data/test'

batch_size = 32
epochs = 50
num_train = 28709
num_val = 7178

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# CNN Model Architecture

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Output layer for 7 emotions
])

model.summary()


# TRAINING PHASE

if mode == "train":
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4, decay=1e-6),
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=num_val // batch_size
    )

    plot_training_curves(history)

    # Save model and weights
    model.save('emotion_model.h5')
    model.save_weights('emotion_weights.h5')

    print("\n✅ Model training completed and saved successfully!")


# REAL-TIME EMOTION DETECTION

elif mode == "display":
    model.load_weights('emotion_model.h5')
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)
    print("🎥 Press 'q' to quit the video window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

            prediction = model.predict(cropped_img)
            emotion_idx = int(np.argmax(prediction))
            emotion_label = emotion_dict[emotion_idx]

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y - 40), (x + w, y + h + 10), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Facial Emotion Recognition', cv2.resize(frame, (1000, 700)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
