import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_and_train_model():
    # Create a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # For 10 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Dummy data for demonstration
    X_train = np.random.rand(100, 224, 224, 3)  # 100 random images
    y_train = np.random.randint(0, 10, 100)     # Random labels

    # Train the model
    model.fit(X_train, y_train, epochs=5)

    # Save the model
    model.save('model/model.h5')
    print("Model saved to model/model.h5")

if __name__ == "__main__":
    create_and_train_model()
