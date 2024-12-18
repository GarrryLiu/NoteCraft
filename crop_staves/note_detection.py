import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import cv2
import os
import json

# Load dataset
def load_dataset(data_dir):
    """
    Load images and corresponding labels (note head coordinates) for training.
    
    Parameters:
    - image_dir: Directory containing cropped stave images.
    - label_file: JSON file containing the note coordinates for each image.
    - img_size: Tuple specifying the size to which each image will be resized.

    Returns:
    - X: Array of images (normalized).
    - y: Array of corresponding note coordinates (x, y).
    """
    X = []
    y = []
    for piece in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, piece, "scores", piece + "_ly", "coords/notes_01.npy")
        image_dir = os.path.join(data_dir, piece, "scores", piece + "_ly", "img/01.png")

        try: 
            image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
            labels = np.load(label_dir)
        except:
            continue
        X.append(image)
        y.append(labels)

    return X, y

# Define CNN model
def create_cnn_model(input_shape=(128, 128, 1)):
    """
    Create a CNN model to predict note head coordinates.
    
    Parameters:
    - input_shape: Shape of the input images.

    Returns:
    - model: A compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(2 * max_notes, activation='linear')  # Output 2 * max_notes (x, y) coordinates
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the CNN model to predict note coordinates.
    
    Parameters:
    - model: The CNN model to train.
    - X_train, y_train: Training images and corresponding labels.
    - X_val, y_val: Validation images and corresponding labels.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.

    Returns:
    - history: Training history.
    """
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size)
    return history

# Predict note coordinates
def predict_notes(model, image_path, img_size=(128, 128)):
    """
    Predict note head coordinates for a given image.
    
    Parameters:
    - model: The trained CNN model.
    - image_path: Path to the image to predict.
    - img_size: Tuple specifying the size to which the image will be resized.

    Returns:
    - Predicted coordinates of note heads.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, img_size) / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    predictions = model.predict(image)[0]  # Output shape is (2 * max_notes,)
    note_coordinates = predictions.reshape(-1, 2)  # Reshape into (num_notes, 2)
    return note_coordinates

# Main script to load data, train, and predict
if __name__ == "__main__":
    # Load data
    data_dir = "data/msmd_aug_v1-1_no-audio"
    X, y = load_dataset(data_dir)

    # Split data into training and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create and train the model
    max_notes = 10  # Adjust based on your dataset
    model = create_cnn_model(input_shape=(128, 128, 1))
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=50)

    # Save the model
    model.save("note_detection_model.h5")

    # Predict on a new image
    test_image_path = "path/to/test_image.png"
    predicted_coords = predict_notes(model, test_image_path)
    print("Predicted Note Coordinates:", predicted_coords)
