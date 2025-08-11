import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

# Directories
TRAIN_DIR = '/home/aneebaaslam/inference/interface/interface/archive/train'
TEST_DIR = '/home/aneebaaslam/inference/interface/interface/archive/test'

# Function to create a DataFrame for image paths and labels
def create_dataframe(directory):
    image_paths = []
    labels = []
    for label in os.listdir(directory):  # Loop through each class folder
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for image in os.listdir(class_dir):  # Loop through images in the class folder
                image_paths.append(os.path.join(class_dir, image))
                labels.append(label)
            print(f"{label} completed")  # Log progress for each class
    return pd.DataFrame({'image': image_paths, 'label': labels})

# Function to preprocess images and labels
def preprocess_images(dataframe, target_size=(48, 48)):
    images = []
    label_categories = dataframe['label'].astype('category')  # Convert labels to numeric
    labels = label_categories.cat.codes  # Convert labels to numeric codes
    label_mapping = dict(zip(label_categories.cat.categories, range(len(label_categories.cat.categories))))  # Map labels to codes

    # Print the label mapping
    print("Label Mapping:")
    for label, code in label_mapping.items():
        print(f"{label} -> {code}")
    
    for img_path in dataframe['image']:
        img = load_img(img_path, target_size=target_size)  # Load and resize image
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
    return np.array(images), to_categorical(labels)

# Load training data
train_df = create_dataframe(TRAIN_DIR)

# Preprocess training data
X, y = preprocess_images(train_df)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')  # Output layer with # of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=35,
    verbose=3
)

# Save the trained model
model.save("my_trained_model_2.keras")

# Evaluate the model on the test dataset (if available)
if os.path.exists(TEST_DIR):
    test_df = create_dataframe(TEST_DIR)
    X_test, y_test = preprocess_images(test_df)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Optional: Visualize Training History
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()